"""Outputs a reviewer actually uses: an Excel review pack and a findings memo.

Every number in docs/findings.md is read from metrics.json / leakage.duckdb produced by
`leakage run`. Nothing here is typed by hand.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

from leakage.vendors import is_org


@dataclass(frozen=True)
class RoiAssumptions:
    """Illustrative defaults. Change them; the point is that they're visible."""

    top_k: int = 250
    confirmation_rate: float = 0.02
    minutes_per_review: float = 15.0
    hourly_rate: float = 85.0


def review_set(queue: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """The top-k in-scope pairs by risk score, one row per suspect payment.

    T1 (identical rows, usually split lines) and excluded payees are left out, and a
    payment that appears in several pairs is counted once, so exposure isn't inflated.
    """
    pool = queue[(queue["tier"] != "T1") & (~queue["excluded"])]
    pool = pool.sort_values("risk_score", ascending=False).drop_duplicates("b_id")
    return pool.head(top_k)


def roi(queue: pd.DataFrame, a: RoiAssumptions) -> dict:
    top = review_set(queue, a.top_k)
    exposure = float(top["exposure"].sum())
    recoverable = exposure * a.confirmation_rate
    cost = len(top) * a.minutes_per_review / 60 * a.hourly_rate
    return {
        "reviewed": len(top),
        "exposure_reviewed": exposure,
        "est_recoverable": recoverable,
        "review_cost": cost,
        "net": recoverable - cost,
    }


SENSITIVITY_RATES = (0.01, 0.02, 0.05, 0.10)
BUNCHING_LIFT = 1.3


def sensitivity(queue: pd.DataFrame, a: RoiAssumptions) -> pd.DataFrame:
    rows = []
    for rate in SENSITIVITY_RATES:
        r = roi(queue, RoiAssumptions(a.top_k, rate, a.minutes_per_review, a.hourly_rate))
        rows.append({"confirmation_rate": rate, **r})
    return pd.DataFrame(rows)


def public_name(name: str) -> str:
    """Mask payees who look like private individuals in anything published.

    The ledger is public, but a findings memo that puts a person's name next to
    "suspected duplicate payment" isn't fair to them. Organisations stay named.
    The Excel pack, which a client's reviewers use internally, keeps full names.
    """
    return name if is_org(name) else "[individual payee]"


def _money(x: float) -> str:
    if abs(x) >= 1e9:
        return f"${x / 1e9:,.2f}B"
    if abs(x) >= 1e6:
        return f"${x / 1e6:,.1f}M"
    if abs(x) >= 1e3:
        return f"${x / 1e3:,.0f}K"
    return f"${x:,.0f}"


def load(out_dir: Path) -> tuple[dict, duckdb.DuckDBPyConnection]:
    metrics = json.loads((out_dir / "metrics.json").read_text())
    con = duckdb.connect(str(out_dir / "leakage.duckdb"), read_only=True)
    return metrics, con


def write_review_pack(out_dir: Path, path: Path, a: RoiAssumptions | None = None) -> Path:
    a = a or RoiAssumptions()
    metrics, con = load(out_dir)
    queue = con.execute("SELECT * FROM review_queue").df()
    top = review_set(queue, a.top_k)
    sheets = {
        "Read me": pd.DataFrame({"": [
            "Spend Leakage Finder: review pack",
            f"Source: {metrics['source']['dataset']} "
            f"(snapshot {metrics['source']['snapshot_utc']})",
            "Every row is a CANDIDATE for review, not a confirmed overpayment.",
            "risk_score ranks pairs; it is not a calibrated probability.",
            "exposure = the smaller payment in the pair (what one duplicate would cost).",
            "Mark the 'reviewer_decision' column: confirmed / legitimate / needs info.",
        ]}),
        "Duplicate candidates": top.assign(reviewer_decision="", reviewer_notes=""),
        "Vendor master dupes": queue[(queue["tier"] == "T3") & (~queue["excluded"])],
        "Exact duplicate rows": queue[queue["tier"] == "T1"],
        "Vendor merges": con.execute(
            "SELECT * FROM vendor_merges ORDER BY ratio").df(),
        "Split purchases": con.execute(
            "SELECT * FROM split_purchases ORDER BY window_total DESC LIMIT 500").df(),
        "Benford by dept": con.execute("SELECT * FROM benford_departments").df(),
        "Assumptions": pd.DataFrame(
            [{"assumption": k, "value": v} for k, v in a.__dict__.items()]
        ),
        "ROI sensitivity": sensitivity(queue, a),
    }
    con.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name[:31], index=False)
            ws = xw.sheets[name[:31]]
            ws.freeze_panes = "A2"
            for col in ws.columns:
                width = max(len(str(c.value or "")) for c in col[:200])
                ws.column_dimensions[col[0].column_letter].width = min(max(width + 2, 10), 60)
    return path


def write_findings(out_dir: Path, path: Path, a: RoiAssumptions | None = None) -> Path:
    a = a or RoiAssumptions()
    m, con = load(out_dir)
    queue = con.execute("SELECT * FROM review_queue").df()
    sens = sensitivity(queue, a)
    top10 = review_set(queue, 10)
    clustering = con.execute("SELECT * FROM threshold_clustering").df()
    benford = con.execute(
        "SELECT department, n, mad, conformity FROM benford_departments LIMIT 6").df()
    spikes = con.execute("""
        SELECT v.canonical_name, s.month, s.amount, s.median
        FROM spend_spikes s JOIN vendors v USING (vendor_id)
        ORDER BY s.robust_z DESC LIMIT 5""").df()
    con.close()

    lifts = ", ".join(
        f"${int(t):,}: {x:.2f}"
        for t, x in zip(clustering["threshold"], clustering["lift"], strict=True)
    )
    bunching = (
        "possible bunching, worth a sample review"
        if m["forensics"]["threshold_max_lift"] >= BUNCHING_LIFT
        else "none found"
    )
    L, V, C, S, F = m["ledger"], m["vendors"], m["candidates"], m["scorer"], m["forensics"]
    tiers = C["by_tier"]

    def tier_row(t: str, label: str) -> str:
        d = tiers.get(t, {"pairs": 0, "exposure": 0.0})
        return f"| {t} | {label} | {d['pairs']:,} | {_money(d['exposure'])} |"

    lines = [
        "# Findings: City of Chicago vendor payments",
        "",
        f"_Generated by `leakage report` from snapshot {m['source']['snapshot_utc']} "
        f"of {m['source']['dataset']}. Do not edit by hand; re-run the pipeline._",
        "",
        "## The ledger",
        "",
        f"- {L['payments']:,} payment rows, {_money(L['total_paid'])} paid, "
        f"{L['min_date']} to {L['max_date']}.",
        f"- {L['voucher_level']:,} are voucher-level payments; {L['annual_rollups']:,} are "
        "annual roll-ups (year only, no voucher), which the detectors skip.",
        f"- {L['eligible_for_tests']:,} payments of $100+ with a full date go through the tests.",
        "",
        "## Vendor master",
        "",
        f"- {V['raw_names']:,} distinct payee spellings → {V['normalised_names']:,} after "
        f"normalisation → **{V['resolved_vendors']:,} resolved vendors** "
        f"({V['fuzzy_merges']:,} fuzzy merges).",
        f"- Recall on {V['perturbation_n']} injected name variants: "
        f"**{V['perturbation_recall']:.0%}**. Tuned for precision on purpose: a false merge "
        "manufactures false duplicate candidates, a missed merge only loses some.",
        f"- {tiers.get('T3', {'pairs': 0})['pairs']} same-day, same-amount payment pairs "
        "went to one vendor under two different names (vendor-master duplicates).",
        "",
        "## Duplicate-payment candidates",
        "",
        "| Tier | Pattern | Pairs | Exposure |",
        "|---|---|---:|---:|",
        tier_row("T1", "Exact duplicate row (likely split lines; data-quality flag)"),
        tier_row("T2", "Same vendor + amount, different voucher, ≤30 days"),
        tier_row("T3", "Same amount + day, vendor under two names"),
        tier_row("T4", "Near or transposed amount, same vendor, ≤14 days"),
        "",
        "Exposure sums the smaller payment of every pair, so it double-counts payments in "
        "several pairs and is mostly recurring, legitimate spend. It is a ceiling, not a finding.",
        "",
        "### How good is the ranking?",
        "",
        f"No public ledger labels its duplicates, so {S['injected_duplicates']:,} realistic "
        "duplicates were injected (re-keyed voucher, shifted date, transposed digits, "
        "vendor-name typos) and the pipeline was scored on finding them, holding out "
        "30% of vendors.",
        "",
        "| | Ranking model | Rule baseline (same vendor + amount, closest date) |",
        "|---|---:|---:|",
        f"| PR-AUC | **{S['pr_auc_model']:.2f}** | {S['pr_auc_rule_baseline']:.2f} |",
        f"| Precision in top 100 | **{S['precision_at_100_model']:.0%}** | "
        f"{S['precision_at_100_rule_baseline']:.0%} |",
        f"| Precision in top 500 | **{S['precision_at_500_model']:.0%}** | "
        f"{S['precision_at_500_rule_baseline']:.0%} |",
        "",
        f"- Candidate generation catches **{S['candidate_generation_recall']:.0%}** of injected "
        "duplicates; misses are mostly vendor-name typos that entity resolution refuses to merge.",
        "- Precision is a lower bound: a few 'negatives' are probably real duplicates.",
        "",
        "### Top 10 pairs by risk score",
        "",
        f"Payees scoped out as statutory or financial transfers (pension funds, bond "
        f"trustees, tax authorities): {C['excluded_pairs']:,} pairs, "
        f"{_money(C['excluded_exposure'])} of exposure. They sat at the top of an "
        "exposure-weighted list on the first run; any AP review would exclude them.",
        "",
        "",
        "| Vendor (as keyed) | Dates | Amounts | Vouchers | Tier | Score |",
        "|---|---|---|---|---|---:|",
    ]
    for r_ in top10.itertuples(index=False):
        a_name, b_name = public_name(r_.a_vendor_name_raw), public_name(r_.b_vendor_name_raw)
        names = a_name if a_name == b_name else f"{a_name} / {b_name}"
        lines.append(
            f"| {names} | {pd.Timestamp(r_.a_check_date).date()} / "
            f"{pd.Timestamp(r_.b_check_date).date()} | ${r_.a_amount:,.2f} / "
            f"${r_.b_amount:,.2f} | {r_.a_voucher_number} / {r_.b_voucher_number} | "
            f"{r_.tier} | {r_.risk_score:.2f} |"
        )
    lines += [
        "",
        "## Forensic tests",
        "",
        f"- **Benford's law:** overall first-digit MAD {F['benford_overall_mad']:.4f} "
        f"({F['benford_overall_band']}). {F['benford_departments_nonconforming']} of "
        f"{F['benford_departments_tested']} departments with 1,000+ payments are nonconforming:",
        "",
        "| Department | Payments | MAD | Band |",
        "|---|---:|---:|---|",
        *[f"| {b.department} | {b.n:,} | {b.mad:.4f} | {b.conformity} |"
          for b in benford.itertuples(index=False)],
        "",
        "  Fixed fees, grants and standard prices break Benford legitimately; nonconformity "
        "says where to look, not that anything is wrong.",
        f"- **Bunching under approval limits:** {bunching}. The highest lift against control "
        f"points was {F['threshold_max_lift']:.2f}× ({lifts}).",
        "- **Split purchases** (direct vouchers, ≥2 payments to one vendor within 7 days "
        "summing over a limit): "
        + ", ".join(f"{n:,} windows at ${int(t):,}" for t, n in F["split_windows"].items())
        + ". Most are routine repeat orders; the pack sorts them by total for review.",
        f"- **Spend spikes:** {F['spikes']:,} vendor-months at 6+ robust z-scores against the "
        "vendor's own history (transfer payees scoped out). The largest deviations, usually "
        "one-off capital or settlement payments worth a look:",
        "",
        *[f"  - {public_name(s.canonical_name)}: {_money(s.amount)} in "
          f"{pd.Timestamp(s.month):%b %Y} (typical month {_money(s.median)})"
          for s in spikes.itertuples(index=False)],
        "",
        "## What a review would be worth (illustrative)",
        "",
        f"Review the top {a.top_k} in-scope suspect payments by risk score "
        f"({_money(sens['exposure_reviewed'].iloc[0])} of exposure) at "
        f"{a.minutes_per_review:.0f} minutes each and ${a.hourly_rate:.0f}/hour "
        f"({_money(sens['review_cost'].iloc[0])} of review time). What it returns depends "
        "on how many turn out to be real duplicates, which only the review can tell you:",
        "",
        "| If this share confirms | Recoverable | Net of review cost |",
        "|---:|---:|---:|",
        *[f"| {s_.confirmation_rate:.0%} | {_money(s_.est_recoverable)} | {_money(s_.net)} |"
          for s_ in sens.itertuples(index=False)],
        "",
        "These are assumptions, not findings. The first job with a client is to review a "
        "sample and measure the confirmation rate.",
        "",
        f"_Pipeline runtime: {m['runtime_seconds']:.0f}s on a laptop._",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return path
