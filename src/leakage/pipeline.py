"""End-to-end run: snapshot -> contract -> vendors -> candidates -> scorer -> forensics.

All outputs land in one DuckDB file (data/processed/leakage.duckdb) plus metrics.json,
which the report, the Excel review pack and the dashboard all read. Nothing downstream
recomputes numbers, so the README and the app can't disagree.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import duckdb
import pandas as pd

from leakage import forensics
from leakage.contract import standardise_chicago
from leakage.duplicates import (
    FEATURES,
    TIER_LABELS,
    CandidateConfig,
    excluded_payee,
    generate_candidates,
    pair_features,
    prepare,
)
from leakage.ingest import CHICAGO_PAYMENTS, ingest, latest_snapshot, load_snapshot
from leakage.scorer import InjectionConfig, build_training_set, fit_final, train_and_evaluate
from leakage.vendors import evaluate_resolution, resolve_vendors

log = logging.getLogger(__name__)

QUEUE_COLUMNS = [
    "a_id", "b_id", "tier", "tier_label", "risk_score", "exposure", "expected_value",
    "day_gap", "a_check_date", "b_check_date", "a_amount", "b_amount",
    "a_voucher_number", "b_voucher_number", "a_vendor_name_raw", "b_vendor_name_raw",
    "a_department", "b_department", "a_contract_number", "b_contract_number", "vendor_id",
    "excluded",
]


@dataclass
class RunConfig:
    raw_root: Path = Path("data/raw")
    out_dir: Path = Path("data/processed")
    offline: bool = False
    candidates: CandidateConfig = field(default_factory=CandidateConfig)
    injection: InjectionConfig = field(default_factory=InjectionConfig)
    split_thresholds: tuple[int, ...] = (2_500, 5_000, 10_000)
    er_eval_n: int = 500


def _snapshot(cfg: RunConfig) -> Path:
    if cfg.offline:
        snap = latest_snapshot(cfg.raw_root, CHICAGO_PAYMENTS)
        if snap is None:
            raise SystemExit("no local snapshot; run without --offline first")
        return snap
    return ingest(CHICAGO_PAYMENTS, cfg.raw_root)


def run(cfg: RunConfig) -> dict:
    t0 = time.time()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    snap = _snapshot(cfg)
    manifest = json.loads((snap / "manifest.json").read_text())
    payments = standardise_chicago(load_snapshot(snap))
    log.info("payments: %d rows", len(payments))

    name_map, merges = resolve_vendors(payments["vendor_name_raw"])
    er_eval = evaluate_resolution(payments["vendor_name_raw"], n=cfg.er_eval_n)
    log.info("vendors: %d ids, %d fuzzy merges", name_map["vendor_id"].nunique(), len(merges))

    p = prepare(payments, name_map, cfg.candidates)
    cands = generate_candidates(p, cfg.candidates)
    feats = pair_features(cands, p)
    log.info("candidates: %d", len(feats))

    train, gen_recall, n_injected = build_training_set(
        payments, name_map, cfg.candidates, cfg.injection
    )
    eval_metrics, _ = train_and_evaluate(train)
    model = fit_final(train)
    feats["risk_score"] = model.predict_proba(feats[FEATURES])[:, 1]
    feats["tier_label"] = feats["tier"].map(TIER_LABELS)
    # Exposure = the smaller side of the pair: what you'd recover if one payment is a dupe.
    feats["exposure"] = feats["pair_amount"]
    feats["expected_value"] = feats["risk_score"] * feats["exposure"]
    feats["vendor_id"] = feats["a_vendor_id"]
    feats["excluded"] = excluded_payee(feats["a_vendor_name_raw"]) | excluded_payee(
        feats["b_vendor_name_raw"])
    queue = feats[QUEUE_COLUMNS].sort_values("risk_score", ascending=False).reset_index(drop=True)

    eligible = p.copy()
    benford_overall = forensics.benford_mad(eligible["amount"])
    benford_dept = forensics.benford_by_group(eligible)
    dv = eligible[eligible["contract_number"] == "DV"]
    clustering = forensics.threshold_clustering(dv)
    splits = pd.concat(
        [forensics.split_purchases(dv, t) for t in cfg.split_thresholds], ignore_index=True
    )
    spikes = forensics.spend_spikes(eligible[~excluded_payee(eligible["vendor_name_raw"])])

    vendors = (
        name_map.groupby(["vendor_id", "canonical_name"], as_index=False)
        .agg(raw_names=("vendor_name_raw", "nunique"), n_payments=("n_payments", "sum"))
    )
    spend = payments.merge(name_map[["vendor_name_raw", "vendor_id"]], on="vendor_name_raw")
    vendors = vendors.merge(
        spend.groupby("vendor_id", as_index=False)["amount"].sum().rename(
            columns={"amount": "total_paid"}), on="vendor_id")

    in_scope = queue[~queue["excluded"]]
    tiers = in_scope.groupby("tier").agg(pairs=("a_id", "size"), exposure=("exposure", "sum"))
    metrics = {
        "source": {
            "dataset": manifest["resource_url"],
            "snapshot_utc": manifest["snapshot_utc"],
            "raw_rows": manifest["total_rows"],
        },
        "ledger": {
            "payments": len(payments),
            "voucher_level": int((~payments["is_annual_rollup"]).sum()),
            "annual_rollups": int(payments["is_annual_rollup"].sum()),
            "eligible_for_tests": len(p),
            "total_paid": float(payments["amount"].sum()),
            "min_date": str(payments["check_date"].min().date()),
            "max_date": str(payments["check_date"].max().date()),
        },
        "vendors": {
            "raw_names": int(name_map["vendor_name_raw"].nunique()),
            "normalised_names": int(name_map["norm_name"].nunique()),
            "resolved_vendors": int(name_map["vendor_id"].nunique()),
            "fuzzy_merges": len(merges),
            "perturbation_recall": er_eval["recall"],
            "perturbation_n": er_eval["n_injected"],
        },
        "candidates": {
            "pairs": len(queue),
            "excluded_pairs": int(queue["excluded"].sum()),
            "excluded_exposure": float(queue.loc[queue["excluded"], "exposure"].sum()),
            "by_tier": {k: {"pairs": int(v.pairs), "exposure": float(v.exposure)}
                        for k, v in tiers.iterrows()},
        },
        "scorer": {
            "injected_duplicates": n_injected,
            "candidate_generation_recall": gen_recall,
            **eval_metrics,
        },
        "forensics": {
            "benford_overall_mad": benford_overall[0],
            "benford_overall_band": forensics.mad_band(benford_overall[0]),
            "benford_departments_tested": len(benford_dept),
            "benford_departments_nonconforming": int(
                (benford_dept["conformity"] == "nonconformity").sum()),
            "threshold_max_lift": float(clustering["lift"].max()),
            "split_windows": {int(t): int((splits["threshold"] == t).sum())
                              for t in cfg.split_thresholds},
            "spikes": len(spikes),
        },
        "config": {
            "candidates": asdict(cfg.candidates),
            "injection": asdict(cfg.injection),
        },
        "runtime_seconds": round(time.time() - t0, 1),
    }

    db = cfg.out_dir / "leakage.duckdb"
    db.unlink(missing_ok=True)
    con = duckdb.connect(str(db))
    for name, df in {
        "payments": payments, "vendor_map": name_map, "vendors": vendors,
        "vendor_merges": merges, "review_queue": queue, "benford_departments": benford_dept,
        "threshold_clustering": clustering, "split_purchases": splits.drop(
            columns=["payment_ids"], errors="ignore"), "spend_spikes": spikes,
    }.items():
        con.register("tmp", df)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM tmp")
        con.unregister("tmp")
    con.close()
    (cfg.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    log.info("done in %.0fs", metrics["runtime_seconds"])
    return metrics
