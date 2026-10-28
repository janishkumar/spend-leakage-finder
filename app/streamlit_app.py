"""Reviewer dashboard over the pipeline's outputs. Reads only; never recomputes."""

from __future__ import annotations

import json
import os
from pathlib import Path

import altair as alt
import duckdb
import numpy as np
import pandas as pd
import streamlit as st

from leakage.forensics import BENFORD
from leakage.report import RoiAssumptions, review_set, sensitivity

DATA = Path(os.environ.get("LEAKAGE_DATA", "data/processed"))
st.set_page_config(page_title="Spend Leakage Finder", layout="wide")


@st.cache_resource
def connect() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(DATA / "leakage.duckdb"), read_only=True)


@st.cache_data
def table(name: str) -> pd.DataFrame:
    return connect().execute(f"SELECT * FROM {name}").df()


def money(x: float) -> str:
    for cut, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(x) >= cut:
            return f"${x / cut:,.1f}{suffix}"
    return f"${x:,.0f}"


if not (DATA / "metrics.json").exists():
    st.error(f"No pipeline output in {DATA}. Run `make run` first.")
    st.stop()

m = json.loads((DATA / "metrics.json").read_text())
queue = table("review_queue")

st.title("Spend Leakage Finder")
st.caption(
    f"City of Chicago vendor payments · snapshot {m['source']['snapshot_utc']} · "
    "every item here is a candidate for review, not a confirmed overpayment"
)

overview, review, vendors_tab, forensics_tab = st.tabs(
    ["Overview", "Review queue", "Vendor master", "Forensic tests"]
)

with overview:
    c = st.columns(4)
    c[0].metric("Payments", f"{m['ledger']['payments']:,}", money(m["ledger"]["total_paid"]))
    c[1].metric("Resolved vendors", f"{m['vendors']['resolved_vendors']:,}",
                f"from {m['vendors']['raw_names']:,} spellings", delta_color="off")
    c[2].metric("Candidate pairs", f"{m['candidates']['pairs']:,}",
                f"{m['candidates']['excluded_pairs']:,} transfers scoped out",
                delta_color="off")
    c[3].metric("Ranking precision, top 100", f"{m['scorer']['precision_at_100_model']:.0%}",
                f"rule baseline {m['scorer']['precision_at_100_rule_baseline']:.0%}",
                delta_color="off")

    st.subheader("What would a review be worth?")
    st.caption("Assumptions, not findings. Only a real review measures the confirmation rate.")
    s = st.columns(4)
    top_k = s[0].slider("Pairs reviewed", 50, 2000, 250, 50)
    rate = s[1].slider("Share that confirm", 0.0, 0.25, 0.02, 0.005, format="%.3f")
    minutes = s[2].slider("Minutes per review", 5, 60, 15, 5)
    hourly = s[3].slider("Reviewer $/hour", 40, 250, 85, 5)
    a = RoiAssumptions(top_k, rate, minutes, hourly)
    sens = sensitivity(queue, a)
    reviewed = review_set(queue, top_k)
    exposure = float(reviewed["exposure"].sum())
    cost = len(reviewed) * minutes / 60 * hourly
    r = st.columns(3)
    r[0].metric("Exposure reviewed", money(exposure))
    r[1].metric("Estimated recoverable", money(exposure * rate))
    r[2].metric("Net of review cost", money(exposure * rate - cost))
    st.dataframe(
        sens.assign(confirmation_rate=sens["confirmation_rate"].map("{:.0%}".format)),
        hide_index=True, use_container_width=True,
    )

with review:
    f = st.columns(4)
    tiers = f[0].multiselect("Tier", ["T2", "T3", "T4", "T1"], default=["T2", "T3", "T4"])
    min_score = f[1].slider("Min risk score", 0.0, 1.0, 0.8, 0.05)
    show_excluded = f[2].checkbox("Show scoped-out transfers", value=False)
    search = f[3].text_input("Vendor contains")
    q = queue[queue["tier"].isin(tiers) & (queue["risk_score"] >= min_score)]
    if not show_excluded:
        q = q[~q["excluded"]]
    if search:
        q = q[q["a_vendor_name_raw"].str.contains(search, case=False, na=False)]
    st.caption(f"{len(q):,} pairs · {money(q['exposure'].sum())} exposure (double-counts)")
    cols = ["risk_score", "tier", "a_vendor_name_raw", "b_vendor_name_raw", "a_check_date",
            "b_check_date", "a_amount", "b_amount", "a_voucher_number", "b_voucher_number",
            "a_department", "a_contract_number", "day_gap"]
    st.dataframe(q[cols].head(2000), hide_index=True, use_container_width=True,
                 column_config={"risk_score": st.column_config.ProgressColumn(
                     "risk", min_value=0.0, max_value=1.0, format="%.2f")})
    st.download_button("Download filtered queue (CSV)", q.to_csv(index=False),
                       "review_queue.csv", "text/csv")

with vendors_tab:
    v = table("vendors")
    merges = table("vendor_merges")
    st.markdown(
        f"**{m['vendors']['raw_names']:,}** payee spellings → "
        f"**{m['vendors']['resolved_vendors']:,}** vendors. Fuzzy matching is tuned for "
        f"precision: {m['vendors']['perturbation_recall']:.0%} recall on injected name "
        "variants, because a false merge manufactures false duplicates."
    )
    st.subheader("Same payment, vendor set up under two names (T3)")
    t3 = queue[(queue["tier"] == "T3") & ~queue["excluded"]]
    st.dataframe(t3[["a_vendor_name_raw", "b_vendor_name_raw", "a_check_date", "a_amount",
                     "a_voucher_number", "b_voucher_number", "risk_score"]],
                 hide_index=True, use_container_width=True)
    st.subheader("Fuzzy merges (lowest similarity first)")
    st.dataframe(merges.sort_values("ratio"), hide_index=True, use_container_width=True)
    st.subheader("Vendors with the most spellings")
    st.dataframe(v.sort_values("raw_names", ascending=False).head(50), hide_index=True,
                 use_container_width=True)

with forensics_tab:
    b = table("benford_departments")
    st.subheader("Benford's law, first digit")
    dept = st.selectbox("Department", b["department"].tolist())
    row = b[b["department"] == dept].iloc[0]
    chart_df = pd.DataFrame({
        "digit": np.arange(1, 10),
        "observed": [row[f"d{i}"] for i in range(1, 10)],
        "expected": BENFORD,
    })
    bars = alt.Chart(chart_df).mark_bar().encode(
        x="digit:O", y=alt.Y("observed:Q", axis=alt.Axis(format="%")))
    line = alt.Chart(chart_df).mark_line(point=True, color="#888").encode(
        x="digit:O", y="expected:Q")
    st.altair_chart(bars + line, use_container_width=True)
    st.caption(f"MAD {row['mad']:.4f} · {row['conformity']} · n = {int(row['n']):,}. "
               "Fixed fees and grants break Benford legitimately.")
    st.dataframe(b[["department", "n", "mad", "conformity"]], hide_index=True,
                 use_container_width=True)

    st.subheader("Bunching under approval limits (direct vouchers)")
    st.dataframe(table("threshold_clustering"), hide_index=True, use_container_width=True)
    st.subheader("Possible split purchases")
    st.dataframe(table("split_purchases").head(500), hide_index=True, use_container_width=True)
    st.subheader("Vendor-month spend spikes")
    spikes = table("spend_spikes").merge(
        table("vendors")[["vendor_id", "canonical_name"]], on="vendor_id")
    st.dataframe(spikes.head(500), hide_index=True, use_container_width=True)
