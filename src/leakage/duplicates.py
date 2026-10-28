"""Duplicate-payment candidate generation and pair features.

Tiers (a pair gets the first tier it qualifies for):
  T1  exact duplicate rows: same voucher, vendor, amount, date, department, contract.
      Public ledgers carry no line numbers, so these are often legitimate split lines.
  T3  same amount, same day, *different raw vendor names* that resolve to one vendor.
      This is the vendor-master-duplicate pattern: one supplier set up twice.
  T2  same resolved vendor, same amount, different voucher, within `window_days`.
  T4  same resolved vendor, different voucher, within `near_days`, amount within
      `near_pct` or the same digits transposed (keying error).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

TIER_LABELS = {
    "T1": "Exact duplicate row",
    "T2": "Same vendor + amount, different voucher",
    "T3": "Same amount + day, vendor set up under two names",
    "T4": "Near / transposed amount, same vendor",
}


# Payees an AP duplicate review normally scopes out: statutory pension contributions,
# bond-trustee debt service and tax remittances are paid in large, identical, same-day
# instalments by design. A client would maintain this list; these are Chicago's.
EXCLUDED_PAYEES = re.compile(
    r"PENSION|ANNUITY|RETIREMENT|A ?& ?B FUND|BENEFIT FUND|\bTRUST(?:EE)?S?\b|\bBANK\d*\b"
    r"|INTERNAL REVENUE|TREASURER|COMPTROLLER|ILLINOIS DEPARTMENT OF REVENUE",
)


def excluded_payee(names: pd.Series) -> pd.Series:
    return names.fillna("").str.upper().str.contains(EXCLUDED_PAYEES)


@dataclass(frozen=True)
class CandidateConfig:
    min_amount: float = 100.0
    window_days: int = 30
    near_days: int = 14
    near_pct: float = 0.005


def digits_key(amount: float) -> str:
    cents = int(round(abs(amount) * 100))
    return "".join(sorted(str(cents)))


def prepare(payments: pd.DataFrame, name_map: pd.DataFrame, cfg: CandidateConfig) -> pd.DataFrame:
    """Eligible payments joined to their resolved vendor, with vendor-level stats."""
    p = payments[
        (~payments["is_annual_rollup"])
        & payments["check_date"].notna()
        & (payments["amount"] >= cfg.min_amount)
    ].merge(name_map[["vendor_name_raw", "vendor_id"]], on="vendor_name_raw", how="left")
    p["digits_key"] = p["amount"].map(digits_key)
    p["amount_cents"] = (p["amount"] * 100).round().astype("int64")
    stats = payments.merge(name_map[["vendor_name_raw", "vendor_id"]], on="vendor_name_raw")
    stats = stats[~stats["is_annual_rollup"]]
    vendor_n = stats.groupby("vendor_id").size().rename("vendor_n")
    amt = (stats["amount"] * 100).round().astype("int64")
    same_amt = stats.assign(amount_cents=amt).groupby(["vendor_id", "amount_cents"]).size()
    p = p.join(vendor_n, on="vendor_id")
    p = p.join(same_amt.rename("vendor_same_amount_n"), on=["vendor_id", "amount_cents"])
    return p


def generate_candidates(p: pd.DataFrame, cfg: CandidateConfig) -> pd.DataFrame:
    con = duckdb.connect()
    con.register("p", p)
    exact = con.execute(
        """
        WITH g AS (
            SELECT payment_id,
                   first_value(payment_id) OVER w AS first_id,
                   count(*) OVER (PARTITION BY voucher_number, vendor_name_raw, amount_cents,
                                  check_date, department, contract_number) AS n
            FROM p WHERE voucher_number IS NOT NULL
            WINDOW w AS (PARTITION BY voucher_number, vendor_name_raw, amount_cents,
                         check_date, department, contract_number ORDER BY payment_id)
        )
        SELECT first_id AS a_id, payment_id AS b_id, 'T1' AS tier
        FROM g WHERE n > 1 AND payment_id <> first_id
        """
    ).df()
    pairs = con.execute(
        f"""
        SELECT a.payment_id AS a_id, b.payment_id AS b_id,
               CASE
                 WHEN a.amount_cents = b.amount_cents AND a.check_date = b.check_date
                      AND a.vendor_name_raw <> b.vendor_name_raw THEN 'T3'
                 WHEN a.amount_cents = b.amount_cents THEN 'T2'
                 ELSE 'T4'
               END AS tier
        FROM p a JOIN p b
          ON a.vendor_id = b.vendor_id
         AND a.payment_id < b.payment_id
         AND abs(date_diff('day', a.check_date, b.check_date)) <= {cfg.window_days}
        WHERE coalesce(a.voucher_number, '') <> coalesce(b.voucher_number, '')
          AND (
               a.amount_cents = b.amount_cents
            OR (abs(date_diff('day', a.check_date, b.check_date)) <= {cfg.near_days}
                AND (abs(a.amount_cents - b.amount_cents)
                         <= {cfg.near_pct} * greatest(a.amount_cents, b.amount_cents)
                     OR a.digits_key = b.digits_key))
          )
        """
    ).df()
    con.close()
    out = pd.concat([exact, pairs], ignore_index=True)
    # DuckDB returns join rows in no fixed order; sort so downstream model fits are
    # reproducible run to run.
    out = out.drop_duplicates(["a_id", "b_id"]).sort_values(["a_id", "b_id"])
    return out.reset_index(drop=True)


def _is_transposition(a: int, b: int) -> bool:
    sa, sb = str(a), str(b)
    if len(sa) != len(sb) or sa == sb:
        return False
    diff = [i for i in range(len(sa)) if sa[i] != sb[i]]
    return len(diff) == 2 and diff[1] == diff[0] + 1 and sa[diff[0]] == sb[diff[1]] \
        and sa[diff[1]] == sb[diff[0]]


FEATURES = [
    "day_gap", "same_day", "log_amount", "amount_rel_diff", "amount_equal", "is_transposition",
    "same_department", "same_contract", "direct_voucher", "raw_name_differs", "name_ratio",
    "vendor_log_n", "amount_share_vendor", "round_100", "monthly_cadence",
]


def pair_features(cands: pd.DataFrame, p: pd.DataFrame) -> pd.DataFrame:
    cols = ["payment_id", "voucher_number", "amount", "amount_cents", "check_date",
            "department", "contract_number", "vendor_name_raw", "vendor_id", "vendor_n",
            "vendor_same_amount_n"]
    side = p[cols].set_index("payment_id")
    a = side.loc[cands["a_id"]].reset_index(drop=True).add_prefix("a_")
    b = side.loc[cands["b_id"]].reset_index(drop=True).add_prefix("b_")
    df = pd.concat([cands.reset_index(drop=True), a, b], axis=1)

    gap = (df["a_check_date"] - df["b_check_date"]).dt.days.abs()
    hi = df[["a_amount", "b_amount"]].max(axis=1)
    df["day_gap"] = gap
    df["same_day"] = (gap == 0).astype(int)
    df["log_amount"] = np.log10(hi.clip(lower=1))
    df["amount_rel_diff"] = (df["a_amount"] - df["b_amount"]).abs() / hi
    df["amount_equal"] = (df["a_amount_cents"] == df["b_amount_cents"]).astype(int)
    df["is_transposition"] = [
        int(_is_transposition(x, y))
        for x, y in zip(df["a_amount_cents"], df["b_amount_cents"], strict=True)
    ]
    df["same_department"] = (
        df["a_department"].fillna("") == df["b_department"].fillna("")
    ).astype(int)
    df["same_contract"] = (
        df["a_contract_number"].fillna("") == df["b_contract_number"].fillna("")
    ).astype(int)
    df["direct_voucher"] = (
        (df["a_contract_number"] == "DV") | (df["b_contract_number"] == "DV")
    ).astype(int)
    df["raw_name_differs"] = (df["a_vendor_name_raw"] != df["b_vendor_name_raw"]).astype(int)
    df["name_ratio"] = [
        fuzz.ratio(x, y)
        for x, y in zip(df["a_vendor_name_raw"], df["b_vendor_name_raw"], strict=True)
    ]
    df["vendor_log_n"] = np.log1p(df["a_vendor_n"].fillna(1))
    # How routine is this amount for this vendor? Rent and subscriptions repeat exactly.
    share = df[["a_vendor_same_amount_n", "b_vendor_same_amount_n"]].max(axis=1)
    df["amount_share_vendor"] = share / df["a_vendor_n"].clip(lower=1)
    df["round_100"] = (df["a_amount_cents"] % 10_000 == 0).astype(int)
    df["monthly_cadence"] = gap.between(27, 32).astype(int)
    df["voucher_prefix_same"] = (
        df["a_voucher_number"].fillna("").str[:4] == df["b_voucher_number"].fillna("").str[:4]
    ).astype(int)
    df["pair_amount"] = df[["a_amount", "b_amount"]].min(axis=1)
    return df
