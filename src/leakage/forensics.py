"""Classic forensic-accounting tests over the payments ledger.

- Benford's law first-digit test with Nigrini's MAD conformity bands.
- Threshold clustering: are payments bunched just under round approval limits?
- Split purchases: several direct-voucher payments to one vendor within a week,
  each under a limit, together over it.
- Vendor-month spend spikes by robust z-score.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

BENFORD = np.log10(1 + 1 / np.arange(1, 10))
# Nigrini (2012) first-digit MAD bands.
MAD_BANDS = [(0.006, "close conformity"), (0.012, "acceptable conformity"),
             (0.015, "marginal conformity"), (np.inf, "nonconformity")]


def first_digit(amounts: pd.Series) -> pd.Series:
    a = amounts[amounts >= 10].abs()
    return (a / 10 ** np.floor(np.log10(a))).astype(int).clip(1, 9)


def benford_mad(amounts: pd.Series) -> tuple[float, np.ndarray, int]:
    d = first_digit(amounts)
    n = len(d)
    if n == 0:
        return float("nan"), np.zeros(9), 0
    observed = np.bincount(d, minlength=10)[1:] / n
    return float(np.mean(np.abs(observed - BENFORD))), observed, n


def mad_band(mad: float) -> str:
    return next(label for cut, label in MAD_BANDS if mad <= cut)


def benford_by_group(payments: pd.DataFrame, group: str = "department",
                     min_n: int = 1000) -> pd.DataFrame:
    rows = []
    for key, g in payments.groupby(group):
        mad, obs, n = benford_mad(g["amount"])
        if n >= min_n:
            rows.append({group: key, "n": n, "mad": mad, "conformity": mad_band(mad),
                         **{f"d{i + 1}": obs[i] for i in range(9)}})
    columns = [group, "n", "mad", "conformity", *[f"d{i}" for i in range(1, 10)]]
    out = pd.DataFrame(rows, columns=columns)
    return out.sort_values("mad", ascending=False).reset_index(drop=True)


DEFAULT_THRESHOLDS = (1_000, 2_500, 5_000, 10_000, 25_000, 50_000, 100_000)


def threshold_clustering(payments: pd.DataFrame,
                         thresholds=DEFAULT_THRESHOLDS, band: float = 0.10) -> pd.DataFrame:
    """Count payments just under vs just over each limit, against non-round control points.

    The amount distribution slopes, so a raw below/above ratio isn't evidence on its own.
    `lift` compares the ratio at the limit with the same ratio at 0.8x and 1.25x the
    limit, where nobody has a reason to bunch. Lift well above 1 suggests bunching.
    """
    a = payments["amount"]

    def ratio(t: float) -> tuple[int, int, float]:
        below = int(((a >= t * (1 - band)) & (a < t)).sum())
        above = int(((a >= t) & (a < t * (1 + band))).sum())
        return below, above, below / above if above else float("nan")

    rows = []
    for t in thresholds:
        below, above, r = ratio(t)
        control = np.nanmean([ratio(t * 0.8)[2], ratio(t * 1.25)[2]])
        rows.append({"threshold": t, "just_below": below, "just_above": above,
                     "below_above_ratio": r, "control_ratio": control,
                     "lift": r / control if control else float("nan")})
    return pd.DataFrame(rows)


def split_purchases(payments: pd.DataFrame, threshold: float, window_days: int = 7,
                    min_share: float = 0.2) -> pd.DataFrame:
    """Windows where ≥2 under-limit payments to one vendor+department sum over the limit.

    Each payment must be at least `min_share` of the limit so trivial items don't qualify.
    Overlapping windows are collapsed to the earliest start.
    """
    p = payments[(payments["amount"] < threshold) & (payments["amount"] >= min_share * threshold)]
    con = duckdb.connect()
    con.register("p", p)
    w = con.execute(
        f"""
        SELECT vendor_id, department, check_date AS window_start,
               sum(amount) OVER win AS window_total,
               count(*) OVER win AS n_payments,
               list(payment_id) OVER win AS payment_ids
        FROM p
        WINDOW win AS (PARTITION BY vendor_id, department ORDER BY check_date
                       RANGE BETWEEN CURRENT ROW AND INTERVAL {window_days - 1} DAYS FOLLOWING)
        """
    ).df()
    con.close()
    w = w[(w["n_payments"] >= 2) & (w["window_total"] >= threshold)]
    if w.empty:
        return w.assign(threshold=threshold)
    w = w.sort_values(["vendor_id", "department", "window_start"])
    keep, last_end = [], {}
    for row in w.itertuples(index=False):
        key = (row.vendor_id, row.department)
        if key in last_end and row.window_start <= last_end[key]:
            continue
        keep.append(row)
        last_end[key] = row.window_start + pd.Timedelta(days=window_days - 1)
    out = pd.DataFrame(keep)
    out["threshold"] = threshold
    out["excess_over_limit"] = out["window_total"] - threshold
    return out.sort_values("window_total", ascending=False).reset_index(drop=True)


def spend_spikes(payments: pd.DataFrame, min_months: int = 6, z_min: float = 6.0,
                 min_amount: float = 100_000) -> pd.DataFrame:
    m = (
        payments.assign(month=payments["check_date"].dt.to_period("M").dt.to_timestamp())
        .groupby(["vendor_id", "month"], as_index=False)["amount"].sum()
    )
    stats = m.groupby("vendor_id")["amount"].agg(
        months="size", median="median",
        mad=lambda s: float(np.median(np.abs(s - np.median(s)))),
    )
    m = m.join(stats, on="vendor_id")
    m = m[m["months"] >= min_months]
    # Floor the scale: a vendor paid the same amount every month has MAD == 0.
    scale = np.maximum.reduce([1.4826 * m["mad"], 0.1 * m["median"].abs(),
                               np.full(len(m), 1000.0)])
    m["robust_z"] = (m["amount"] - m["median"]) / scale
    out = m[(m["robust_z"] >= z_min) & (m["amount"] >= min_amount)]
    return out.sort_values("robust_z", ascending=False).reset_index(drop=True)
