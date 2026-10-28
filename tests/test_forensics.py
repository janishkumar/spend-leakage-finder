import numpy as np
import pandas as pd

from leakage import forensics


def test_benford_sample_conforms():
    rng = np.random.default_rng(0)
    amounts = pd.Series(10 ** rng.uniform(1, 6, 50_000))  # log-uniform obeys Benford
    mad, observed, n = forensics.benford_mad(amounts)
    assert n == 50_000
    assert forensics.mad_band(mad) == "close conformity"
    assert abs(observed.sum() - 1) < 1e-9


def test_fixed_price_ledger_does_not_conform():
    mad, _, _ = forensics.benford_mad(pd.Series([500.0] * 1000 + [750.0] * 1000))
    assert forensics.mad_band(mad) == "nonconformity"


def _ledger(rows):
    df = pd.DataFrame(rows, columns=["vendor_id", "department", "check_date", "amount"])
    df["check_date"] = pd.to_datetime(df["check_date"])
    df["payment_id"] = [f"P{i}" for i in range(len(df))]
    return df


def test_split_purchase_detected_and_collapsed():
    df = _ledger([
        ("V1", "D", "2024-01-01", 4000.0),
        ("V1", "D", "2024-01-03", 3500.0),
        ("V1", "D", "2024-01-05", 3000.0),
        ("V2", "D", "2024-01-01", 4000.0),   # alone: not a split
        ("V3", "D", "2024-01-01", 4000.0),
        ("V3", "D", "2024-01-20", 4000.0),   # outside the window
    ])
    out = forensics.split_purchases(df, threshold=5000)
    assert out["vendor_id"].tolist() == ["V1"]
    assert out.iloc[0]["n_payments"] == 3


def test_threshold_clustering_detects_bunching():
    rng = np.random.default_rng(1)
    smooth = rng.uniform(100, 20000, 20000)
    bunched = rng.uniform(9100, 9999, 3000)
    out = forensics.threshold_clustering(
        pd.DataFrame({"amount": np.concatenate([smooth, bunched])}), thresholds=(10_000,)
    )
    assert out.iloc[0]["lift"] > 2


def test_spikes_handle_constant_vendors():
    months = pd.date_range("2023-01-01", periods=12, freq="MS")
    rows = [("V1", "D", m, 1000.0) for m in months] + [("V1", "D", "2024-01-15", 500000.0)]
    out = forensics.spend_spikes(_ledger(rows), min_amount=100_000)
    assert len(out) == 1 and np.isfinite(out.iloc[0]["robust_z"])
