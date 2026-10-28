import pandas as pd
import pytest

from leakage.report import RoiAssumptions, review_set, roi, sensitivity


@pytest.fixture
def queue():
    return pd.DataFrame({
        "a_id": ["a1", "a2", "a3", "a4", "a5"],
        "b_id": ["b1", "b1", "b2", "b3", "b4"],   # b1 appears in two pairs
        "tier": ["T2", "T2", "T1", "T4", "T2"],
        "excluded": [False, False, False, False, True],
        "risk_score": [0.9, 0.8, 0.99, 0.7, 0.95],
        "exposure": [100.0, 100.0, 5000.0, 200.0, 1e9],
    })


def test_review_set_skips_t1_excluded_and_double_counts(queue):
    rs = review_set(queue, 10)
    assert rs["b_id"].tolist() == ["b1", "b3"]


def test_roi_math(queue):
    r = roi(queue, RoiAssumptions(top_k=10, confirmation_rate=0.5,
                                  minutes_per_review=60, hourly_rate=10))
    assert r["exposure_reviewed"] == 300.0
    assert r["est_recoverable"] == 150.0
    assert r["review_cost"] == 20.0
    assert r["net"] == 130.0


def test_sensitivity_is_monotonic(queue):
    s = sensitivity(queue, RoiAssumptions(top_k=10))
    assert s["est_recoverable"].is_monotonic_increasing
