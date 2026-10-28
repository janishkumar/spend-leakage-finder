import numpy as np

from leakage.duplicates import CandidateConfig
from leakage.scorer import (
    InjectionConfig,
    _transpose,
    build_training_set,
    inject_duplicates,
    train_and_evaluate,
)
from leakage.vendors import resolve_vendors


def test_transpose_swaps_adjacent_distinct_digits():
    rng = np.random.default_rng(0)
    for cents in (123456, 557550, 100):
        out = _transpose(cents, rng)
        assert sorted(str(out)) == sorted(str(cents))
        assert len(str(out)) == len(str(cents))
    assert _transpose(1111, rng) == 1111  # nothing to swap


def test_injection_keeps_voucher_series(small_ledger):
    aug, truth = inject_duplicates(small_ledger, InjectionConfig(n=50, seed=1), 100.0)
    assert len(truth) == 50
    by_id = aug.set_index("payment_id")
    for src, dup in truth:
        # Regression: copies drawn from other voucher series leaked the label.
        assert by_id.loc[src, "voucher_number"][:4] == by_id.loc[dup, "voucher_number"][:4]
        gap = abs((by_id.loc[src, "check_date"] - by_id.loc[dup, "check_date"]).days)
        assert gap <= 30


def test_training_and_evaluation_run_end_to_end(small_ledger):
    nm, _ = resolve_vendors(small_ledger["vendor_name_raw"])
    feats, gen_recall, n = build_training_set(
        small_ledger, nm, CandidateConfig(), InjectionConfig(n=120, seed=3)
    )
    assert n == 120 and 0.5 < gen_recall <= 1.0
    metrics, test = train_and_evaluate(feats)
    assert 0.0 <= metrics["pr_auc_model"] <= 1.0
    assert "score" in test
