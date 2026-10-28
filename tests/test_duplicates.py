import pandas as pd
import pytest

from leakage.duplicates import (
    CandidateConfig,
    _is_transposition,
    digits_key,
    excluded_payee,
    generate_candidates,
    pair_features,
    prepare,
)
from leakage.vendors import resolve_vendors


@pytest.fixture
def candidates(small_ledger):
    cfg = CandidateConfig()
    nm, _ = resolve_vendors(small_ledger["vendor_name_raw"])
    p = prepare(small_ledger, nm, cfg)
    cands = generate_candidates(p, cfg)
    return pair_features(cands, p)


def _pair(feats: pd.DataFrame, vendor: str) -> pd.DataFrame:
    hit = feats["a_vendor_name_raw"].str.upper().str.contains(vendor)
    return feats[hit]


def test_each_planted_tier_is_found(candidates):
    assert set(_pair(candidates, "OMEGA")["tier"]) == {"T1"}
    assert set(_pair(candidates, "SIGMA")["tier"]) == {"T2"}
    assert set(_pair(candidates, "KAPPA")["tier"]) == {"T3"}
    zeta = _pair(candidates, "ZETA")
    assert set(zeta["tier"]) == {"T4"} and zeta["is_transposition"].iloc[0] == 1


def test_rollups_never_become_candidates(candidates):
    assert _pair(candidates, "ROLLUP").empty


def test_candidate_order_is_deterministic(small_ledger):
    cfg = CandidateConfig()
    nm, _ = resolve_vendors(small_ledger["vendor_name_raw"])
    p = prepare(small_ledger, nm, cfg)
    a = generate_candidates(p, cfg)
    b = generate_candidates(p.sample(frac=1, random_state=1), cfg)
    pd.testing.assert_frame_equal(a, b)


def test_window_is_respected(small_ledger):
    cfg = CandidateConfig(window_days=5)
    nm, _ = resolve_vendors(small_ledger["vendor_name_raw"])
    cands = generate_candidates(prepare(small_ledger, nm, cfg), cfg)
    p = small_ledger.set_index("payment_id")
    sigma = p[p["vendor_name_raw"] == "SIGMA CORP"].index
    assert not cands["a_id"].isin(sigma).any()  # 9 days apart > 5-day window


@pytest.mark.parametrize("a,b,expected", [
    (557550, 555750, True),
    (123400, 124300, True),
    (123400, 123400, False),
    (123400, 432100, False),
])
def test_is_transposition(a, b, expected):
    assert _is_transposition(a, b) is expected


def test_digits_key_matches_transpositions():
    assert digits_key(5575.50) == digits_key(5557.50)


def test_excluded_payees():
    s = pd.Series(["MUNICIPAL EMPLOYEE PENSION FD", "ZIONS BANK", "POLICEMENS A & B FUND",
                   "ACME PLUMBING", "TRUSTEE OF X", "BANKS PLUMBING",
                   "U S BANK01 NATIONAL ASSOCIATION"])
    assert excluded_payee(s).tolist() == [True, True, True, False, True, False, True]
