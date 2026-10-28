"""Rank duplicate-payment candidates with a model trained on injected duplicates.

There are no labels in a public ledger: nobody publishes "this payment was a duplicate".
So we use the standard audit-analytics evaluation design: copy real payments, perturb
them the way duplicates actually happen (re-keyed invoice, a few days later, transposed
digits, a vendor-name typo), inject them, and see whether the pipeline finds them.

Every candidate pair that isn't an injected (source, copy) pair is treated as a negative.
A handful of those negatives are probably real duplicates, so measured precision is a
lower bound (positive-unlabelled learning).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupShuffleSplit

from leakage.duplicates import (
    FEATURES,
    CandidateConfig,
    generate_candidates,
    pair_features,
    prepare,
)
from leakage.vendors import perturb_name, resolve_vendors


@dataclass(frozen=True)
class InjectionConfig:
    n: int = 3000
    seed: int = 42
    p_vendor_typo: float = 0.15
    p_transposed: float = 0.10
    p_near_amount: float = 0.10


def _transpose(cents: int, rng: np.random.Generator) -> int:
    s = str(cents)
    spots = [i for i in range(len(s) - 1) if s[i] != s[i + 1] and not (i == 0 and s[1] == "0")]
    if not spots:
        return cents
    i = int(rng.choice(spots))
    return int(s[:i] + s[i + 1] + s[i] + s[i + 2 :])


def inject_duplicates(payments: pd.DataFrame, cfg: InjectionConfig, min_amount: float):
    """Return (augmented payments, truth pairs as a set of (source_id, copy_id))."""
    rng = np.random.default_rng(cfg.seed)
    eligible = payments[
        (~payments["is_annual_rollup"])
        & payments["check_date"].notna()
        & (payments["amount"] >= min_amount)
        & payments["voucher_number"].notna()
    ]
    src = eligible.sample(min(cfg.n, len(eligible)), random_state=cfg.seed).copy()

    dup = src.copy()
    dup["payment_id"] = [f"INJ{i:06d}" for i in range(len(dup))]
    # Realistic voucher: borrow one from a real payment in the same voucher series
    # (same 4-char prefix). A first version drew from *any* voucher, and the model
    # learned "prefix differs => injected" (single-feature AUC 0.05). Don't do that.
    prefix = eligible["voucher_number"].str[:4]
    pools = eligible.groupby(prefix)["voucher_number"].apply(lambda s: s.to_numpy())
    dup["voucher_number"] = [
        rng.choice(pools[v[:4]]) for v in src["voucher_number"]
    ]
    kind = rng.random(len(dup))
    shift = np.where(kind < 0.3, 0, np.where(kind < 0.8, rng.integers(1, 11, len(dup)),
                                               rng.integers(11, 31, len(dup))))
    sign = rng.choice([-1, 1], size=len(dup))
    dup["check_date"] = dup["check_date"] + pd.to_timedelta(shift * sign, unit="D")

    cents = (dup["amount"] * 100).round().astype("int64").to_numpy()
    roll = rng.random(len(dup))
    new_cents = cents.copy()
    trans = roll < cfg.p_transposed
    near = (roll >= cfg.p_transposed) & (roll < cfg.p_transposed + cfg.p_near_amount)
    new_cents[trans] = [_transpose(int(c), rng) for c in cents[trans]]
    new_cents[near] = (cents[near] * (1 + rng.uniform(-0.004, 0.004, near.sum()))).round()
    # Amount edits only make sense within the near-amount window; keep those close in time.
    edited = new_cents != cents
    dup.loc[edited, "check_date"] = src.loc[edited, "check_date"] + pd.to_timedelta(
        rng.integers(0, 10, edited.sum()), unit="D"
    )
    dup["amount"] = new_cents / 100

    typo = rng.random(len(dup)) < cfg.p_vendor_typo
    dup.loc[typo, "vendor_name_raw"] = [
        perturb_name(n, rng) for n in dup.loc[typo, "vendor_name_raw"]
    ]
    dup["check_year"] = dup["check_date"].dt.year
    truth = set(zip(src["payment_id"], dup["payment_id"], strict=True))
    return pd.concat([payments, dup], ignore_index=True), truth


def label_pairs(cands: pd.DataFrame, truth: set[tuple[str, str]]) -> pd.Series:
    fwd = set(truth)
    rev = {(b, a) for a, b in truth}
    keys = list(zip(cands["a_id"], cands["b_id"], strict=True))
    return pd.Series([int(k in fwd or k in rev) for k in keys], index=cands.index)


def _model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0,
        class_weight="balanced", random_state=0,
    )


def build_training_set(payments: pd.DataFrame, name_map: pd.DataFrame,
                       cand_cfg: CandidateConfig, inj_cfg: InjectionConfig):
    aug, truth = inject_duplicates(payments, inj_cfg, cand_cfg.min_amount)
    # Typos in injected vendor names have to go through entity resolution too:
    # if ER misses them, the duplicate is (correctly) never found.
    new_names = set(aug["vendor_name_raw"]) - set(name_map["vendor_name_raw"])
    if new_names:
        aug_map, _ = resolve_vendors(aug["vendor_name_raw"])
    else:
        aug_map = name_map
    p = prepare(aug, aug_map, cand_cfg)
    cands = generate_candidates(p, cand_cfg)
    feats = pair_features(cands, p)
    feats["label"] = label_pairs(feats, truth)
    injected = feats["a_id"].str.startswith("INJ") | feats["b_id"].str.startswith("INJ")
    # Drop injected rows paired with anything other than their source: ambiguous labels.
    feats = feats[~injected | (feats["label"] == 1)].reset_index(drop=True)
    generation_recall = feats["label"].sum() / len(truth)
    return feats, generation_recall, len(truth)


def train_and_evaluate(feats: pd.DataFrame, seed: int = 0) -> tuple[dict, pd.DataFrame]:
    groups = feats["a_vendor_id"].fillna("NA")
    split = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    tr, te = next(split.split(feats, feats["label"], groups))
    train, test = feats.iloc[tr], feats.iloc[te]
    model = _model().fit(train[FEATURES], train["label"])
    score = model.predict_proba(test[FEATURES])[:, 1]

    # Baseline the model has to beat: the classic rule "same vendor + same amount,
    # closest dates first", which is what most AP duplicate checks do.
    base = test["amount_equal"] * 1.0 - test["day_gap"] / 1000.0

    test = test.assign(score=score, baseline=base.to_numpy())
    pos = int(test["label"].sum())
    out = {
        "test_pairs": len(test),
        "test_positives": pos,
        "pr_auc_model": float(average_precision_score(test["label"], score)),
        "pr_auc_rule_baseline": float(average_precision_score(test["label"], base)),
    }
    for k in (100, 500):
        out[f"precision_at_{k}_model"] = float(test.nlargest(k, "score")["label"].mean())
        out[f"precision_at_{k}_rule_baseline"] = float(test.nlargest(k, "baseline")["label"].mean())
    out[f"recall_at_{pos}_model"] = float(test.nlargest(pos, "score")["label"].sum() / max(pos, 1))
    return out, test


def fit_final(feats: pd.DataFrame) -> HistGradientBoostingClassifier:
    return _model().fit(feats[FEATURES], feats["label"])
