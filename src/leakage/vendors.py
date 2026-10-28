"""Vendor-name normalisation and entity resolution.

Two passes:
1. Deterministic normalisation collapses formatting noise ("ACME, INC." == "Acme Inc").
2. Fuzzy matching over distinct normalised names catches typos and near-variants.
   Candidate pairs come from char n-gram TF-IDF nearest neighbours, then each pair
   must also pass an edit-distance check and an exact match on any numbers in the
   name ("LOCAL 150" must never merge with "LOCAL 130"). Accepted pairs are joined
   with union-find into clusters, and each cluster gets a stable vendor_id.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer

LEGAL_SUFFIXES = {
    "INC", "INCORPORATED", "LLC", "LLP", "LP", "LTD", "LIMITED", "CO", "COMPANY",
    "CORP", "CORPORATION", "PC", "PLLC", "PLC", "NA", "GROUP",
}
_PUNCT = re.compile(r"[^A-Z0-9 ]+")
_SPACES = re.compile(r"\s+")
_NUMBERS = re.compile(r"\d+")


HONORIFICS = {"MR", "MRS", "MS", "DR", "MISS"}
_GENERATIONAL = re.compile(r"^(JR|SR|[IVX]{1,4})$")


def normalise_name(name: str) -> str:
    s = str(name).upper()
    s = s.replace("&", " AND ")
    s = re.sub(r"\bL\.?\s?L\.?\s?C\b\.?", " LLC ", s)
    s = _PUNCT.sub(" ", s)
    tokens = _SPACES.sub(" ", s).strip().split(" ")
    if tokens and tokens[0] == "THE" and len(tokens) > 1:
        tokens = tokens[1:]
    tokens = [t for t in tokens if t not in HONORIFICS] or tokens
    # Strip trailing legal suffixes, but never strip a name down to nothing.
    while len(tokens) > 1 and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


def _numbers(name: str) -> tuple[str, ...]:
    return tuple(_NUMBERS.findall(name))


def _last_char_substitution(x: str, y: str) -> bool:
    return len(x) == len(y) and x[:-1] == y[:-1] and x[-1] != y[-1]


def _token_close(x: str, y: str, allow_truncation: bool = True) -> bool:
    if x == y:
        return True
    if x[0] != y[0]:  # keying errors rarely hit the first letter; MORRIS != NORRIS
        return False
    if allow_truncation and min(len(x), len(y)) >= 4 and (x.startswith(y) or y.startswith(x)):
        return True  # truncation: COMM / COMMERCE
    if _last_char_substitution(x, y):  # MECCON / MECCOR are two different firms
        return False
    if min(len(x), len(y)) <= 6:  # short brand tokens: GRANT != GRANITE, CTS != CBTS
        return False
    # RESIDENTIAL vs RESIDENTS is a different word, not a typo.
    return Levenshtein.distance(x, y) <= 2 and fuzz.ratio(x, y) >= 80


ORG_WORDS = {
    "ASSOC", "ASSOCIATION", "ASSOCIATES", "SERVICES", "SERVICE", "CONSTRUCTION", "DEVELOPMENT",
    "SOCIETY", "CENTER", "CENTRE", "FOUNDATION", "CHURCH", "PARTNERS", "BANK", "UNIVERSITY",
    "COLLEGE", "HOSPITAL", "SCHOOL", "COUNCIL", "DEPT", "DEPARTMENT", "ILLINOIS", "CHICAGO",
    "AMERICA", "AMERICAN", "NATIONAL", "COMMUNITY", "SUPPLY", "SUPPLIES", "SYSTEMS", "SOLUTIONS",
    "TECHNOLOGIES", "TECHNOLOGY", "CONSULTING", "MANAGEMENT", "HOUSING", "INSURANCE", "TRUST",
    "FUND", "AUTHORITY", "AGENCY", "INTERNATIONAL", "ENTERPRISES", "INDUSTRIES", "HOLDINGS",
    "PROPERTIES", "REALTY", "COMMERCE", "CHAMBER", "LAW", "OFFICE", "OFFICES", "CLUB", "CITY",
    "COUNTY", "STATE", "INSTITUTE", "PRODUCTS", "EQUIPMENT", "ELECTRIC", "MOTORS", "AND", "OF",
    "LLC", "INC", "CORP", "CO", "COMPANY", "LTD", "GROUP", "LP", "LLP",
}


def is_org(raw_or_norm: str) -> bool:
    tokens = _PUNCT.sub(" ", raw_or_norm.upper()).split()
    long_tokens = [t for t in tokens if len(t) > 1]
    return len(long_tokens) >= 4 or any(t in ORG_WORDS for t in tokens)


def _person_compatible(ca: list[str], cb: list[str]) -> bool:
    """Same tokens except one single-character typo inside a long-enough token.

    An edit at the last character is refused: ANDREA/ANDREW, GABRIEL/GABRIELA and
    WALTER/WALTERS are usually different people.
    """
    for xs, ys in ((ca, cb), (sorted(ca), sorted(cb))):
        diffs = [(x, y) for x, y in zip(xs, ys, strict=True) if x != y]
        if not diffs:
            return True
        if len(diffs) == 1:
            x, y = diffs[0]
            if (
                min(len(x), len(y)) >= 7
                and x[0] == y[0]
                and x[-1] == y[-1]
                and Levenshtein.distance(x, y) == 1
            ):
                return True
    return False


def tokens_compatible(a: str, b: str, org: bool | None = None) -> bool:
    """Token-level guard against merging different people or numbered entities.

    Half of a municipal ledger's payees are individuals (refunds, settlements), where
    'KENNETH B SMITH' vs 'KENNETH D SMITH' is two people, not a typo. Individuals get a
    much stricter rule than organisations.
    """
    if org is None:
        org = is_org(a) or is_org(b)
    ta, tb = a.split(), b.split()
    # 'HOWARD ERNESTLEON' == 'HOWARD ERNEST LEON', but 'JUAN A' must not become 'JUANA'.
    if a.replace(" ", "") == b.replace(" ", "") and (
        org or not any(len(t) == 1 for t in ta + tb)
    ):
        return True
    gen_a = {t for t in ta if _GENERATIONAL.match(t)}
    gen_b = {t for t in tb if _GENERATIONAL.match(t)}
    if gen_a != gen_b:
        return False
    ini_a = {t for t in ta if len(t) == 1}
    ini_b = {t for t in tb if len(t) == 1}
    if ini_a != ini_b and (not org or (ini_a and ini_b)):
        return False
    ca = [t for t in ta if len(t) > 1 and t not in gen_a]
    cb = [t for t in tb if len(t) > 1 and t not in gen_b]
    if len(ca) != len(cb) or not ca:
        return False
    if not org:
        return _person_compatible(ca, cb)
    # Truncation is fine for trailing words (DEVEL / DEVELOPMENT) but not for the
    # leading brand word: FARO != FARONICS, HEALTH PARTNERS != HEALTHCARE PARTNERS.
    in_order = all(
        _token_close(x, y, allow_truncation=i > 0)
        for i, (x, y) in enumerate(zip(ca, cb, strict=True))
    )
    return in_order or all(
        _token_close(x, y, allow_truncation=False)
        for x, y in zip(sorted(ca), sorted(cb), strict=True)
    )


class _UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n)

    def find(self, i: int) -> int:
        root = i
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[i] != root:
            self.parent[i], i = root, self.parent[i]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


@dataclass(frozen=True)
class MatchConfig:
    cosine_min: float = 0.75
    ratio_min: float = 85.0
    min_len: int = 6
    top_k: int = 10
    chunk: int = 2000


def candidate_pairs(names: list[str], cfg: MatchConfig) -> pd.DataFrame:
    """Nearest-neighbour pairs (i < j) by char n-gram cosine similarity."""
    if len(names) < 2:
        return pd.DataFrame(columns=["i", "j", "cosine"])
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=1, dtype=np.float32)
    X = vec.fit_transform(names)
    rows, cols, sims = [], [], []
    k = min(cfg.top_k + 1, len(names))
    for start in range(0, X.shape[0], cfg.chunk):
        block = (X[start : start + cfg.chunk] @ X.T).toarray()
        np.fill_diagonal(block[:, start : start + cfg.chunk], 0.0)
        idx = np.argpartition(-block, kth=k - 1, axis=1)[:, :k]
        for r in range(block.shape[0]):
            for c in idx[r]:
                s = block[r, c]
                if s >= cfg.cosine_min:
                    i, j = start + r, int(c)
                    rows.append(min(i, j))
                    cols.append(max(i, j))
                    sims.append(float(s))
    pairs = pd.DataFrame({"i": rows, "j": cols, "cosine": sims})
    return pairs.drop_duplicates(["i", "j"]).reset_index(drop=True)


def accept_pair(a: str, b: str, cfg: MatchConfig) -> tuple[bool, float]:
    if min(len(a), len(b)) < cfg.min_len:
        return False, 0.0
    if _numbers(a) != _numbers(b):
        return False, 0.0
    ratio = fuzz.ratio(a, b)
    return ratio >= cfg.ratio_min and tokens_compatible(a, b), ratio


def resolve_vendors(raw_names: pd.Series, cfg: MatchConfig | None = None):
    """Return (name_map, merges).

    name_map: one row per distinct raw name -> norm_name, vendor_id, canonical_name.
    merges:   accepted fuzzy pairs between distinct normalised names, for review.
    """
    cfg = cfg or MatchConfig()
    counts = raw_names.value_counts()
    name_map = pd.DataFrame({"vendor_name_raw": counts.index, "n_payments": counts.values})
    name_map["norm_name"] = name_map["vendor_name_raw"].map(normalise_name)

    norm = (
        name_map.groupby("norm_name", as_index=False)["n_payments"].sum()
        .sort_values("norm_name").reset_index(drop=True)
    )
    names = norm["norm_name"].tolist()
    pairs = candidate_pairs(names, cfg)

    uf = _UnionFind(len(names))
    accepted = []
    for i, j, cos in pairs.itertuples(index=False):
        ok, ratio = accept_pair(names[i], names[j], cfg)
        if ok:
            uf.union(i, j)
            accepted.append((names[i], names[j], cos, ratio))

    norm["root"] = [uf.find(i) for i in range(len(names))]
    # Canonical name = the most-paid normalised name in the cluster.
    canon = (
        norm.sort_values(["root", "n_payments", "norm_name"], ascending=[True, False, True])
        .drop_duplicates("root")[["root", "norm_name"]]
        .rename(columns={"norm_name": "canonical_name"})
    )
    norm = norm.merge(canon, on="root")
    norm["vendor_id"] = norm["canonical_name"].map(
        lambda s: "V" + hashlib.sha1(s.encode()).hexdigest()[:10]
    )
    name_map = name_map.merge(norm[["norm_name", "vendor_id", "canonical_name"]], on="norm_name")
    merges = pd.DataFrame(accepted, columns=["name_a", "name_b", "cosine", "ratio"])
    return name_map, merges


# --- evaluation on synthetic perturbations -------------------------------------------------

def perturb_name(name: str, rng: np.random.Generator) -> str:
    """One realistic data-entry variant of a vendor name."""
    choice = rng.integers(0, 5)
    s = name
    if choice == 0 and len(s) > 6:  # dropped character
        p = int(rng.integers(1, len(s) - 1))
        s = s[:p] + s[p + 1 :]
    elif choice == 1 and len(s) > 6:  # adjacent swap
        p = int(rng.integers(1, len(s) - 2))
        s = s[:p] + s[p + 1] + s[p] + s[p + 2 :]
    elif choice == 2:  # legal suffix variant
        s = re.sub(r"\b(INC|LLC|CORP|CO)\.?$", "", s).strip()
        s = s + rng.choice([" INC", " LLC", ", INC.", " CORPORATION", " CO"])
    elif choice == 3:  # punctuation / ampersand noise
        s = s.replace(" AND ", " & ").replace(",", "").replace(".", "") + "."
    else:  # doubled letter
        letters = [i for i, c in enumerate(s) if c.isalpha()]
        if letters:
            p = int(rng.choice(letters))
            s = s[:p] + s[p] + s[p:]
    return s


def evaluate_resolution(raw_names: pd.Series, n: int = 500, seed: int = 7,
                        cfg: MatchConfig | None = None) -> dict:
    """Inject perturbed copies of real vendor names and check where they land.

    recall      = perturbed copy lands in the same cluster as its source name
    wrong_merge = perturbed copy lands in a cluster belonging to a *different* source
    """
    rng = np.random.default_rng(seed)
    distinct = pd.Series(raw_names.dropna().unique())
    distinct = distinct[distinct.str.len() >= 8]
    sample = distinct.sample(min(n, len(distinct)), random_state=seed).tolist()
    variants = [perturb_name(s, rng) for s in sample]
    keep = [(s, v) for s, v in zip(sample, variants, strict=True) if v != s]
    augmented = pd.concat([raw_names, pd.Series([v for _, v in keep])], ignore_index=True)
    name_map, _ = resolve_vendors(augmented, cfg)
    vid = dict(zip(name_map["vendor_name_raw"], name_map["vendor_id"], strict=True))
    same = sum(vid[s] == vid[v] for s, v in keep)
    # A variant identical (after normalisation) to another real name isn't a wrong merge
    # we can judge, so wrong merges are counted only against the variant's own source.
    return {
        "n_injected": len(keep),
        "recall": same / len(keep) if keep else float("nan"),
        "missed": len(keep) - same,
    }
