import numpy as np
import pandas as pd
import pytest

from leakage.vendors import normalise_name, perturb_name, resolve_vendors, tokens_compatible


@pytest.mark.parametrize("raw,expected", [
    ("Acme, Inc.", "ACME"),
    ("THE ACME COMPANY", "ACME"),
    ("A & B Supply L.L.C.", "A AND B SUPPLY"),
    ("MS BOWE DAWN", "BOWE DAWN"),
    ("INC", "INC"),  # never normalise a name down to nothing
])
def test_normalise_name(raw, expected):
    assert normalise_name(raw) == expected


@pytest.mark.parametrize("a,b,same", [
    ("KENNETH B SMITH", "KENNETH D SMITH", False),       # different middle initials
    ("CURTIS MORRIS", "CURTIS NORRIS", False),           # first-letter change
    ("ANDREA MALONE", "ANDREW MALONE", False),           # last-letter change, people
    ("RODRIGUEZ ANGEL", "RODRIGUEZ ANGEL SR", False),    # generational suffix
    ("COMMUNITY HOUSING PARTNERS II", "COMMUNITY HOUSING PARTNERS XV", False),
    ("GRANITE CONSTRUCTION", "GRANT CONSTRUCTION", False),
    ("CHICAGO RESIDENTIAL", "CHICAGO RESIDENTS", False),
    ("JUAN A RODRIGUEZ", "JUANA RODRIGUEZ", False),
    ("FARO TECHNOLOGIES", "FARONICS TECHNOLOGIES", False),
    ("HEALTH PARTNERS", "HEALTHCARE PARTNERS", False),
    ("HISPANIC HOUSING DEVEL", "HISPANIC HOUSING DEVELOPMENT", True),
    ("CUBAN AMERICAN CHAMBER OF COMM", "CUBAN AMERICAN CHAMBER OF COMMERCE", True),
    ("ADVANCE ELECTRICAL SUPPLY", "ADVANCED ELECTRICAL SUPPLY", True),
    ("HOWARD ERNEST LEON", "HOWARD ERNESTLEON", True),
    ("CHRISTOPHER KIRKENDOLL", "CHRISTOPHER KIRLENDOLL", True),
])
def test_tokens_compatible(a, b, same):
    assert tokens_compatible(a, b) is same


def test_resolution_merges_typos_but_not_numbered_entities():
    names = pd.Series([
        "SCHWAB REHABILITATION HOSPITAL", "SCHWAB REHABILATION HOSPITAL",
        "LOCAL 150 UNION", "LOCAL 130 UNION",
        "Acme, Inc.", "ACME INC",
    ])
    nm, merges = resolve_vendors(names)
    vid = dict(zip(nm["vendor_name_raw"], nm["vendor_id"], strict=True))
    assert vid["SCHWAB REHABILITATION HOSPITAL"] == vid["SCHWAB REHABILATION HOSPITAL"]
    assert vid["LOCAL 150 UNION"] != vid["LOCAL 130 UNION"]
    assert vid["Acme, Inc."] == vid["ACME INC"]  # exact after normalisation, not fuzzy
    assert len(merges) == 1


def test_vendor_ids_are_stable_across_runs():
    names = pd.Series(["ALPHA SERVICES", "ALPHA SERVICE", "BETA LLC"])
    a, _ = resolve_vendors(names)
    b, _ = resolve_vendors(names.sample(frac=1, random_state=3))
    key = ["vendor_name_raw", "vendor_id"]
    assert a[key].sort_values(key).values.tolist() == b[key].sort_values(key).values.tolist()


def test_perturb_name_is_deterministic():
    rng1, rng2 = np.random.default_rng(1), np.random.default_rng(1)
    assert [perturb_name("ACME PLUMBING INC", rng1) for _ in range(5)] == [
        perturb_name("ACME PLUMBING INC", rng2) for _ in range(5)
    ]
