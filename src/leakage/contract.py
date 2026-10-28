"""Standardise a raw payments extract and enforce the data contract.

Everything downstream reads the standard schema below, so adding a new ledger means
writing one `standardise_*` function, not touching the detectors.
"""

from __future__ import annotations

import hashlib

import pandas as pd
import pandera.pandas as pa

PAYMENTS_SCHEMA = pa.DataFrameSchema(
    {
        "payment_id": pa.Column(str, unique=True),
        "voucher_number": pa.Column(str, nullable=True),
        "amount": pa.Column(float, pa.Check(lambda s: s.abs() < 1e10), nullable=False),
        "check_date": pa.Column("datetime64[ns]", nullable=True),
        "check_year": pa.Column(int, pa.Check.in_range(1990, 2100)),
        "department": pa.Column(str, nullable=True),
        "contract_number": pa.Column(str, nullable=True),
        "vendor_name_raw": pa.Column(str, pa.Check.str_length(min_value=1)),
        "is_annual_rollup": pa.Column(bool),
    },
    strict=True,
    coerce=True,
)


def _clean_text(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    return s.mask(s == "")


def standardise_chicago(raw: pd.DataFrame) -> pd.DataFrame:
    """Map the City of Chicago payments extract to the standard schema.

    Chicago publishes two kinds of rows: voucher-level payments with a full MM/DD/YYYY
    check date, and annual roll-ups (check_date = 'YYYY', no voucher) that aggregate
    small payments to a contract. Roll-ups are kept but flagged so detectors skip them.
    """
    df = pd.DataFrame(index=raw.index)
    date_txt = _clean_text(raw.get("check_date", pd.Series(pd.NA, index=raw.index)))
    full_date = pd.to_datetime(date_txt, format="%m/%d/%Y", errors="coerce")
    year_only = pd.to_numeric(
        date_txt.where(date_txt.str.len() == 4), errors="coerce"
    ).astype("float64")

    df["voucher_number"] = _clean_text(raw.get("voucher_number", pd.Series(pd.NA, index=raw.index)))
    df["amount"] = pd.to_numeric(raw["amount"], errors="coerce")
    df["check_date"] = full_date
    df["check_year"] = full_date.dt.year.fillna(year_only)
    df["department"] = _clean_text(raw.get("department_name", pd.Series(pd.NA, index=raw.index)))
    df["contract_number"] = _clean_text(
        raw.get("contract_number", pd.Series(pd.NA, index=raw.index))
    )
    df["vendor_name_raw"] = _clean_text(raw["vendor_name"])
    df["is_annual_rollup"] = full_date.isna() & year_only.notna()

    before = len(df)
    df = df.dropna(subset=["amount", "check_year", "vendor_name_raw"]).copy()
    df.attrs["dropped_rows"] = before - len(df)
    df["check_year"] = df["check_year"].astype(int)

    # Stable row id: content hash plus an occurrence counter, so genuinely identical
    # source rows (which do exist) still get distinct, reproducible ids.
    key = (
        df["voucher_number"].fillna("")
        + "|" + df["amount"].map("{:.2f}".format)
        + "|" + df["check_date"].dt.strftime("%Y-%m-%d").fillna(df["check_year"].astype(str))
        + "|" + df["vendor_name_raw"]
        + "|" + df["department"].fillna("")
        + "|" + df["contract_number"].fillna("")
    )
    occurrence = key.groupby(key).cumcount().astype(str)
    df["payment_id"] = (key + "#" + occurrence).map(
        lambda k: hashlib.sha1(k.encode()).hexdigest()[:16]
    )
    df = df[list(PAYMENTS_SCHEMA.columns)]
    return PAYMENTS_SCHEMA.validate(df.reset_index(drop=True))
