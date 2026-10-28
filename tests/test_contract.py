import pandas as pd
import pandera.errors
import pytest

from leakage.contract import PAYMENTS_SCHEMA, standardise_chicago
from tests.conftest import raw_rows


def test_rollup_rows_are_flagged_not_dated():
    df = standardise_chicago(raw_rows([
        {"amount": "10.00", "check_date": "2024", "vendor_name": "A"},
        {"voucher_number": "PV1", "amount": "20.00", "check_date": "12/31/2025",
         "vendor_name": "B"},
    ]))
    rollup = df.set_index("vendor_name_raw").loc["A"]
    assert bool(rollup["is_annual_rollup"]) and pd.isna(rollup["check_date"])
    assert rollup["check_year"] == 2024
    payment = df.set_index("vendor_name_raw").loc["B"]
    assert not payment["is_annual_rollup"]
    assert payment["check_date"] == pd.Timestamp("2025-12-31")


def test_identical_source_rows_get_distinct_stable_ids():
    row = {"voucher_number": "PV1", "amount": "5.00", "check_date": "01/02/2024",
           "vendor_name": "X"}
    a = standardise_chicago(raw_rows([row, row]))
    b = standardise_chicago(raw_rows([row, row]))
    assert a["payment_id"].is_unique
    assert a["payment_id"].tolist() == b["payment_id"].tolist()


def test_unparseable_rows_are_dropped_and_counted():
    df = standardise_chicago(raw_rows([
        {"amount": "not a number", "check_date": "01/01/2024", "vendor_name": "A"},
        {"amount": "1.00", "check_date": "01/01/2024", "vendor_name": "  "},
        {"amount": "1.00", "check_date": "01/01/2024", "vendor_name": "OK"},
    ]))
    assert len(df) == 1
    assert df.attrs["dropped_rows"] == 2


def test_schema_rejects_extra_columns(small_ledger):
    with pytest.raises((pandera.errors.SchemaError, pandera.errors.SchemaErrors)):
        PAYMENTS_SCHEMA.validate(small_ledger.assign(extra=1))
