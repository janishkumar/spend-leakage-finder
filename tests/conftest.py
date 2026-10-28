import numpy as np
import pandas as pd
import pytest


def raw_rows(rows: list[dict]) -> pd.DataFrame:
    """Rows shaped like the Chicago SODA extract (all strings)."""
    df = pd.DataFrame(rows)
    for col in ["voucher_number", "amount", "check_date", "department_name",
                "contract_number", "vendor_name"]:
        if col not in df:
            df[col] = pd.NA
    return df.astype("string")


def small_raw() -> pd.DataFrame:
    """A raw extract with one planted example of each duplicate tier plus routine noise."""
    rng = np.random.default_rng(0)
    rows = []
    vendors = ["ACME PLUMBING INC", "BETA ELECTRIC CO", "GAMMA SUPPLY LLC", "DELTA PAVING CORP"]
    for i in range(400):
        d = pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(rng.integers(0, 360)))
        rows.append({
            "voucher_number": f"PV{100000 + i}",
            "amount": f"{rng.uniform(150, 9000):.2f}",
            "check_date": d.strftime("%m/%d/%Y"),
            "department_name": "DEPT OF FLEET",
            "contract_number": "DV",
            "vendor_name": vendors[i % 4],
        })
    rows += [
        # T1 exact duplicate row
        {"voucher_number": "PV900001", "amount": "1234.56", "check_date": "03/01/2024",
         "department_name": "DEPT OF FLEET", "contract_number": "DV", "vendor_name": "OMEGA LLC"},
        {"voucher_number": "PV900001", "amount": "1234.56", "check_date": "03/01/2024",
         "department_name": "DEPT OF FLEET", "contract_number": "DV", "vendor_name": "OMEGA LLC"},
        # T2 same vendor + amount, different voucher, 9 days apart
        {"voucher_number": "PV900010", "amount": "7777.00", "check_date": "05/01/2024",
         "department_name": "DEPT OF FLEET", "contract_number": "DV", "vendor_name": "SIGMA CORP"},
        {"voucher_number": "PV900011", "amount": "7777.00", "check_date": "05/10/2024",
         "department_name": "DEPT OF FLEET", "contract_number": "DV", "vendor_name": "SIGMA CORP"},
        # T3 same day + amount, vendor keyed two ways
        {"voucher_number": "PV900020", "amount": "4321.00", "check_date": "06/03/2024",
         "department_name": "DEPT OF FLEET", "contract_number": "DV",
         "vendor_name": "KAPPA CONSTRUCTION, INC."},
        {"voucher_number": "PV900021", "amount": "4321.00", "check_date": "06/03/2024",
         "department_name": "DEPT OF FLEET", "contract_number": "DV",
         "vendor_name": "Kappa Construction Inc"},
        # T4 transposed amount
        {"voucher_number": "PV900030", "amount": "5575.50", "check_date": "07/01/2024",
         "department_name": "DEPT OF FLEET", "contract_number": "DV", "vendor_name": "ZETA MECH"},
        {"voucher_number": "PV900031", "amount": "5557.50", "check_date": "07/03/2024",
         "department_name": "DEPT OF FLEET", "contract_number": "DV", "vendor_name": "ZETA MECH"},
        # annual roll-up row: must be ignored by detectors
        {"amount": "99999.00", "check_date": "2023", "contract_number": "12345",
         "vendor_name": "ROLLUP VENDOR"},
        {"amount": "99999.00", "check_date": "2023", "contract_number": "12345",
         "vendor_name": "ROLLUP VENDOR"},
    ]
    return raw_rows(rows)


@pytest.fixture
def small_ledger() -> pd.DataFrame:
    from leakage.contract import standardise_chicago

    return standardise_chicago(small_raw())
