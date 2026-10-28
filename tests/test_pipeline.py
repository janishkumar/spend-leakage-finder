import json

import duckdb
import openpyxl

from leakage.pipeline import RunConfig, run
from leakage.report import write_findings, write_review_pack
from leakage.scorer import InjectionConfig
from tests.conftest import small_raw


def _write_snapshot(root):
    snap = root / "chicago_payments" / "20260101T000000Z"
    snap.mkdir(parents=True)
    small_raw().to_parquet(snap / "page_0000.parquet", index=False)
    (snap / "manifest.json").write_text(json.dumps({
        "source": "chicago_payments",
        "resource_url": "https://example.org/resource/test.json",
        "source_rows_updated_at": 1,
        "snapshot_utc": "20260101T000000Z",
        "total_rows": len(small_raw()),
        "pages": [{"file": "page_0000.parquet", "rows": len(small_raw()), "sha256": "x"}],
    }))


def test_offline_run_then_report(tmp_path):
    _write_snapshot(tmp_path / "raw")
    out = tmp_path / "out"
    metrics = run(RunConfig(raw_root=tmp_path / "raw", out_dir=out, offline=True,
                            injection=InjectionConfig(n=100, seed=5), er_eval_n=20))
    assert metrics["ledger"]["annual_rollups"] == 2
    assert set(metrics["candidates"]["by_tier"]) >= {"T1", "T2", "T3", "T4"}

    con = duckdb.connect(str(out / "leakage.duckdb"), read_only=True)
    tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    assert {"payments", "review_queue", "vendors", "benford_departments"} <= tables
    con.close()

    md = write_findings(out, tmp_path / "findings.md").read_text()
    assert "Do not edit by hand" in md and "Top 10 pairs by risk score" in md
    wb = openpyxl.load_workbook(write_review_pack(out, tmp_path / "pack.xlsx"))
    assert {"Duplicate candidates", "ROI sensitivity", "Read me"} <= set(wb.sheetnames)
