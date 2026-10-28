import json

from leakage import ingest
from leakage.ingest import SocrataSource, load_snapshot


def _fake_api(rows_updated_at: int, total: int, calls: list):
    def fake_get(session, url, params=None, retries=5):
        calls.append(params)
        if url.endswith(".json") and "/api/views/" in url:
            return {"rowsUpdatedAt": rows_updated_at}
        offset, limit = params["$offset"], params["$limit"]
        return [{"amount": str(i), "vendor_name": f"V{i}"}
                for i in range(offset, min(offset + limit, total))]
    return fake_get


def test_paginates_and_writes_manifest(tmp_path, monkeypatch):
    calls: list = []
    monkeypatch.setattr(ingest, "_get", _fake_api(100, 25, calls))
    src = SocrataSource("t", "example.org", "abcd-1234", page_size=10)
    snap = ingest.ingest(src, tmp_path)
    manifest = json.loads((snap / "manifest.json").read_text())
    assert manifest["total_rows"] == 25
    assert [p["rows"] for p in manifest["pages"]] == [10, 10, 5]
    assert all(c["$order"] == ":id" for c in calls if c)
    assert len(load_snapshot(snap)) == 25


def test_unchanged_source_reuses_snapshot(tmp_path, monkeypatch):
    calls: list = []
    monkeypatch.setattr(ingest, "_get", _fake_api(100, 5, calls))
    src = SocrataSource("t", "example.org", "abcd-1234", page_size=10)
    first = ingest.ingest(src, tmp_path)
    n_calls = len(calls)
    second = ingest.ingest(src, tmp_path)
    assert first == second
    assert len(calls) == n_calls + 1  # only the metadata check
