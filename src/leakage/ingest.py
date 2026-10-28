"""Snapshot ingestion from a Socrata (SODA) open-data endpoint.

Each run writes one immutable snapshot: raw JSON pages converted to Parquet plus a
manifest with row counts and SHA-256 hashes. If the source's `rowsUpdatedAt` hasn't
moved since the last snapshot, the run is a no-op, so re-running is always safe.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SocrataSource:
    name: str
    domain: str
    dataset_id: str
    page_size: int = 50_000

    @property
    def resource_url(self) -> str:
        return f"https://{self.domain}/resource/{self.dataset_id}.json"

    @property
    def metadata_url(self) -> str:
        return f"https://{self.domain}/api/views/{self.dataset_id}.json"


CHICAGO_PAYMENTS = SocrataSource(
    name="chicago_payments", domain="data.cityofchicago.org", dataset_id="s4vu-giwb"
)


def _get(session: requests.Session, url: str, params: dict | None = None, retries: int = 5):
    for attempt in range(retries):
        try:
            resp = session.get(url, params=params, timeout=120)
            if resp.status_code == 200:
                return resp.json()
            log.warning("HTTP %s from %s (attempt %d)", resp.status_code, url, attempt + 1)
        except requests.RequestException as exc:
            log.warning("request failed: %s (attempt %d)", exc, attempt + 1)
        time.sleep(2**attempt)
    raise RuntimeError(f"giving up on {url} after {retries} attempts")


def latest_snapshot(raw_root: Path, source: SocrataSource) -> Path | None:
    base = raw_root / source.name
    if not base.exists():
        return None
    snaps = sorted(p for p in base.iterdir() if (p / "manifest.json").exists())
    return snaps[-1] if snaps else None


def ingest(source: SocrataSource, raw_root: Path, force: bool = False) -> Path:
    """Download a full snapshot of `source` unless the newest one is still current."""
    session = requests.Session()
    meta = _get(session, source.metadata_url)
    rows_updated_at = int(meta.get("rowsUpdatedAt") or 0)

    prev = latest_snapshot(raw_root, source)
    if prev and not force:
        manifest = json.loads((prev / "manifest.json").read_text())
        if manifest.get("source_rows_updated_at") == rows_updated_at:
            log.info("source unchanged since %s; reusing snapshot", prev.name)
            return prev

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = raw_root / source.name / stamp
    out.mkdir(parents=True, exist_ok=True)

    pages = []
    offset = 0
    while True:
        # Ordering by the system row id gives stable pagination across pages.
        rows = _get(
            session,
            source.resource_url,
            {"$limit": source.page_size, "$offset": offset, "$order": ":id"},
        )
        if not rows:
            break
        page_no = len(pages)
        path = out / f"page_{page_no:04d}.parquet"
        pd.DataFrame(rows).astype("string").to_parquet(path, index=False)
        pages.append(
            {
                "file": path.name,
                "rows": len(rows),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
        log.info("page %d: %d rows", page_no, len(rows))
        offset += len(rows)
        if len(rows) < source.page_size:
            break

    manifest = {
        "source": source.name,
        "resource_url": source.resource_url,
        "source_rows_updated_at": rows_updated_at,
        "snapshot_utc": stamp,
        "total_rows": sum(p["rows"] for p in pages),
        "pages": pages,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out


def load_snapshot(snapshot: Path) -> pd.DataFrame:
    manifest = json.loads((snapshot / "manifest.json").read_text())
    frames = [pd.read_parquet(snapshot / p["file"]) for p in manifest["pages"]]
    df = pd.concat(frames, ignore_index=True)
    if len(df) != manifest["total_rows"]:
        raise ValueError(f"snapshot row count {len(df)} != manifest {manifest['total_rows']}")
    return df
