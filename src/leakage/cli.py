"""Command line: `leakage run`, `leakage report`."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from leakage.pipeline import RunConfig, run
from leakage.report import write_findings, write_review_pack


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="leakage")
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="ingest + detect; writes data/processed/")
    r.add_argument("--offline", action="store_true", help="reuse the latest local snapshot")
    sub.add_parser("report", help="write docs/findings.md and reports/review_pack.xlsx")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = Path("data/processed")
    if args.cmd == "run":
        run(RunConfig(offline=args.offline))
    else:
        print(write_findings(out, Path("docs/findings.md")))
        print(write_review_pack(out, Path("reports/review_pack.xlsx")))


if __name__ == "__main__":
    main()
