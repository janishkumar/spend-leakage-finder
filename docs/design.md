# Spend Leakage Finder — design

Date: 2026-10-28. Author: Janish Kumar.

## Intent

Mid-market CFOs lose cash to payment leakage: the same invoice paid twice, a vendor that exists three times in the vendor master, purchases split to stay under an approval limit. Recovery-audit firms find these and charge a share of what's recovered. This project is a solution accelerator a consultancy (Practus, Office of CFO / Performance Improvement) can run on a client's AP ledger in days and pitch with a dollar figure.

It is demonstrated on a **real public ledger**: City of Chicago vendor payments (data.cityofchicago.org dataset `s4vu-giwb`, ~360K voucher-level payments). No client data, ever.

## Goals

1. Ingest a real payments ledger incrementally and reproducibly, with a data contract enforced at the boundary.
2. Resolve vendor names into a clean vendor master (entity resolution) and surface likely vendor-master duplicates.
3. Generate duplicate-payment candidates in explainable tiers and rank them with an ML scorer.
4. Run classic forensic-accounting tests: Benford's law, split purchasing below thresholds, vendor spend spikes.
5. Hand a reviewer an evidence-backed review queue (dashboard + Excel workbook) and an ROI estimate with explicit, adjustable assumptions.
6. Ship it like production: tests, CI/CD, Docker, one-command run.

## Non-goals

- Spend categorisation into a procurement taxonomy (roadmap; needs labelled descriptions).
- Multi-source ingestion beyond one ledger tonight. The source adapter is pluggable; a second ledger is a config + adapter.
- An LLM in the critical path. No API key is assumed; everything runs offline after download.

## Anti-goals (failure even if it "works")

- Calling a candidate an overpayment. Public data has no invoice numbers or line numbers, so identical rows can be legitimate split lines. Everything is a **candidate for review**, and the README says so.
- Any metric in the README not produced by code in this repo.
- Headline dollar figures without the assumption behind them.

## Design decisions

- **Stack:** Python 3.12, uv, DuckDB (local warehouse), pandas, scikit-learn, rapidfuzz, pandera (contract), Streamlit (UI), openpyxl (Excel review pack), pytest, ruff, GitHub Actions, Docker.
- **Ingestion:** Socrata SODA API, paged by `$offset` with a stable `$order`, retries with backoff, raw pages written to Parquet with a manifest (row counts, SHA-256). Year-only aggregate rows (`check_date` = `YYYY`, no voucher) are kept in raw but excluded from duplicate tests: they're annual roll-ups, not payments.
- **Vendor ER:** deterministic normalisation (case, punctuation, legal suffixes, `&`/`AND`, `THE`) → blocking on first token + sorted-token key → similarity (rapidfuzz token_set_ratio + char n-gram TF-IDF cosine) → union-find clusters above threshold → `vendor_id`. Evaluated on synthetic perturbations of real vendor names (typos, suffix swaps, word order) since there's no ground truth.
- **Duplicate candidates (tiers):**
  - T1 exact duplicate rows (same voucher, vendor, amount, date): likely split lines, reported but lowest priority.
  - T2 same resolved vendor + same amount, different voucher, within 30 days.
  - T3 same amount + same date, *different raw vendor names that resolve to one entity* (vendor-master duplicate).
  - T4 near-amount (±0.5%) or digit-transposed amount, same vendor, within 14 days, different voucher.
- **ML scorer:** gradient-boosted classifier trained on **injected** duplicates (a realistic perturbation generator: re-keyed voucher, date shift, amount transposition, vendor typo) vs. real candidate pairs. Features include day gap, amount size, roundness, vendor recurrence regularity (suppresses rent/subscription false positives), contract match, department match, name similarity. Measured as precision/recall on held-out injections. This is the standard evaluation approach in audit analytics when labels don't exist, and the README says so.
- **Forensics:** first-digit Benford test with MAD conformity (Nigrini thresholds) by department; split-purchase detection (≥2 payments, same vendor + department, within 7 days, each below a threshold, sum above it), with thresholds discovered empirically from amount clustering; vendor-month spend spikes by robust z-score.
- **ROI:** dollars in each tier × an adjustable confirmation rate (default deliberately conservative) − review cost. Shown as a calculator, never as a claim.

## Open implementation details

File layout, thresholds within reason, chart choices, test fixtures (small synthetic fixtures for CI, no network in tests).
