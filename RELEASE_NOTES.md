# PipeShift v1.1.0

Compared with `v1.0.0` (2026-10-23, later made private):
- Results, generated programs, prompts, configuration, baselines and tests are unchanged. The exporter was re-run from the frozen private commit f6d7c39; see `PUBLIC_EXPORT_MANIFEST.json`.
- Redaction: quoted source-data values in `reason` cells (918 cells; timestamps and concatenated fares quoted by library error messages) are replaced by `<value>`; no other cell, outcome or number changed (`pipeshift/redact.py`, `redact_records.py`, `tests/test_redact.py`, README *Redacted values*, `docs/DEVIATIONS.md` item 6). The private archive keeps the originals.
- History: this release starts a fresh repository history; `v1.0.0` and its asset (unredacted records) are not part of it.
- Changed: `pipeshift/figures.py`, `pipeshift/llm_evaluate.py`, `pipeshift/run_baselines.py` (redaction at source), `results_v2/**/episode_results*.csv` and `docs/records/**/episode_results*.csv` (redaction only), and Fig. 3 (`figures_v2/fig_fault_heatmap.{pdf,png}` drawn with vector cells instead of a 100-ppi raster; data identical), `README.md` (expected results, reproduction of tables and figures, limitations, release history), `THIRD_PARTY_NOTICES.md` (provider terms).
- Added: `CITATION.cff`, `.zenodo.json`, `RELEASE_NOTES.md`, `RELEASE_MANIFEST_SHA256.txt`.
- Known limits: same-machine, AI-assisted re-execution of archived programs; no fresh model sampling; no independent human replication; the legacy v1 prototype and its tests are not included; some LLM-output licensing questions are author decisions (see `THIRD_PARTY_NOTICES.md`).
