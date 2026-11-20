# PipeShift v2: frozen evaluation plan (written before any model generation)

Frozen: 2026-10-22, git commit containing this file. All model generations and their
evaluation occur after this commit. Deviations will be listed in `docs/DEVIATIONS.md`.

## Research questions
- RQ1: How often do analytics workflows return numerically wrong results without signalling a
  problem (silent errors) when their input data or pipeline metadata change?
- RQ2: How do independently generated LLM workflows compare under three prompting conditions:
  A unassisted, B schema-aware (data contract), C provenance-guided (contract + manifest
  verification protocol)?
- RQ3: Which deterministic validation components (schema normalization, contract checks,
  statistical profiling, manifest hash/row-count verification, independent trust anchor) prevent
  which failures, and at what false-rejection cost?

## Data
Official NYC TLC Yellow Taxi monthly Parquet files, 2024-12 (prior-period reference only) and
2025-01..2025-12 (study periods); hashes in `data/source_manifest.json`. Per month, two disjoint
day-stratified samples of 5,000 admissible records (replicates A and B), plus disjoint
late-record (150) and long-trip (>= 5 mi, 5,000) pools. Natural data-quality conditions are
profiled on the full raw files and are excluded from clean snapshots by a documented filter.

## Episodes and ground truth
32 fault conditions (`pipeshift/spec.py`) x 24 snapshots = 768 main-suite episodes, plus an
intensity sweep of 8 faults x 4 levels x 24 snapshots = 768 episodes (deterministic baselines
only). Each fault has a pre-registered desired behaviour (recover or abstain). Truth = metrics
of the intended data file computed by DuckDB SQL and cross-checked by pure Python.

## Outcomes (per episode)
correct_recovery, correct_abstention, false_rejection, silent_error (status ok with any metric
outside tolerance), unverified_accept (status ok, numbers correct, but abstention desired),
invalid_output (no code, malformed result, crash or timeout). Raising an exception and returning
status "abstain" are both non-silent and scored the same.

Tolerances: trip_count exact; total_fare, mean_fare, mean_trip_distance 0.005 absolute;
card_share 5e-5 absolute.

**Primary outcome:** silent-error rate = silent_error / all evaluated episodes.
Secondary: false-rejection rate on recover-desired episodes; desired-behaviour rate
(correct_recovery + correct_abstention) / episodes; silent-error rate restricted to
metric-impacting faults; invalid-output rate; per-fault-family rates; generation-to-generation
consistency; latency.

## LLM evaluation
- Models: claude-haiku-5-5, claude-sonnet-5-5, claude-opus-5-5 via the Claude Code CLI in
  headless mode (tools disabled, custom system prompt, no persisted session, empty working
  directory); optionally one open-weight model via local Ollama if the runtime can be installed.
- Prompt SHA-256: A `834b3886...`, B `0b939c49...`, C `df680767...`, system `632284d3...`.
- Target n = 10 independent generations per model x condition (interleaved order).
- Pilot: 1 generation per model x condition evaluated on snapshot 2025-06_B only, to verify the
  harness. Pilot generations are excluded from all reported results. Prompts will not be
  changed after the pilot unless the harness is broken, which would be reported as a deviation.
- Test split: main-suite episodes of all 12 replicate-A snapshots (384 episodes per generation).
- All generations are kept, including ones with no extractable code or crashing code.
- Budget cap: USD 20 of reported list-price cost for all model calls (user cap USD 25).

## Statistical analysis
- Unit of independence for LLM results: the generation. Per-generation rates are averaged
  within model x condition; 95% CIs by cluster (generation-level) percentile bootstrap,
  10,000 resamples, seed 20261022.
- Condition contrasts within a model: difference of mean per-generation silent-error rates,
  bootstrap CI, two-sided permutation test on generation labels (10,000 permutations).
  Holm correction across the 3 pairwise contrasts per model. Effect size = risk difference.
- Deterministic baselines: proportions with Wilson 95% intervals; the snapshot (24) is the
  unit for per-fault rates. No p-values are reported for deterministic baselines because their
  outcomes are fixed by design.
- Synthetic legacy results, real-data fault-injection results, LLM results and targeted
  precision regressions are reported separately and never pooled.
