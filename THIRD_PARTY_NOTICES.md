# Third-party material and notices

The MIT License in `LICENSE` covers the author's code and documentation. It does not cover the third-party
material described below, which keeps its own terms.

## NYC TLC trip records (data source; not redistributed)

- **Publisher.** New York City Taxi and Limousine Commission (TLC):
  <https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page>.
- **Not in this repository.** No TLC Parquet file, sample or row-level table is included.
  `pipeshift/acquire.py` downloads the official files and verifies the SHA-256 hashes recorded in
  `data/source_manifest.json`. Use of the data is subject to the terms of TLC and NYC.
- **Accuracy.** TLC states that trip records are submitted by technology providers and that their accuracy
  and completeness are not guaranteed.
- **Values that do appear here.** Aggregates computed from the data: snapshot profiles, oracle metrics and
  timing. Exception messages in the `reason` column of the episode-level results that quoted field values
  (timestamps, concatenated fares) have those values replaced by `<value>` (see README, *Redacted values*).
  Counts and column names in generic messages are kept.

## Evaluated models and their outputs

- **Claude models.** Anthropic claude-haiku-5-5, claude-sonnet-5-5 and claude-opus-5-5 were queried through
  the Claude Code CLI. Their raw responses and extracted programs are in `results_v2/llm/generations/` and
  are included as research data for reproducibility. Session and request identifiers in the generation logs
  are replaced by truncated SHA-256 pseudonyms.
- **Qwen model.** Qwen2.5-Coder-7B-Instruct (Q4_K_M GGUF), released by the Qwen team under the Apache License
  2.0, was run locally with llama.cpp. The model weights and the runtime are not redistributed;
  `docs/environment/environment.md` gives the source and the SHA-256 of the model file. The model's
  generated programs are included as research data.
- **Provider terms.** Under Anthropic's Consumer Terms (effective 2025-10-08) and Commercial Terms (effective 2025-06-17), as read on 2026-11-15, Anthropic assigns its rights in Outputs to the user, subject to compliance with the terms. Neither document states a restriction on publishing Outputs, but both prohibit using the services to train competing AI models, and Anthropic's Usage Policy prohibits using outputs to train a model without authorization. The author states that the calls were made with the author's own personal Claude Code account; the subscription tier is not recorded, and the generation records contain no account or billing-plan field (the CLI reports `total_cost_usd` at list price whatever the plan). Anthropic's Claude Code documentation (read 2026-11-20) applies the Consumer Terms to Free, Pro, Max and Ultra users and the Commercial Terms to Team, Enterprise and API users; both term documents were unchanged when re-read on 2026-11-20. The terms in force on the generation dates (2026-10-22/23) were not retrieved. This describes the documents read and is not legal advice. Recipients must comply with the providers' terms; this data is released for research reproduction.
- **Status of the generated programs.** They are experimental observations, not recommended code. Many of
  them are wrong by design of the study.

## Python dependencies

No third-party package is vendored. The packages in `requirements-lock.txt` are installed from PyPI and are
governed by their own licenses.
