# Deviations from docs/PREREGISTRATION.md

1. **Open-weight runtime (2026-10-22, before any open-weight generation).** The pre-registration
   named Ollama as the local runtime. The Ollama Windows package (1.58 GB) downloaded at about
   1 MB/s and was abandoned; the open-weight model is served with the llama.cpp server
   (release b11829, Vulkan build) instead. Sampling controls (temperature 0.8, per-generation
   seed 1000+i, max 4096 output tokens) are equivalent. Prompts, episodes and scoring are
   unchanged.

2. **Post-hoc descriptors (added after seeing results; primary outcome unchanged).** Because raised
   exceptions are scored as abstentions, a generation that crashes on every input attains
   "correct abstention" on all 20 abstain-desired conditions. After observing that most open-weight
   generations under conditions B and C never ran successfully, we added three descriptors that the
   pre-registration did not list: (a) *functional generation* = correct recovery on the clean control
   in at least one snapshot; (b) *balanced accuracy* = mean of recovery rate (recover-desired) and
   abstention rate (abstain-desired), for which a constant rejector scores 50%; (c) share of
   non-accepted episodes that were explicit `abstain` results rather than exceptions. These are
   labelled post hoc in the paper.
3. **Pooled condition estimate restricted to Claude models (post hoc).** The pre-registration did not
   specify pooling. The headline pooled estimate uses the three Claude models (all generations of
   which produced runnable code); the open-weight model is reported separately because its low
   silent-error rate under guidance reflects non-functional code. A four-model pooled table is also
   reported (`results_v2/tables/T7b_llm_pooled_condition.csv`).

4. **Changes after an independent reviewer-style critique (2026-10-22, after main results).**
   a. *Held-out scenarios H01-H03* (re-keyed replays, payment-code swap, 10% fare inflation) were
      defined, materialized and evaluated by re-executing the same 120 archived generations (no new
      model calls) and all baselines. They were added because the guided prompts state most labelling
      rules; results are exploratory. Existing episodes were verified byte-identical after adding them.
   b. *Valid detection* descriptor: an abstention counts only if the same generation recovered the
      clean control on the same snapshot.
   c. *Terminology:* fault "conditions" are called *scenarios*; prompt conditions are A/B/C; compound
      scenarios C01-C04 are displayed as X01-X04 (internal IDs unchanged); baseline B1 is displayed as
      "normalizing"; prompt B is called "contract-guided" (internal ID `B_schema_aware` unchanged).
   d. *Wording:* the document `docs/PREREGISTRATION.md` is a frozen analysis plan committed to the local
      git repository immediately before the first generation (commit 44114e7); it is not an external
      registration, and the paper now says so. The file itself is left unmodified.
   e. A factual error ("18 metric-changing faults"; correct: 15) was corrected and is now macro-bound.

5. **Correction of a harness defect (2026-10-23, after the main results; no new model calls).**
   *Defect.* The evaluation harness used for the archived main-suite run (commit 5a52fcf) started each
   sandbox interpreter with `stderr=subprocess.PIPE` and read the pipe only after the process ended.
   One open-weight generation (`test__qwen2.5-coder-7b-instruct-q4_k_m__A_unassisted__02`) emits two
   pandas `FutureWarning`s (about 290 bytes) per episode; after 14 episodes the 4 KB Windows pipe buffer
   was full, the interpreter blocked on its next write, and the watchdog scored the blocked episode as a
   timeout (`invalid_output`). The pattern repeated after every restart, giving 24 spurious timeouts
   (two per snapshot, at batch positions 13/14 and 25/26). The same-machine re-run used the same code,
   so it reproduced the artifact (it was not independent evidence).
   *Detection.* A consolidation audit found that a clean-environment rebuild (which already used the
   later harness that writes stderr to a file) differed from the archive in exactly these 24 episodes,
   and that the paper's explanation ("run times close to the watchdog") contradicted the records
   (the episodes complete in about 0.02 s). The deadlock was reproduced with the old harness
   (blocks after 14 completed jobs) and not with the current one.
   *Correction.* The main-suite (46,080) and pilot evaluations were re-executed twice with the current
   harness; the archived generations were not re-sampled. The corrected outcomes are identical to the
   independent clean-environment rebuild in all 46,080 episodes (outcomes and metric values). Relative
   to the superseded archive, only those 24 episodes changed, from `invalid_output` to `silent_error`.
   All tables, figures, macros and the manuscript were regenerated. A regression test
   (`test_sandbox_verbose_stderr_does_not_deadlock`) fails on the old harness and passes on the current
   one. The superseded records are kept in `results_v2/llm/superseded_pipe_deadlock/`. The held-out
   results were not affected (three episodes per batch; a fresh-environment rebuild with the current
   harness reproduced 4,320/4,320 outcomes). The primary outcome and the classification rule are unchanged.

6. **Release-time redaction (2026-11-15, public package only; no experiment change).** Quoted source-data values in `reason` cells of the packaged episode records (918 cells) were replaced by `<value>` with `pipeshift/redact_records.py`. No other cell, outcome or number changed; the private archive keeps the original records. `llm_evaluate.sanitize` and `run_baselines.py` now apply the same redaction when new records are written.
