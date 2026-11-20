# Superseded LLM evaluation records (harness pipe deadlock)

These files are the main-suite, same-machine re-run and pilot results produced by the evaluation harness
of commit 5a52fcf. That harness did not drain the sandbox's stderr pipe, so 24 episodes of
`test__qwen2.5-coder-7b-instruct-q4_k_m__A_unassisted__02` were wrongly scored as timeouts.
They are retained only for transparency. The reported results are in `../episode_results_*.csv`.
See `docs/DEVIATIONS.md`, item 5.
