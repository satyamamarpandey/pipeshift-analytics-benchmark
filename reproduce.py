"""One-command reproduction of the PipeShift v2 results.

    python reproduce.py                 # full pipeline from the official TLC files
    python reproduce.py --skip-download # reuse data/raw if already present
    python reproduce.py --quick         # skip intensity sweep timing and legacy v1 re-run

Model generations are NOT re-sampled (that requires provider credentials and would be a new
sample); the archived generated code in results_v2/llm/generations/ is re-executed in the
sandbox and re-scored against a freshly rebuilt oracle. The script then compares the
regenerated episode-level records with the archived ones and reports any difference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
PS = ROOT / 'pipeshift'
PY = sys.executable


def step(title: str, args: list[str], cwd: Path = PS) -> None:
    print(f'\n=== {title}', flush=True)
    subprocess.run([PY, *args], cwd=cwd, check=True)


def stable(df: pd.DataFrame, drop=('seconds',)) -> str:
    cols = [c for c in df.columns if c not in drop]
    return hashlib.sha256(df[cols].sort_values(cols[:3]).to_csv(index=False).encode()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-download', action='store_true')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--out', type=Path, default=ROOT / 'reproduction_check')
    args = ap.parse_args()
    archive = ROOT / 'results_v2'
    keep = {'baselines': archive / 'baselines' / 'episode_results.csv',
            'llm': archive / 'llm' / 'episode_results_test.csv'}
    args.out.mkdir(exist_ok=True)
    for k, p in keep.items():
        if p.exists():
            shutil.copy2(p, args.out / f'archived_{k}.csv')
    if not args.skip_download:
        step('Acquire official TLC files (verifies recorded SHA-256)', ['acquire.py'])
    step('Profile natural data quality and draw stratified snapshots', ['snapshots.py'])
    step('Materialize fault-injected episodes', ['faults.py'])
    step('Compute independent oracle (DuckDB SQL + pure Python)', ['oracle.py'])
    step('Run deterministic baselines, ablations, trust-anchor test', ['run_baselines.py'])
    if not args.quick:
        step('Full-month overhead timing', ['timing.py'])
    llm_out = args.out / 'llm_episode_results_test.csv'
    llm_out.unlink(missing_ok=True)
    step('Re-execute archived LLM generations in sandbox', ['llm_evaluate.py', '--split', 'test', '--out', str(llm_out)])
    held_out = args.out / 'llm_episode_results_heldout.csv'
    held_out.unlink(missing_ok=True)
    step('Re-execute archived LLM generations on held-out scenarios',
         ['llm_evaluate.py', '--split', 'heldout', '--out', str(held_out)])
    report = {}
    arch_h = archive / 'llm' / 'episode_results_heldout.csv'
    if arch_h.exists():
        a = pd.read_csv(arch_h, keep_default_na=False, dtype=str)
        b = pd.read_csv(held_out, keep_default_na=False, dtype=str)
        mh = a.merge(b, on=['gen_id', 'episode_id'], suffixes=('', '_new'))
        report['heldout_pairs_compared'] = len(mh)
        report['heldout_same_outcome_fraction'] = float((mh.outcome == mh.outcome_new).mean())
    a = pd.read_csv(args.out / 'archived_baselines.csv', keep_default_na=False, dtype=str) \
        if (args.out / 'archived_baselines.csv').exists() else None
    b = pd.read_csv(keep['baselines'], keep_default_na=False, dtype=str)
    if a is not None:
        report['baselines_identical_outcomes'] = bool(
            a.set_index(['workflow', 'episode_id']).outcome.sort_index().equals(
                b.set_index(['workflow', 'episode_id']).outcome.sort_index()))
    if (args.out / 'archived_llm.csv').exists():
        a = pd.read_csv(args.out / 'archived_llm.csv', keep_default_na=False, dtype=str)
        b = pd.read_csv(llm_out, keep_default_na=False, dtype=str)
        m = a.merge(b, on=['gen_id', 'episode_id'], suffixes=('', '_new'))
        report['llm_pairs_compared'] = len(m)
        report['llm_same_outcome_fraction'] = float((m.outcome == m.outcome_new).mean())
        report['llm_differences'] = m[m.outcome != m.outcome_new][['gen_id', 'episode_id', 'fid', 'outcome', 'outcome_new']] \
            .to_dict('records')  # all differences, not a sample
        shutil.copy2(llm_out, archive / 'llm' / 'episode_results_test_rerun.csv')
    step('Recompute tables and macros', ['analysis.py'])
    step('Regenerate figures', ['figures.py'])
    if not args.quick and (ROOT / 'benchmark').is_dir():  # the legacy v1 prototype is not in the public release
        print('\n=== Legacy v1 benchmark (synthetic + 300-row excerpt)')
        subprocess.run([PY, 'run_benchmark.py', '--out', str(args.out / 'legacy')], cwd=ROOT / 'benchmark', check=True)
        # The legacy 300-row excerpt is fetched from NYC Open Data (not redistributed) so its tests can run.
        fetched = subprocess.run([PY, 'fetch_tlc_socrata.py'], cwd=ROOT / 'benchmark')
        if fetched.returncode == 0:
            subprocess.run([PY, 'run_real_pilot.py', '--out', str(args.out / 'legacy_real')], cwd=ROOT / 'benchmark',
                           check=True)
            legacy = all((args.out / 'legacy_real' / f).read_bytes().replace(b'\r\n', b'\n') ==
                         (ROOT / 'results' / 'real' / f).read_bytes().replace(b'\r\n', b'\n')
                         for f in ('trials.csv', 'summary.csv'))
            report['legacy_real_excerpt_identical'] = legacy
    (args.out / 'reproduction_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    ok = (report.get('baselines_identical_outcomes', True)
          and report.get('llm_same_outcome_fraction', 1.0) >= 0.999
          and report.get('heldout_same_outcome_fraction', 1.0) >= 0.999)
    print('REPRODUCTION', 'PASSED' if ok else 'DIFFERENCES FOUND (see report)')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
