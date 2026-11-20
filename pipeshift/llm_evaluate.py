"""Execute every generated workflow on its evaluation split and score it against the oracle.

Splits: 'pilot' = main-suite episodes of snapshot 2025-06_B (harness check only; excluded
from reported results); 'test' = main-suite episodes of all replicate-A snapshots
(12 months x 32 faults = 384 episodes per generation).
Each (generation, snapshot) batch runs in a fresh sandboxed interpreter with a clean
environment and an empty working directory. A job that produces no result for
--job-timeout seconds is killed and scored 'invalid_output' (timeout); remaining jobs
are resumed in a new process.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from scoring import classify
from spec import METRICS

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'work'
LLM = ROOT / 'results_v2' / 'llm'
RUNNER = Path(__file__).resolve().parent / 'sandbox_runner.py'
PILOT_SNAPSHOT = '2025-06_B'


_PATH_RE = None


def sanitize(text: str) -> str:
    """Replace machine-specific absolute paths in captured messages with placeholders."""
    global _PATH_RE
    if _PATH_RE is None:
        import re
        roots = {str(WORK), WORK.as_posix(), str(ROOT), ROOT.as_posix(), tempfile.gettempdir(),
                 Path(tempfile.gettempdir()).as_posix(), str(Path.home()), Path.home().as_posix()}
        variants = set()
        for r in roots:
            variants |= {r, r.replace('\\', '\\\\')}
        alts = '|'.join(re.escape(v) for v in sorted(variants, key=len, reverse=True))
        _PATH_RE = re.compile(f'({alts})[^\'"\\s,)]*', re.IGNORECASE)
    from redact import redact_values  # quoted source-data values are not released (see redact.py)
    return redact_values(_PATH_RE.sub('<path>', text))


def split_episodes(eps: pd.DataFrame, split: str) -> pd.DataFrame:
    if split == 'heldout':  # exploratory held-out scenarios, replicate-A snapshots
        return eps[(eps.suite == 'heldout') & (eps.replicate == 'A')]
    m = eps[eps.suite == 'main']
    return m[m.snapshot == PILOT_SNAPSHOT] if split == 'pilot' else m[m.replicate == 'A']


def clean_env(tmp: str) -> dict:
    keep = {k: os.environ[k] for k in ('SYSTEMROOT', 'SystemRoot', 'WINDIR', 'NUMBER_OF_PROCESSORS') if k in os.environ}
    keep.update({'PATH': os.path.dirname(sys.executable), 'TEMP': tmp, 'TMP': tmp, 'PYTHONHASHSEED': '0',
                 'OMP_NUM_THREADS': '1', 'PYTHONIOENCODING': 'utf-8'})
    return keep


def kill_tree(proc: subprocess.Popen) -> None:
    """Terminate the runner and any child processes (defensive; plain kill sufficed in testing)."""
    if os.name == 'nt':
        subprocess.run(['taskkill', '/F', '/T', '/PID', str(proc.pid)], capture_output=True)
    else:
        import signal
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def run_batch(module: Path, jobs: list[dict], job_timeout: float) -> dict[str, dict]:
    results: dict[str, dict] = {}
    pending = list(jobs)
    while pending:
        with tempfile.TemporaryDirectory(prefix='pssbx_', ignore_cleanup_errors=True) as tmp:
            jobs_file = Path(tmp) / 'jobs.json'
            out_file = Path(tmp) / 'out.jsonl'
            err_file = Path(tmp) / 'stderr.txt'
            jobs_file.write_text(json.dumps(pending))
            with err_file.open('wb') as err:
                popen_kw = ({'creationflags': subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP}
                            if os.name == 'nt' else {'start_new_session': True})
                proc = subprocess.Popen([sys.executable, '-I', str(RUNNER), str(module), str(jobs_file),
                                         str(out_file)], cwd=tmp, env=clean_env(tmp), stdout=subprocess.DEVNULL,
                                        stderr=err, **popen_kw)
                seen, last_progress = 0, time.time()
                while True:
                    rc = proc.poll()
                    lines = out_file.read_text(encoding='utf-8').splitlines() if out_file.exists() else []
                    if len(lines) > seen:
                        seen, last_progress = len(lines), time.time()
                    if rc is not None:
                        break
                    if time.time() - last_progress > job_timeout + (20 if seen == 0 else 0):
                        kill_tree(proc)
                        break
                    time.sleep(0.05)
            lines = out_file.read_text(encoding='utf-8').splitlines() if out_file.exists() else []
            stderr = err_file.read_text(encoding='utf-8', errors='replace')[-300:] if err_file.exists() else ''
        for line in lines:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            results[rec['episode_id']] = rec
        done = {r for r in results}
        rest = [j for j in pending if j['episode_id'] not in done]
        if rest:
            # The first unfinished job hung or crashed the interpreter.
            bad = rest[0]
            results[bad['episode_id']] = {'episode_id': bad['episode_id'], 'seconds': job_timeout,
                                          'result': {'_timeout': True, '_stderr': stderr}}
            rest = rest[1:]
        pending = rest
    return results


def evaluate_generation(gen: dict, episodes: pd.DataFrame, job_timeout: float) -> list[dict]:
    module = LLM / 'generations' / f"{gen['gen_id']}.py"
    rows = []
    for _snap, group in episodes.groupby('snapshot', sort=True):
        jobs = [{'episode_id': e, 'data_path': str(WORK / 'episodes' / e / 'trips.parquet'),
                 'manifest_path': str(WORK / 'episodes' / e / 'manifest.json')} for e in group.episode_id]
        if module.exists():
            res = run_batch(module, jobs, job_timeout)
        else:
            res = {j['episode_id']: {'seconds': 0.0, 'result': 'NO_CODE_EXTRACTED'} for j in jobs}
        for ep in group.to_dict('records'):
            r = res[ep['episode_id']]
            result = r['result']
            truth = {m: ep[f'truth_{m}'] for m in METRICS}
            sc = classify(result, ep['desired'], truth)
            reason = ''
            if isinstance(result, dict):
                reason = sanitize(str(result.get('_exception') or result.get('reason') or ''))[:200]
            metrics = result.get('metrics') if isinstance(result, dict) else None
            rows.append({'gen_id': gen['gen_id'], 'model': gen['model'], 'condition': gen['condition'],
                         'episode_id': ep['episode_id'], 'snapshot': ep['snapshot'], 'month': ep['month'],
                         'fid': ep['fid'], 'fault': ep['fault'], 'family': ep['family'], 'desired': ep['desired'],
                         'seconds': r.get('seconds', 0.0), 'reason': reason,
                         'metrics_json': json.dumps(metrics, sort_keys=True) if isinstance(metrics, dict) else '',
                         **sc})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='test', choices=['test', 'pilot', 'heldout'])
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--job-timeout', type=float, default=30.0)
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--rerun', action='store_true', help='ignore existing results (reproducibility re-execution)')
    args = ap.parse_args()
    out = args.out or (LLM / f'episode_results_{args.split}.csv')
    eps = split_episodes(pd.read_csv(WORK / 'oracle' / 'episodes.csv', keep_default_na=False), args.split)
    gens = pd.concat([pd.read_csv(f) for f in sorted((LLM / 'logs').glob('generations_*.csv'))], ignore_index=True)
    gens = gens[gens.split == ('test' if args.split == 'heldout' else args.split)]
    done = set()
    if out.exists() and not args.rerun:
        done = set(pd.read_csv(out, usecols=['gen_id']).gen_id)
    todo = [g for g in gens.to_dict('records') if g['gen_id'] not in done]
    print(f'{len(todo)} generations x {len(eps)} episodes to evaluate', flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        for i, rows in enumerate(ex.map(lambda g: evaluate_generation(g, eps, args.job_timeout), todo)):
            df = pd.DataFrame(rows)
            df.to_csv(out, mode='a', header=not out.exists(), index=False)
            vc = df.outcome.value_counts().to_dict()
            print(f'[{i + 1}/{len(todo)}] {rows[0]["gen_id"]}: {vc}  ({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
