"""Run deterministic baselines, ablations and the trust-anchor experiment on all episodes.

Outputs (results_v2/baselines/):
  episode_results.csv   one row per (workflow, episode)
  run_meta.json         environment and counts
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from baselines import BASELINES, COMPONENTS, make_workflow, reference_profile
from faults import prev_month
from scoring import classify
from redact import redact_values
from spec import METRICS

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'work'


def workflow_plan() -> list[tuple[str, frozenset, bool, str]]:
    """(name, components, uses_registry, suites)"""
    plan = [(name, frozenset(c), False, 'main+sweep+heldout') for name, c in BASELINES.items()]
    full = BASELINES['B6_all_checks']
    for comp in sorted(full):
        plan.append((f'ABL_minus_{comp}', frozenset(full - {comp}), False, 'main'))
    plan.append(('B7_contract_registry', frozenset(BASELINES['B5_contract_manifest']), True, 'main+sweep+heldout'))
    return plan


def _run_chunk(args):
    rows, plan = args
    out = []
    refs = {}
    for ep in rows:
        rep_key = f"{prev_month(ep['month'])}_{ep['replicate']}"
        if rep_key not in refs:
            refs[rep_key] = reference_profile(pd.read_parquet(WORK / 'snapshots' / f'{rep_key}_snapshot.parquet'))
        registry = json.loads((WORK / 'registry' / f"{ep['episode_id']}.json").read_text())
        truth = {m: ep[f'truth_{m}'] for m in METRICS}
        d = WORK / 'episodes' / ep['episode_id']
        for name, comps, uses_reg, suites in plan:
            if ep['suite'] not in suites.split('+'):
                continue
            wf = make_workflow(set(comps), reference=refs[rep_key], registry=registry if uses_reg else None)
            t0 = time.perf_counter()
            try:
                res = wf(str(d / 'trips.parquet'), str(d / 'manifest.json'))
            except Exception as exc:  # noqa: BLE001 - any failure is a non-silent signal
                res = {'_exception': f'{type(exc).__name__}: {exc}'[:200]}
            dt = time.perf_counter() - t0
            sc = classify(res, ep['desired'], truth)
            metrics = res.get('metrics') if isinstance(res, dict) else None
            out.append({'workflow': name, 'episode_id': ep['episode_id'], 'seconds': dt,
                        'reason': redact_values(res.get('_exception', res.get('reason', '')))[:200],
                        'metrics_json': json.dumps(metrics, sort_keys=True) if metrics else '', **sc})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=ROOT / 'results_v2' / 'baselines')
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    eps = pd.read_csv(WORK / 'oracle' / 'episodes.csv', dtype={'intensity': str}, keep_default_na=False)
    plan = workflow_plan()
    records = eps.to_dict('records')
    chunks = [records[i::args.workers * 4] for i in range(args.workers * 4)]
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        results = [r for part in ex.map(_run_chunk, [(c, plan) for c in chunks]) for r in part]
    res = pd.DataFrame(results)
    res = res.merge(eps[['episode_id', 'suite', 'month', 'replicate', 'snapshot', 'fid', 'fault', 'family',
                         'intensity', 'desired', 'manifest_mode']], on='episode_id', validate='many_to_one')
    res = res.sort_values(['workflow', 'suite', 'snapshot', 'fid', 'intensity']).reset_index(drop=True)
    res.to_csv(args.out / 'episode_results.csv', index=False)
    stable = res.drop(columns=['seconds']).to_csv(index=False).encode()
    meta = {'episodes': int(len(eps)), 'rows': int(len(res)), 'workflows': [p[0] for p in plan],
            'components': list(COMPONENTS), 'wall_seconds': round(time.time() - t0, 1),
            'results_sha256_excluding_timing': hashlib.sha256(stable).hexdigest(),
            'python': sys.version.split()[0], 'platform': platform.platform(), 'pandas': pd.__version__}
    (args.out / 'run_meta.json').write_text(json.dumps(meta, indent=2) + '\n')
    main_ = res[res.suite == 'main']
    print(pd.crosstab(main_.workflow, main_.outcome).to_string())
    print(json.dumps({k: meta[k] for k in ('rows', 'wall_seconds', 'results_sha256_excluding_timing')}))


if __name__ == '__main__':
    main()
