"""Validation overhead on a full month of admissible records (single machine, indicative only)."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from baselines import BASELINES, make_workflow, reference_profile
from faults import manifest_for
from snapshots import admissible_frame, write_parquet

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--month', default='2025-03')
    ap.add_argument('--repeats', type=int, default=5)
    args = ap.parse_args()
    out = ROOT / 'results_v2' / 'timing'
    out.mkdir(parents=True, exist_ok=True)
    work = ROOT / 'work' / 'timing'
    work.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    frame = admissible_frame(con, ROOT / 'data' / 'raw' / f'yellow_tripdata_{args.month}.parquet', args.month)
    frame.insert(0, 'record_id', np.arange(len(frame), dtype=np.int64))
    data = work / 'trips.parquet'
    write_parquet(frame, data)
    man = manifest_for(data, args.month, f'{args.month}_full', 1)
    (work / 'manifest.json').write_text(json.dumps(man))
    ref = reference_profile(frame)
    flows = {k: make_workflow(set(v), reference=ref) for k, v in BASELINES.items()}
    flows['sha256_only'] = lambda d, m: hashlib.sha256(Path(d).read_bytes()).hexdigest()
    rows = []
    for rep in range(args.repeats + 1):
        for name, fn in flows.items():
            t0 = time.perf_counter()
            res = fn(str(data), str(work / 'manifest.json'))
            dt = time.perf_counter() - t0
            if rep == 0:
                continue  # warm-up
            status = res if isinstance(res, str) else res['status']
            rows.append({'workflow': name, 'repeat': rep, 'seconds': dt, 'rows': len(frame),
                         'file_mb': data.stat().st_size / 2**20, 'status': str(status)[:12]})
    df = pd.DataFrame(rows)
    df.to_csv(out / 'full_month_timing.csv', index=False)
    (out / 'timing_env.json').write_text(json.dumps({'platform': platform.platform(), 'processor': platform.processor(),
                                                     'month': args.month, 'rows': len(frame)}, indent=2))
    print(df.groupby('workflow').seconds.median().round(3).to_string())


if __name__ == '__main__':
    main()
