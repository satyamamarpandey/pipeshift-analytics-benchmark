"""Independent ground-truth oracle.

Truth metrics are computed over each episode's *intended* data file with DuckDB SQL
and cross-checked with a pure-Python computation (pyarrow -> Python floats,
math.fsum). Neither path shares code with any evaluated workflow. The oracle never
reads the observed (fault-injected) inputs.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow.parquet as pq

from spec import METRICS

ROOT = Path(__file__).resolve().parents[1]


def truth_sql(con: duckdb.DuckDBPyConnection, path: Path) -> dict:
    row = con.execute(f"""
        SELECT count(*), sum(fare_amount), avg(fare_amount), avg(trip_distance),
               avg(CASE WHEN payment_type = 1 THEN 1.0 ELSE 0.0 END),
               count(DISTINCT record_id)
        FROM read_parquet('{path.as_posix()}')""").fetchone()
    if row[0] != row[5]:
        raise AssertionError(f'target has duplicate record_id: {path}')
    return dict(zip(METRICS, (int(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]))))


def truth_python(path: Path) -> dict:
    t = pq.read_table(path, columns=['fare_amount', 'trip_distance', 'payment_type']).to_pydict()
    n = len(t['fare_amount'])
    total = math.fsum(t['fare_amount'])
    return {'trip_count': n, 'total_fare': total, 'mean_fare': total / n,
            'mean_trip_distance': math.fsum(t['trip_distance']) / n,
            'card_share': sum(1 for p in t['payment_type'] if p == 1) / n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', type=Path, default=ROOT / 'work')
    args = ap.parse_args()
    index = pd.read_csv(args.work / 'oracle' / 'episodes_index.csv')
    con = duckdb.connect()
    cache = {}
    max_dev = 0.0
    for tf in sorted(index.target_file.unique()):
        p = args.work / tf
        a, b = truth_sql(con, p), truth_python(p)
        assert a['trip_count'] == b['trip_count']
        for m in METRICS[1:]:
            max_dev = max(max_dev, abs(a[m] - b[m]))
            assert abs(a[m] - b[m]) <= 1e-6 * max(1.0, abs(b[m])), (tf, m, a[m], b[m])
        cache[tf] = a
    truth = pd.DataFrame([{'target_file': k, **{f'truth_{m}': v[m] for m in METRICS}} for k, v in cache.items()])
    out = index.merge(truth, on='target_file', how='left', validate='many_to_one')
    assert out[[f'truth_{m}' for m in METRICS]].notna().all().all()
    out.to_csv(args.work / 'oracle' / 'episodes.csv', index=False)
    print(f'{len(cache)} target files; SQL vs Python max abs deviation {max_dev:.3e}; {len(out)} episodes')


if __name__ == '__main__':
    main()
