import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pipeshift'))


def tlc_like(month: str, n: int, seed: int, rep_offset: int = 1, min_distance: float = 0.0) -> pd.DataFrame:
    """Small TLC-shaped frame with the delivered schema (test fixture only)."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(month + '-01')
    days = (start + pd.offsets.MonthBegin(1) - start).days
    pickup = start + pd.to_timedelta(rng.integers(0, days * 86400, n), unit='s')
    dist = np.round(rng.uniform(0.3, 12, n) + min_distance, 2)
    yyyymm = int(month.replace('-', ''))
    return pd.DataFrame({
        'record_id': np.int64(yyyymm) * 10**7 + rep_offset * 10**6 + np.arange(n, dtype=np.int64),
        'VendorID': rng.choice([1, 2], n).astype('int32'),
        'tpep_pickup_datetime': pickup.astype('datetime64[us]'),
        'tpep_dropoff_datetime': (pickup + pd.to_timedelta(rng.integers(60, 3600, n), unit='s')).astype('datetime64[us]'),
        'PULocationID': rng.integers(1, 264, n).astype('int32'),
        'DOLocationID': rng.integers(1, 264, n).astype('int32'),
        'trip_distance': dist,
        'payment_type': rng.choice([0, 1, 2, 4], n, p=[.2, .6, .15, .05]).astype('int64'),
        'fare_amount': np.round(3 + 2.8 * dist + rng.uniform(0, 5, n), 2),
        'tip_amount': np.round(rng.uniform(0, 5, n), 2),
        'total_amount': np.round(rng.uniform(5, 80, n), 2),
    })


@pytest.fixture(scope='session')
def mini_world(tmp_path_factory):
    """Materialized mini benchmark: months 2025-01 (prev) and 2025-02, replicate A, n=240."""
    from faults import materialize
    from oracle import truth_python, truth_sql
    from snapshots import write_parquet
    import duckdb
    root = tmp_path_factory.mktemp('world')
    snap = root / 'snapshots'
    snap.mkdir()
    for i, month in enumerate(('2025-01', '2025-02')):
        write_parquet(tlc_like(month, 240, 10 + i), snap / f'{month}_A_snapshot.parquet')
        write_parquet(tlc_like(month, 8, 20 + i, rep_offset=1).assign(
            record_id=lambda d: d.record_id + 5000), snap / f'{month}_A_late.parquet')
        write_parquet(tlc_like(month, 240, 30 + i, rep_offset=5, min_distance=5.0), snap / f'{month}_A_longtrip.parquet')
    index = materialize(root, snap, months=('2025-02',), replicates=('A',), include_sweep=True)
    con = duckdb.connect()
    truth = {tf: truth_sql(con, root / tf) for tf in index.target_file.unique()}
    for tf in truth:
        py = truth_python(root / tf)
        assert all(abs(truth[tf][k] - py[k]) < 1e-6 for k in truth[tf])
    for m in truth[index.target_file.iloc[0]]:
        index[f'truth_{m}'] = index.target_file.map(lambda t: truth[t][m])
    return root, index
