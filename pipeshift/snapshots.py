"""Natural data-quality profiling and deterministic stratified snapshot sampling.

For every acquired month this module
  1. profiles naturally occurring data-quality conditions in the full raw file
     (reported separately; never treated as injected faults), and
  2. draws disjoint, day-stratified samples of admissible records:
       replicate A and B clean snapshots (SNAPSHOT_ROWS each),
       a 'late' pool per replicate (records used for appended versions), and
       a long-trip pool per replicate (for the valid distribution-shift control).
Only records from the snapshot's own month are used, so no future information
enters an earlier period. Output: work/snapshots/*.parquet and
results_v2/data_profile/*.csv.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from spec import ADMISSIBLE, DELIVERED_COLUMNS, PAYMENT_CODES, REPLICATES, SAMPLING_SEED, \
    SNAPSHOT_ROWS, SOURCE_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
LATE_ROWS = 150          # 3% of SNAPSHOT_ROWS, for stale-manifest / appended versions
LONG_TRIP_MILES = 5.0


def arrow_schema(columns: dict[str, str] = DELIVERED_COLUMNS) -> pa.Schema:
    mapping = {'int64': pa.int64(), 'int32': pa.int32(), 'double': pa.float64(),
               'timestamp[us]': pa.timestamp('us')}
    return pa.schema([(k, mapping[v]) for k, v in columns.items()])


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    table = pa.Table.from_pandas(df[list(DELIVERED_COLUMNS)], schema=arrow_schema(), preserve_index=False)
    pq.write_table(table, path, compression='zstd')


def month_bounds(month: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(month + '-01')
    return start, start + pd.offsets.MonthBegin(1)


def profile(con: duckdb.DuckDBPyConnection, path: Path, month: str) -> dict:
    start, end = month_bounds(month)
    cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{path.as_posix()}')").fetchall()]
    src = f"read_parquet('{path.as_posix()}')"
    codes = ','.join(str(c) for c in PAYMENT_CODES)
    q = f"""
    SELECT count(*) AS raw_rows,
      count(*) FILTER (WHERE tpep_pickup_datetime < TIMESTAMP '{start}' OR tpep_pickup_datetime >= TIMESTAMP '{end}') AS pickup_outside_month,
      count(*) FILTER (WHERE tpep_dropoff_datetime < tpep_pickup_datetime) AS dropoff_before_pickup,
      count(*) FILTER (WHERE trip_distance <= 0) AS distance_nonpositive,
      count(*) FILTER (WHERE trip_distance > {ADMISSIBLE['trip_distance_max']}) AS distance_over_100mi,
      count(*) FILTER (WHERE fare_amount < 0) AS fare_negative,
      count(*) FILTER (WHERE fare_amount > {ADMISSIBLE['fare_amount_max']}) AS fare_over_1000,
      count(*) FILTER (WHERE payment_type NOT IN ({codes})) AS payment_code_undocumented,
      count(*) FILTER (WHERE payment_type = 0) AS payment_code_0_flex,
      count(*) FILTER (WHERE passenger_count IS NULL) AS passenger_count_null,
      count(*) FILTER (WHERE RatecodeID IS NULL) AS ratecode_null,
      count(*) FILTER (WHERE {' OR '.join(f'"{c}" IS NULL' for c in SOURCE_COLUMNS)}) AS required_field_null
    FROM {src}"""
    row = con.execute(q).df().iloc[0].to_dict()
    dup = con.execute(f"SELECT count(*) - (SELECT count(*) FROM (SELECT DISTINCT * FROM {src})) FROM {src}").fetchone()[0]
    out = {'month': month, **{k: int(v) for k, v in row.items()}, 'exact_duplicate_rows': int(dup),
           'columns': '|'.join(cols), 'n_columns': len(cols),
           'has_cbd_congestion_fee': 'cbd_congestion_fee' in cols,
           'airport_fee_column_spelling': next((c for c in cols if c.lower() == 'airport_fee'), '')}
    return out


def admissible_frame(con: duckdb.DuckDBPyConnection, path: Path, month: str) -> pd.DataFrame:
    start, end = month_bounds(month)
    codes = ','.join(str(c) for c in PAYMENT_CODES)
    sel = ', '.join(f'"{c}"' for c in SOURCE_COLUMNS)
    nonnull = ' AND '.join(f'"{c}" IS NOT NULL' for c in SOURCE_COLUMNS)
    q = f"""
    SELECT DISTINCT {sel} FROM read_parquet('{path.as_posix()}')
    WHERE {nonnull}
      AND tpep_pickup_datetime >= TIMESTAMP '{start}' AND tpep_pickup_datetime < TIMESTAMP '{end}'
      AND tpep_dropoff_datetime >= tpep_pickup_datetime
      AND trip_distance > {ADMISSIBLE['trip_distance_min_exclusive']} AND trip_distance <= {ADMISSIBLE['trip_distance_max']}
      AND fare_amount >= {ADMISSIBLE['fare_amount_min']} AND fare_amount <= {ADMISSIBLE['fare_amount_max']}
      AND payment_type IN ({codes})
    ORDER BY {sel}"""
    df = con.execute(q).df()
    df['tpep_pickup_datetime'] = df['tpep_pickup_datetime'].astype('datetime64[us]')
    df['tpep_dropoff_datetime'] = df['tpep_dropoff_datetime'].astype('datetime64[us]')
    for c, t in DELIVERED_COLUMNS.items():
        if c in df and t in ('int32', 'int64'):
            df[c] = df[c].astype(t)
    return df


def allocate(day_counts: np.ndarray, total: int) -> np.ndarray:
    """Proportional allocation by largest remainder (deterministic)."""
    quota = day_counts / day_counts.sum() * total
    base = np.floor(quota).astype(int)
    remainder = total - base.sum()
    order = np.lexsort((np.arange(len(quota)), -(quota - base)))
    base[order[:remainder]] += 1
    return base


def sample_month(df: pd.DataFrame, month: str, rng: np.random.Generator) -> dict[str, pd.DataFrame]:
    day = df['tpep_pickup_datetime'].dt.day.to_numpy()
    days = np.unique(day)
    groups = {d: rng.permutation(np.flatnonzero(day == d)) for d in days}
    counts = np.array([len(groups[d]) for d in days])
    alloc_main = allocate(counts, SNAPSHOT_ROWS)
    alloc_late = allocate(counts, LATE_ROWS)
    out: dict[str, pd.DataFrame] = {}
    used = []
    cursor = {d: 0 for d in days}
    yyyymm = int(month.replace('-', ''))
    for r, rep in enumerate(REPLICATES, start=1):
        for kind, alloc, offset in (('snapshot', alloc_main, 0), ('late', alloc_late, SNAPSHOT_ROWS)):
            idx = []
            for d, k in zip(days, alloc):
                idx.extend(groups[d][cursor[d]:cursor[d] + k])
                cursor[d] += k
            idx = np.sort(np.array(idx))
            part = df.iloc[idx].copy().reset_index(drop=True)
            part.insert(0, 'record_id', np.int64(yyyymm) * 10**7 + r * 10**6 + offset + np.arange(len(part), dtype=np.int64))
            out[f'{rep}_{kind}'] = part
            used.append(idx)
    # Long-trip pool from records not used above.
    taken = np.zeros(len(df), dtype=bool)
    taken[np.concatenate(used)] = True
    long_idx = np.flatnonzero(~taken & (df['trip_distance'].to_numpy() >= LONG_TRIP_MILES))
    long_perm = rng.permutation(long_idx)
    for r, rep in enumerate(REPLICATES, start=1):
        idx = np.sort(long_perm[(r - 1) * SNAPSHOT_ROWS: r * SNAPSHOT_ROWS])
        part = df.iloc[idx].copy().reset_index(drop=True)
        part.insert(0, 'record_id', np.int64(yyyymm) * 10**7 + (r + 4) * 10**6 + np.arange(len(part), dtype=np.int64))
        out[f'{rep}_longtrip'] = part
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-manifest', type=Path, default=ROOT / 'data' / 'source_manifest.json')
    ap.add_argument('--out', type=Path, default=ROOT / 'work' / 'snapshots')
    ap.add_argument('--profile-out', type=Path, default=ROOT / 'results_v2' / 'data_profile')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.profile_out.mkdir(parents=True, exist_ok=True)
    src = json.loads(args.source_manifest.read_text())
    con = duckdb.connect()
    con.execute('SET threads TO 8')
    profiles, sample_rows = [], []
    for f in src["files"]:
        month = f['month']
        path = ROOT / f['local_file']
        prof = profile(con, path, month)
        frame = admissible_frame(con, path, month)
        prof['admissible_distinct_rows'] = len(frame)
        profiles.append(prof)
        rng = np.random.default_rng([SAMPLING_SEED, int(month.replace('-', ''))])
        parts = sample_month(frame, month, rng)
        for key, part in parts.items():
            p = args.out / f'{month}_{key}.parquet'
            write_parquet(part, p)
            sample_rows.append({'month': month, 'part': key, 'rows': len(part),
                                'fare_sum': round(float(part.fare_amount.sum()), 2),
                                'distance_sum': round(float(part.trip_distance.sum()), 2),
                                'card_share': round(float((part.payment_type == 1).mean()), 6),
                                'days_covered': int(part.tpep_pickup_datetime.dt.day.nunique())})
        print(f"{month}: raw={prof['raw_rows']:,} admissible={len(frame):,} outside_month={prof['pickup_outside_month']:,}")
    pd.DataFrame(profiles).to_csv(args.profile_out / 'natural_quality_profile.csv', index=False)
    pd.DataFrame(sample_rows).to_csv(args.profile_out / 'snapshot_samples.csv', index=False)


if __name__ == '__main__':
    main()
