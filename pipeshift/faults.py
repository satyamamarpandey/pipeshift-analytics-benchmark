"""Deterministic fault injection and episode materialization.

An *episode* = (clean snapshot, fault, intensity). For each episode we write
  work/episodes/<opaque_id>/trips.parquet (+ manifest.json unless absent)  -> evaluated inputs
  work/oracle/episodes.csv                                               -> ground truth index
  work/oracle/targets/<snapshot>[_v2].parquet                            -> intended data
  work/registry/<opaque_id>.json                                         -> independent trust anchor
Input directories are named by an opaque hash so evaluated workflows cannot infer the
fault from paths. Truth metrics are computed by oracle.py, not by any evaluated workflow.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from snapshots import month_bounds, write_parquet
from spec import ALL_FAULT_BY_ID, DELIVERED_COLUMNS, FAULTS, HELDOUT_FAULTS, INTENSITY_SWEEP, REPLICATES

ROOT = Path(__file__).resolve().parents[1]
STUDY_MONTHS = tuple(f'2025-{m:02d}' for m in range(1, 13))
SALT = 'pipeshift-v2-episode'


def prev_month(month: str) -> str:
    return (pd.Timestamp(month + '-01') - pd.offsets.MonthBegin(1)).strftime('%Y-%m')


def sha256_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def opaque(key: str) -> str:
    return hashlib.sha256(f'{SALT}|{key}'.encode()).hexdigest()[:16]


def count_for(intensity, n: int) -> int:
    return 1 if intensity == 'one' else max(1, int(round(float(intensity) * n)))


def write_observed(df: pd.DataFrame, path: Path) -> None:
    if list(df.columns) == list(DELIVERED_COLUMNS) and all(
            str(df[c].dtype) == {'timestamp[us]': 'datetime64[us]', 'double': 'float64'}.get(t, t) for c, t in DELIVERED_COLUMNS.items()):
        write_parquet(df, path)  # canonical writer: byte-identical to the published target
        return
    table = pa.Table.from_pandas(df, preserve_index=False)
    # Normalize naive nanosecond timestamps to microseconds (TLC native precision).
    fields = []
    for f in table.schema:
        if pa.types.is_timestamp(f.type) and f.type.unit == 'ns':
            fields.append(pa.field(f.name, pa.timestamp('us', tz=f.type.tz)))
        else:
            fields.append(f)
    table = table.cast(pa.schema(fields))
    table = table.replace_schema_metadata(None)
    pq.write_table(table, path, compression='zstd')


def manifest_for(path: Path, month: str, snapshot_id: str, version: int,
                 schema_changes=None, description: str = '') -> dict:
    start, end = month_bounds(month)
    schema = pq.read_schema(path)
    return {
        'dataset': 'NYC TLC Yellow Taxi trip records (benchmark snapshot)',
        'snapshot_id': snapshot_id,
        'snapshot_version': version,
        'reporting_period': {'start': start.isoformat(), 'end': end.isoformat(),
                             'timezone': 'America/New_York'},
        'file': 'trips.parquet',
        'file_sha256': sha256_bytes(path),
        'row_count': pq.ParquetFile(path).metadata.num_rows,
        'columns': [{'name': f.name, 'type': str(f.type)} for f in schema],
        'units': {'trip_distance': 'miles', 'fare_amount': 'USD'},
        'schema_changes': schema_changes or [],
        'description': description,
    }


def rows_of(rng, n, k):
    return np.sort(rng.choice(n, size=k, replace=False))


def to_text_numeric(df, cols=('fare_amount', 'trip_distance')):
    for c in cols:
        if c in df:
            df[c] = df[c].map(lambda x: repr(float(x)) if pd.notna(x) else None)
    return df


RENAMES = [{'type': 'rename', 'from': 'trip_distance', 'to': 'trip_miles'},
           {'type': 'rename', 'from': 'fare_amount', 'to': 'fare'}]


def inject(fid: str, intensity, base: pd.DataFrame, ctx: dict, rng: np.random.Generator):
    """Return (observed, target, manifest_mode, manifest_kwargs).

    target is the intended data whose metrics are the ground truth.
    """
    n = len(base)
    obs = base.copy()
    target = base
    kw: dict = {}
    mode = ALL_FAULT_BY_ID[fid].manifest
    k = count_for(intensity, n) if intensity is not None else None
    if fid == 'F00':
        pass
    elif fid == 'S01':
        obs = obs[list(reversed(obs.columns))]
    elif fid == 'S02':
        obs['cbd_congestion_fee'] = rng.choice([0.0, 0.75], size=n)
        obs['ingest_note'] = 'batch-7'
    elif fid == 'S03':
        obs = obs.rename(columns={'trip_distance': 'trip_miles', 'fare_amount': 'fare'})
        kw['schema_changes'] = RENAMES
    elif fid == 'S04':
        obs = to_text_numeric(obs)
    elif fid == 'S05':
        for c in ('payment_type', 'PULocationID', 'DOLocationID'):
            obs[c] = obs[c].astype('float64')
    elif fid == 'S06':
        obs = obs.drop(columns=['fare_amount'])
    elif fid == 'T01':
        for c in ('tpep_pickup_datetime', 'tpep_dropoff_datetime'):
            obs[c] = obs[c].dt.strftime('%Y-%m-%dT%H:%M:%S.%f').str[:-3]
    elif fid == 'T02':
        alt = rng.random(n) < 0.5
        for c in ('tpep_pickup_datetime', 'tpep_dropoff_datetime'):
            iso = obs[c].dt.strftime('%Y-%m-%d %H:%M:%S')
            us = obs[c].dt.strftime('%m/%d/%Y %I:%M:%S %p')
            obs[c] = np.where(alt, us, iso)
    elif fid == 'T03':
        for c in ('tpep_pickup_datetime', 'tpep_dropoff_datetime'):
            obs[c] = obs[c].dt.tz_localize('America/New_York', ambiguous=np.ones(n, dtype=bool),
                                           nonexistent='shift_forward').dt.tz_convert('UTC')
    elif fid == 'T04':
        obs = ctx['prev_snapshot'].copy()
    elif fid == 'T05':
        idx = rows_of(rng, n, k)
        repl = ctx['prev_snapshot'].iloc[rng.choice(len(ctx['prev_snapshot']), size=k, replace=False)].copy()
        repl['record_id'] = obs['record_id'].to_numpy()[idx]
        for c in obs.columns:
            col = obs[c].to_numpy().copy()
            col[idx] = repl[c].to_numpy()
            obs[c] = col
    elif fid == 'T06':
        kw['period_month'] = prev_month(ctx['month'])
    elif fid == 'R01':
        obs = pd.concat([obs, obs.iloc[rows_of(rng, n, k)]], ignore_index=True)
    elif fid == 'R02':
        dup = obs.iloc[rows_of(rng, n, k)].copy()
        dup['fare_amount'] = np.round(dup['fare_amount'] + rng.uniform(1, 10, size=k), 2)
        obs = pd.concat([obs, dup], ignore_index=True)
    elif fid == 'R03':
        obs.loc[rows_of(rng, n, k), 'fare_amount'] = np.nan
    elif fid == 'R04':
        obs = obs.drop(index=rows_of(rng, n, k)).reset_index(drop=True)
    elif fid == 'R05':
        card = np.flatnonzero(obs['payment_type'].to_numpy() == 1)
        obs.loc[np.sort(rng.choice(card, size=min(k, len(card)), replace=False)), 'payment_type'] = 9
    elif fid == 'D01':
        target = ctx['longtrip']
        obs = target.copy()
        kw['description'] = 'Producer-declared extract: trips with trip_distance >= 5 miles.'
    elif fid == 'N01':
        idx = rows_of(rng, n, k)
        obs.loc[idx, 'fare_amount'] = np.round(obs.loc[idx, 'fare_amount'] + rng.uniform(0.01, 1.0, size=k), 2)
    elif fid == 'N02':
        obs['fare_amount'] = np.round(obs['fare_amount'])
    elif fid == 'N03':
        obs['fare_amount'] = np.round(obs['fare_amount'] * 100, 6)
    elif fid == 'N04':
        obs['trip_distance'] = obs['trip_distance'] * 1.609344
    elif fid == 'N05':
        idx = rows_of(rng, n, k)
        obs.loc[idx, 'tpep_pickup_datetime'] = obs.loc[idx, 'tpep_pickup_datetime'] + pd.Timedelta(milliseconds=500)
    elif fid in ('P01', 'P02'):
        pass
    elif fid == 'P03':
        target = pd.concat([base, ctx['late']], ignore_index=True)
        obs = target.copy()
    elif fid == 'P04':
        idx = rows_of(rng, n, k)
        obs.loc[idx, 'fare_amount'] = np.round(obs.loc[idx, 'fare_amount'] * 1.5, 2)
    elif fid == 'C01':
        obs = pd.concat([obs, obs.iloc[rows_of(rng, n, count_for(0.05, n))]], ignore_index=True)
        obs = obs.rename(columns={'trip_distance': 'trip_miles', 'fare_amount': 'fare'})
        obs = to_text_numeric(obs, ('fare', 'trip_miles'))
        obs = obs[list(reversed(obs.columns))]
        kw['schema_changes'] = RENAMES
    elif fid == 'C02':
        obs.loc[rows_of(rng, n, count_for(0.02, n)), 'fare_amount'] = np.nan
        obs = obs.rename(columns={'trip_distance': 'trip_miles', 'fare_amount': 'fare'})
        kw['schema_changes'] = RENAMES
    elif fid == 'C03':
        prev = ctx['prev_snapshot']
        obs = pd.concat([prev, prev.iloc[rows_of(rng, len(prev), count_for(0.05, len(prev)))]], ignore_index=True)
    elif fid == 'C04':
        obs['fare_amount'] = np.round(obs['fare_amount'] * 100, 6)
        obs = to_text_numeric(obs, ('fare_amount',))
    elif fid == 'H01':
        rep = obs.iloc[rows_of(rng, n, k)].copy()
        rep['record_id'] = obs['record_id'].max() + 1_000 + np.arange(k, dtype=np.int64)
        obs = pd.concat([obs, rep], ignore_index=True)
    elif fid == 'H02':
        pt = obs['payment_type'].to_numpy()
        obs['payment_type'] = np.where(pt == 1, 2, np.where(pt == 2, 1, pt)).astype('int64')
    elif fid == 'H03':
        obs['fare_amount'] = np.round(obs['fare_amount'] * 1.10, 2)
    else:
        raise KeyError(fid)
    return obs, target, mode, kw


def materialize(out_root: Path, snap_dir: Path, months=STUDY_MONTHS, replicates=REPLICATES,
                include_sweep: bool = True, include_heldout: bool = True) -> pd.DataFrame:
    ep_dir = out_root / 'episodes'
    tgt_dir = out_root / 'oracle' / 'targets'
    reg_dir = out_root / 'registry'
    for d in (ep_dir, tgt_dir, reg_dir):
        d.mkdir(parents=True, exist_ok=True)
    rows = []
    for month in months:
        for rep in replicates:
            snap_id = f'{month}_{rep}'
            base = pd.read_parquet(snap_dir / f'{snap_id}_snapshot.parquet')
            ctx = {'month': month,
                   'prev_snapshot': pd.read_parquet(snap_dir / f'{prev_month(month)}_{rep}_snapshot.parquet'),
                   'late': pd.read_parquet(snap_dir / f'{snap_id}_late.parquet'),
                   'longtrip': pd.read_parquet(snap_dir / f'{snap_id}_longtrip.parquet')}
            # Intended (clean) data files and their published manifests.
            targets = {'base': base, 'v2': pd.concat([base, ctx['late']], ignore_index=True),
                       'longtrip': ctx['longtrip']}
            target_paths = {}
            for name, frame in targets.items():
                p = tgt_dir / f'{snap_id}_{name}.parquet'
                write_parquet(frame, p)
                target_paths[name] = p
            plan = [(f.fid, f.default_intensity, 'main') for f in FAULTS]
            if include_sweep:
                plan += [(fid, lvl, 'sweep') for fid, lvls in INTENSITY_SWEEP.items() for lvl in lvls]
            if include_heldout:
                plan += [(f.fid, f.default_intensity, 'heldout') for f in HELDOUT_FAULTS]
            for fid, intensity, suite in plan:
                key = f'{snap_id}|{fid}|{intensity}|{suite}'
                eid = opaque(key)
                seed_material = hashlib.sha256(key.encode()).digest()
                rng = np.random.default_rng(int.from_bytes(seed_material[:8], 'little'))
                obs, target, mode, kw = inject(fid, intensity, base, ctx, rng)
                d = ep_dir / eid
                d.mkdir(exist_ok=True)
                data_path = d / 'trips.parquet'
                write_observed(obs, data_path)
                period_month = kw.pop('period_month', month)
                if target is base:
                    tname = 'base'
                elif target is ctx['longtrip']:
                    tname = 'longtrip'
                else:
                    tname = 'v2'
                original_manifest = manifest_for(target_paths[tname], month, snap_id,
                                                 2 if tname == 'v2' else 1, kw.get('schema_changes'),
                                                 kw.get('description', ''))
                if mode in ('consistent', 'forged', 'wrong_period', 'corrupted', 'absent'):
                    man = manifest_for(data_path, period_month, snap_id, 2 if tname == 'v2' else 1, **kw)
                    if mode == 'corrupted':
                        man['file_sha256'] = hashlib.sha256(b'corrupted|' + man['file_sha256'].encode()).hexdigest()
                elif mode == 'original':
                    man = original_manifest
                elif mode == 'previous_version':
                    man = manifest_for(target_paths['base'], month, snap_id, 1)
                else:
                    raise ValueError(mode)
                man_path = d / 'manifest.json'
                if mode == 'absent':
                    man_path.unlink(missing_ok=True)
                else:
                    man_path.write_text(json.dumps(man, indent=2) + '\n')
                # Independent trust anchor: what the legitimate producer actually published
                # (the delivered manifest when the producer regenerated it; otherwise the
                # original publication record). Never derived from an attacker's manifest.
                registry = man if mode == 'consistent' else original_manifest
                (reg_dir / f'{eid}.json').write_text(json.dumps(registry, indent=2) + '\n')
                f = ALL_FAULT_BY_ID[fid]
                rows.append({'episode_id': eid, 'suite': suite, 'month': month, 'replicate': rep,
                             'snapshot': snap_id, 'fid': fid, 'fault': f.name, 'family': f.family,
                             'intensity': '' if intensity is None else str(intensity),
                             'desired': f.desired, 'manifest_mode': mode,
                             'target_file': target_paths[tname].relative_to(out_root).as_posix(),
                             'observed_rows': len(obs),
                             'observed_sha256': sha256_bytes(data_path)})
        print('materialized', month)
    index = pd.DataFrame(rows)
    index.to_csv(out_root / 'oracle' / 'episodes_index.csv', index=False)
    return index


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', type=Path, default=ROOT / 'work')
    ap.add_argument('--no-sweep', action='store_true')
    args = ap.parse_args()
    idx = materialize(args.work, args.work / 'snapshots', include_sweep=not args.no_sweep)
    print(idx.groupby('suite').size().to_string())


if __name__ == '__main__':
    main()
