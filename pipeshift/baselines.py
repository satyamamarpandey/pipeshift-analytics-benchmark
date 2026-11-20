"""Deterministic reference workflows built from independently switchable components.

Each workflow implements the same interface as generated workflows:
    analyze(data_path, manifest_path) -> {'status', 'metrics', 'reason'}
and raises/abstains when a validation component fails.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from spec import ADMISSIBLE, PAYMENT_CODES

REQUIRED = ('record_id', 'tpep_pickup_datetime', 'trip_distance', 'fare_amount', 'payment_type')
COMPONENTS = ('normalize_names', 'coerce_types', 'parse_time', 'dedup_exact', 'conflict_check',
              'null_check', 'domain_check', 'range_check', 'period_check', 'stats_check',
              'manifest_required', 'hash_check', 'rowcount_check')

_B1 = {'normalize_names', 'coerce_types', 'parse_time', 'dedup_exact'}
_CONTRACT = {'conflict_check', 'null_check', 'domain_check', 'range_check', 'period_check'}
_PROV = {'manifest_required', 'hash_check', 'rowcount_check'}
BASELINES = {
    'B0_naive': set(),
    'B1_schema_aware': _B1,
    'B2_statistical': _B1 | {'stats_check'},
    'B3_contract': _B1 | _CONTRACT,
    'B4_manifest': _B1 | _PROV,
    'B5_contract_manifest': _B1 | _CONTRACT | _PROV,
    'B6_all_checks': _B1 | _CONTRACT | _PROV | {'stats_check'},
}
STATS_LIMITS = {'rows_rel': 0.10, 'mean_fare_rel': 0.25, 'mean_distance_rel': 0.25,
                'card_share_abs': 0.10, 'null_frac': 0.001, 'unseen_code_frac': 0.001,
                'modal_month_min': 0.99}


class Abstain(Exception):
    pass


def reference_profile(df: pd.DataFrame) -> dict:
    return {'rows': len(df), 'mean_fare': float(df.fare_amount.mean()),
            'mean_distance': float(df.trip_distance.mean()),
            'card_share': float((df.payment_type == 1).mean())}


def compute_metrics(df: pd.DataFrame) -> dict:
    fare = df['fare_amount']
    return {'trip_count': int(len(df)), 'total_fare': float(fare.sum()), 'mean_fare': float(fare.mean()),
            'mean_trip_distance': float(df['trip_distance'].mean()),
            'card_share': float((df['payment_type'] == 1).mean())}


def _load_manifest(manifest_path) -> dict | None:
    p = Path(manifest_path)
    if not p.is_file():
        return None
    return json.loads(p.read_text())


def make_workflow(components: set[str], reference: dict | None = None, registry: dict | None = None):
    unknown = set(components) - set(COMPONENTS)
    if unknown:
        raise ValueError(unknown)

    def analyze(data_path, manifest_path):
        manifest = _load_manifest(manifest_path)
        anchor = registry if registry is not None else manifest
        if 'manifest_required' in components and anchor is None:
            raise Abstain('manifest_missing')
        if 'hash_check' in components and anchor is not None:
            h = hashlib.sha256(Path(data_path).read_bytes()).hexdigest()
            if h != anchor.get('file_sha256'):
                raise Abstain('sha256_mismatch')
        df = pd.read_parquet(data_path)
        if 'rowcount_check' in components and anchor is not None and len(df) != anchor.get('row_count'):
            raise Abstain('row_count_mismatch')
        if 'normalize_names' in components:
            if manifest:
                back = {c['to']: c['from'] for c in manifest.get('schema_changes', []) if c.get('type') == 'rename'}
                df = df.rename(columns={k: v for k, v in back.items() if k in df.columns and v not in df.columns})
            lower = {c.lower(): c for c in df.columns}
            df = df.rename(columns={lower[r.lower()]: r for r in REQUIRED if r not in df.columns and r.lower() in lower})
        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            raise Abstain('missing_columns:' + ','.join(missing))
        if 'coerce_types' in components:
            for c in ('record_id', 'trip_distance', 'fare_amount', 'payment_type'):
                if not pd.api.types.is_numeric_dtype(df[c]):
                    df[c] = pd.to_numeric(df[c], errors='coerce')
        if 'parse_time' in components:
            ts = df['tpep_pickup_datetime']
            if isinstance(ts.dtype, pd.DatetimeTZDtype):
                ts = ts.dt.tz_convert('America/New_York').dt.tz_localize(None)
            elif not pd.api.types.is_datetime64_any_dtype(ts):
                ts = pd.to_datetime(ts, format='mixed', errors='coerce')
            df['tpep_pickup_datetime'] = ts
        n_raw = len(df)
        if 'dedup_exact' in components:
            df = df.drop_duplicates()
            if 'conflict_check' in components and df['record_id'].duplicated().any():
                raise Abstain('conflicting_duplicate_record_id')
            df = df.drop_duplicates('record_id', keep='first')
        req = df[list(REQUIRED)]
        if 'null_check' in components and req.isna().any().any():
            raise Abstain('null_required_values')
        if 'domain_check' in components:
            pt = df['payment_type']
            if not (pt.isin(list(PAYMENT_CODES)) & (pt == np.floor(pt))).all():
                raise Abstain('undocumented_payment_type')
        if 'range_check' in components:
            if not df['fare_amount'].between(ADMISSIBLE['fare_amount_min'], ADMISSIBLE['fare_amount_max']).all():
                raise Abstain('fare_out_of_range')
            d = df['trip_distance']
            if not ((d > ADMISSIBLE['trip_distance_min_exclusive']) & (d <= ADMISSIBLE['trip_distance_max'])).all():
                raise Abstain('distance_out_of_range')
        if 'period_check' in components and anchor is not None:
            start = pd.Timestamp(anchor['reporting_period']['start'])
            end = pd.Timestamp(anchor['reporting_period']['end'])
            ts = df['tpep_pickup_datetime']
            if not pd.api.types.is_datetime64_any_dtype(ts) or not ts.between(start, end, inclusive='left').all():
                raise Abstain('pickup_outside_reporting_period')
        if 'stats_check' in components and reference is not None:
            lim = STATS_LIMITS
            fare = pd.to_numeric(df['fare_amount'], errors='coerce')
            dist = pd.to_numeric(df['trip_distance'], errors='coerce')
            pt = pd.to_numeric(df['payment_type'], errors='coerce')
            ts = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
            checks = {
                'rows': abs(n_raw / reference['rows'] - 1) <= lim['rows_rel'],
                'mean_fare': abs(fare.mean() / reference['mean_fare'] - 1) <= lim['mean_fare_rel'],
                'mean_distance': abs(dist.mean() / reference['mean_distance'] - 1) <= lim['mean_distance_rel'],
                'card_share': abs((pt == 1).mean() - reference['card_share']) <= lim['card_share_abs'],
                'nulls': req.isna().mean().max() <= lim['null_frac'],
                'unseen_codes': (~pt.isin(list(PAYMENT_CODES))).mean() <= lim['unseen_code_frac'],
                'modal_month': ts.dt.to_period('M').value_counts(normalize=True).max() >= lim['modal_month_min'],
            }
            failed = [k for k, v in checks.items() if not bool(v)]
            if failed:
                raise Abstain('statistical_anomaly:' + ','.join(failed))
        return {'status': 'ok', 'metrics': compute_metrics(df), 'reason': ''}

    return analyze


def naive_workflow(data_path, manifest_path):
    df = pd.read_parquet(data_path)
    return {'status': 'ok', 'metrics': compute_metrics(df), 'reason': ''}
