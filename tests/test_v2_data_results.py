"""PipeShift v2: source fidelity and result integrity (skipped when data/results are absent)."""
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from scoring import classify
from spec import METRICS, OUTCOMES, SAMPLING_SEED, SOURCE_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'data' / 'source_manifest.json'
RAW = ROOT / 'data' / 'raw'
WORK = ROOT / 'work'
R2 = ROOT / 'results_v2'

needs_raw = pytest.mark.skipif(not (SRC.exists() and any(RAW.glob('*.parquet'))), reason='raw TLC files not downloaded')
needs_work = pytest.mark.skipif(not (WORK / 'oracle' / 'episodes.csv').exists(), reason='episodes not materialized')
needs_base = pytest.mark.skipif(not (R2 / 'baselines' / 'episode_results.csv').exists(), reason='baselines not run')
needs_llm = pytest.mark.skipif(not (R2 / 'llm' / 'episode_results_test.csv').exists(), reason='LLM eval not run')


@needs_raw
def test_source_files_match_recorded_hashes():
    man = json.loads(SRC.read_text())
    assert len(man['files']) == 13
    for f in man['files']:
        p = ROOT / f['local_file']
        h = hashlib.sha256()
        with p.open('rb') as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b''):
                h.update(chunk)
        assert h.hexdigest() == f['sha256'], f['month']
        assert f['url'].startswith('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_')


@needs_raw
@needs_work
def test_snapshot_records_are_genuine_source_rows():
    cols = ', '.join(f's."{c}"' for c in SOURCE_COLUMNS)
    on = ' AND '.join(f's."{c}" IS NOT DISTINCT FROM r."{c}"' for c in SOURCE_COLUMNS)
    con = duckdb.connect()
    for month in ('2025-03', '2025-11'):
        snap = (WORK / 'snapshots' / f'{month}_A_snapshot.parquet').as_posix()
        raw = (RAW / f'yellow_tripdata_{month}.parquet').as_posix()
        n = con.execute(f"""SELECT count(*) FROM (SELECT DISTINCT {cols} FROM read_parquet('{snap}') s
                            WHERE EXISTS (SELECT 1 FROM read_parquet('{raw}') r WHERE {on}))""").fetchone()[0]
        assert n == 5000, month


@needs_raw
@needs_work
def test_snapshot_sampling_reproduces_stored_files():
    from snapshots import admissible_frame, sample_month
    month = '2025-02'
    frame = admissible_frame(duckdb.connect(), RAW / f'yellow_tripdata_{month}.parquet', month)
    parts = sample_month(frame, month, np.random.default_rng([SAMPLING_SEED, 202502]))
    for key in ('A_snapshot', 'B_late', 'A_longtrip'):
        stored = pd.read_parquet(WORK / 'snapshots' / f'{month}_{key}.parquet')
        pd.testing.assert_frame_equal(parts[key].reset_index(drop=True), stored, check_dtype=False)


@needs_work
def test_snapshots_have_no_future_leakage():
    eps = pd.read_csv(WORK / 'oracle' / 'episodes.csv', keep_default_na=False)
    for snap in eps.snapshot.unique()[:6]:
        month = snap[:7]
        df = pd.read_parquet(WORK / 'snapshots' / f'{snap}_snapshot.parquet')
        assert (df.tpep_pickup_datetime.dt.strftime('%Y-%m') == month).all()


@needs_base
@needs_work
def test_baseline_results_complete_and_rescorable():
    res = pd.read_csv(R2 / 'baselines' / 'episode_results.csv', keep_default_na=False, dtype={'intensity': str})
    eps = pd.read_csv(WORK / 'oracle' / 'episodes.csv', keep_default_na=False, dtype={'intensity': str})
    assert len(eps) == 1608  # 768 main + 768 sweep + 72 held-out
    assert not res.duplicated(['workflow', 'episode_id']).any()
    assert set(res.outcome) <= set(OUTCOMES)
    per = res.groupby('workflow').size()
    assert per[[w for w in per.index if w.startswith(('B0', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7'))]].eq(1608).all()
    assert per[[w for w in per.index if w.startswith('ABL_')]].eq(768).all()
    # Re-score every accepted result from stored metrics against the oracle.
    m = res[res.metrics_json != ''].merge(eps, on='episode_id', suffixes=('', '_e'))
    for r in m.sample(n=min(3000, len(m)), random_state=0).itertuples():
        truth = {k: getattr(r, f'truth_{k}') for k in METRICS}
        out = classify({'status': 'ok', 'metrics': json.loads(r.metrics_json)}, r.desired, truth)['outcome']
        assert out == r.outcome


@needs_llm
@needs_work
def test_llm_results_complete_and_rescorable():
    ep = pd.read_csv(R2 / 'llm' / 'episode_results_test.csv', keep_default_na=False)
    gens = pd.concat([pd.read_csv(f) for f in (R2 / 'llm' / 'logs').glob('generations_test_*.csv')])
    assert set(ep.gen_id) == set(gens.gen_id)
    assert ep.groupby('gen_id').size().eq(384).all()
    assert not ep.duplicated(['gen_id', 'episode_id']).any()
    assert set(ep.outcome) <= set(OUTCOMES)
    assert gens.cost_usd.sum() <= 25.0
    eps = pd.read_csv(WORK / 'oracle' / 'episodes.csv', keep_default_na=False)
    m = ep[ep.metrics_json != ''].merge(eps[['episode_id', 'desired'] + [f'truth_{k}' for k in METRICS]],
                                        on='episode_id', suffixes=('', '_e'))
    for r in m.sample(n=min(3000, len(m)), random_state=0).itertuples():
        truth = {k: getattr(r, f'truth_{k}') for k in METRICS}
        out = classify({'status': 'ok', 'metrics': json.loads(r.metrics_json)}, r.desired, truth)['outcome']
        assert out == r.outcome
