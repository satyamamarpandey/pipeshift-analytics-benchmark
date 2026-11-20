"""PipeShift v2: fault injection, materialization, manifests, trust anchors and baselines."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from baselines import BASELINES, make_workflow, reference_profile
from conftest import tlc_like
from faults import count_for, inject, materialize, opaque
from scoring import classify
from spec import FAULT_BY_ID, FAULTS, INTENSITY_SWEEP, METRICS


def _ep(world, fid, suite='main', intensity=None):
    root, idx = world
    sel = idx[(idx.fid == fid) & (idx.suite == suite)]
    if intensity is not None:
        sel = sel[sel.intensity == str(intensity)]
    row = sel.iloc[0]
    d = root / 'episodes' / row.episode_id
    return row, d


def _man(d):
    p = d / 'manifest.json'
    return json.loads(p.read_text()) if p.exists() else None


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_episode_counts(mini_world):
    _, idx = mini_world
    assert (idx.suite == 'main').sum() == len(FAULTS)
    assert (idx.suite == 'sweep').sum() == sum(len(v) for v in INTENSITY_SWEEP.values())
    assert idx.episode_id.is_unique


def test_episode_paths_and_manifests_are_opaque(mini_world):
    root, idx = mini_world
    names = {f.name for f in FAULTS} | set(FAULT_BY_ID)
    for row in idx.itertuples():
        assert row.episode_id not in names and len(row.episode_id) == 16
        mp = root / 'episodes' / row.episode_id / 'manifest.json'
        if mp.exists():
            text = mp.read_text()
            assert not any(f'"{n}"' in text for n in names if len(n) > 3)
        assert sorted(p.name for p in (root / 'episodes' / row.episode_id).iterdir()) in (
            ['manifest.json', 'trips.parquet'], ['trips.parquet'])


def test_materialization_is_deterministic(mini_world, tmp_path):
    root, idx = mini_world
    again = materialize(tmp_path, root / 'snapshots', months=('2025-02',), replicates=('A',), include_sweep=True)
    pd.testing.assert_frame_equal(idx[again.columns].reset_index(drop=True), again)


def test_inject_determinism_and_counts():
    base = tlc_like('2025-02', 400, 4)
    ctx = {'month': '2025-02', 'prev_snapshot': tlc_like('2025-01', 400, 5),
           'late': tlc_like('2025-02', 10, 6), 'longtrip': tlc_like('2025-02', 400, 7, min_distance=5)}
    for f in FAULTS:
        a = inject(f.fid, f.default_intensity, base, ctx, np.random.default_rng(7))[0]
        b = inject(f.fid, f.default_intensity, base, ctx, np.random.default_rng(7))[0]
        pd.testing.assert_frame_equal(a, b)
    assert count_for('one', 5000) == 1 and count_for(0.02, 5000) == 100 and count_for(0.0001, 100) == 1
    obs = inject('R03', 0.02, base, ctx, np.random.default_rng(1))[0]
    assert obs.fare_amount.isna().sum() == 8
    obs = inject('R01', 0.05, base, ctx, np.random.default_rng(1))[0]
    assert len(obs) == 420 and obs.duplicated().sum() == 20
    obs = inject('R02', 0.01, base, ctx, np.random.default_rng(1))[0]
    assert obs.record_id.duplicated().sum() == 4 and obs.duplicated().sum() == 0
    obs = inject('R05', 0.03, base, ctx, np.random.default_rng(1))[0]
    assert (obs.payment_type == 9).sum() == 12
    obs = inject('N05', 0.02, base, ctx, np.random.default_rng(1))[0]
    delta = obs.tpep_pickup_datetime - base.tpep_pickup_datetime
    assert (delta == pd.Timedelta(milliseconds=500)).sum() == 8 and (delta == pd.Timedelta(0)).sum() == 392
    obs = inject('T05', 0.10, base, ctx, np.random.default_rng(1))[0]
    assert obs.record_id.equals(base.record_id)
    assert (obs.tpep_pickup_datetime.dt.month == 1).sum() == 40
    obs = inject('T03', None, base, ctx, np.random.default_rng(1))[0]
    back = obs.tpep_pickup_datetime.dt.tz_convert('America/New_York').dt.tz_localize(None)
    assert (back.dt.floor('h') == base.tpep_pickup_datetime.dt.floor('h')).mean() > 0.99


def test_manifest_modes(mini_world):
    _, d = _ep(mini_world, 'F00')
    assert _man(d)['file_sha256'] == _sha(d / 'trips.parquet')
    for fid in ('N01', 'N02', 'N05', 'R04', 'T04'):          # data changed after publication
        _, d = _ep(mini_world, fid)
        assert _man(d)['file_sha256'] != _sha(d / 'trips.parquet'), fid
    for fid in ('S01', 'S03', 'S06', 'R03', 'N03', 'P04'):    # producer/attacker regenerated manifest
        _, d = _ep(mini_world, fid)
        m = _man(d)
        assert m['file_sha256'] == _sha(d / 'trips.parquet') and m['row_count'] == len(pd.read_parquet(d / 'trips.parquet'))
    _, d = _ep(mini_world, 'P01')
    assert _man(d) is None
    _, d = _ep(mini_world, 'P02')
    assert _man(d)['file_sha256'] != _sha(d / 'trips.parquet')
    _, d = _ep(mini_world, 'S03')
    assert {c['to'] for c in _man(d)['schema_changes']} == {'trip_miles', 'fare'}
    _, d = _ep(mini_world, 'T06')
    assert _man(d)['reporting_period']['start'].startswith('2025-01')


def test_clean_delivery_is_byte_identical_to_published_target(mini_world):
    root, _ = mini_world
    row, d = _ep(mini_world, 'F00')
    assert _sha(d / 'trips.parquet') == _sha(root / row.target_file)


def test_registry_is_independent_of_attacker(mini_world):
    root, _ = mini_world
    row, d = _ep(mini_world, 'P04')
    reg = json.loads((root / 'registry' / f'{row.episode_id}.json').read_text())
    assert reg['file_sha256'] != _man(d)['file_sha256']
    assert reg['file_sha256'] == _sha(root / row.target_file)


def _run(world, wf_name, fid, registry=False, suite='main', intensity=None):
    root, _ = world
    row, d = _ep(world, fid, suite, intensity)
    ref = reference_profile(pd.read_parquet(root / 'snapshots' / '2025-01_A_snapshot.parquet'))
    reg = json.loads((root / 'registry' / f'{row.episode_id}.json').read_text()) if registry else None
    wf = make_workflow(set(BASELINES[wf_name]), reference=ref, registry=reg)
    try:
        res = wf(str(d / 'trips.parquet'), str(d / 'manifest.json'))
    except Exception as exc:  # noqa: BLE001
        res = {'_exception': str(exc)}
    truth = {m: row[f'truth_{m}'] for m in METRICS}
    return classify(res, row.desired, truth)['outcome']


@pytest.mark.parametrize('fid', [f.fid for f in FAULTS if f.desired == 'recover' and f.fid != 'D01'])
def test_contract_manifest_baseline_recovers_benign_changes(mini_world, fid):
    assert _run(mini_world, 'B5_contract_manifest', fid) == 'correct_recovery'


@pytest.mark.parametrize('fid', ['S06', 'R02', 'R03', 'R04', 'R05', 'T04', 'T05', 'T06', 'N01', 'N02', 'N03',
                                 'N05', 'P01', 'P02', 'P03', 'C02', 'C03', 'C04'])
def test_contract_manifest_baseline_abstains(mini_world, fid):
    assert _run(mini_world, 'B5_contract_manifest', fid) == 'correct_abstention'


def test_trust_boundary_forged_manifest(mini_world):
    # A consistent forged manifest defeats manifest verification; an independent anchor does not.
    assert _run(mini_world, 'B5_contract_manifest', 'P04') == 'silent_error'
    # B7 = B5 components with the independent registry as trust anchor.
    assert _run(mini_world, 'B5_contract_manifest', 'P04', registry=True) == 'correct_abstention'


def test_naive_baseline_failure_modes(mini_world):
    assert _run(mini_world, 'B0_naive', 'F00') == 'correct_recovery'
    assert _run(mini_world, 'B0_naive', 'R01') == 'silent_error'
    assert _run(mini_world, 'B0_naive', 'N01') == 'silent_error'
    assert _run(mini_world, 'B0_naive', 'S03') == 'false_rejection'


def test_hash_is_the_only_check_for_post_publication_value_changes(mini_world):
    assert _run(mini_world, 'B3_contract', 'N01') == 'silent_error'
    assert _run(mini_world, 'B4_manifest', 'N01') == 'correct_abstention'
    assert _run(mini_world, 'B3_contract', 'N05') == 'unverified_accept'


def test_statistical_check_false_alarm_on_valid_shift(mini_world):
    assert _run(mini_world, 'B2_statistical', 'D01') == 'false_rejection'
    assert _run(mini_world, 'B3_contract', 'D01') == 'correct_recovery'


def test_single_record_faults_escape_statistics_but_not_contract(tmp_path):
    # At the benchmark's 5,000-row scale one undocumented code is 0.02% of rows, below the
    # statistical monitor's 0.1% tolerance, while the contract's domain rule is exact.
    from snapshots import write_parquet
    df = tlc_like('2025-02', 5000, 3)
    ref = reference_profile(df)
    df.loc[df.index[df.payment_type == 1][0], 'payment_type'] = 9
    p = tmp_path / 'trips.parquet'
    write_parquet(df, p)
    stats_wf = make_workflow(set(BASELINES['B2_statistical']), reference=ref)
    assert stats_wf(str(p), str(tmp_path / 'none.json'))['status'] == 'ok'
    with pytest.raises(Exception, match='undocumented_payment_type'):
        make_workflow(set(BASELINES['B3_contract']), reference=ref)(str(p), str(tmp_path / 'none.json'))


def test_opaque_ids_are_salted_hashes():
    assert opaque('x') != opaque('y') and len(opaque('x')) == 16
