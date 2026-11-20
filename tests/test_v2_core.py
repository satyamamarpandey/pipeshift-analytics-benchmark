"""PipeShift v2: specification, scoring, oracle, sampling and statistics."""
import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from conftest import tlc_like
from oracle import truth_python, truth_sql
from scoring import classify, metrics_match
from snapshots import allocate, sample_month, write_parquet
from spec import FAULTS, FAULT_BY_ID, INTENSITY_SWEEP, METRICS, OUTCOMES, TOLERANCE
from stats import bootstrap_diff, cluster_bootstrap_mean, holm, permutation_test, wilson

TRUTH = {'trip_count': 100, 'total_fare': 1500.0, 'mean_fare': 15.0, 'mean_trip_distance': 3.2, 'card_share': 0.6}


# ---------------------------------------------------------------- specification
def test_fault_catalog_is_well_formed():
    assert len(FAULTS) == 32 and len(FAULT_BY_ID) == 32
    assert {f.desired for f in FAULTS} == {'recover', 'abstain'}
    assert sum(f.desired == 'recover' for f in FAULTS) == 12
    assert set(INTENSITY_SWEEP) <= set(FAULT_BY_ID)
    assert set(TOLERANCE) == set(METRICS)
    # every recover-desired fault must leave the metrics computable
    assert all(not f.metric_impact for f in FAULTS if f.desired == 'recover')


def test_fault_families_cover_requested_change_types():
    fams = {f.family for f in FAULTS}
    assert {'schema', 'record', 'temporal', 'numeric', 'provenance', 'compound', 'control'} <= fams


# ---------------------------------------------------------------- scoring
@pytest.mark.parametrize('desired,result,expected', [
    ('recover', {'status': 'ok', 'metrics': TRUTH}, 'correct_recovery'),
    ('abstain', {'status': 'ok', 'metrics': TRUTH}, 'unverified_accept'),
    ('recover', {'status': 'abstain', 'metrics': None}, 'false_rejection'),
    ('abstain', {'status': 'abstain', 'metrics': None}, 'correct_abstention'),
    ('abstain', {'_exception': 'ValueError: x'}, 'correct_abstention'),
    ('recover', {'_exception': 'KeyError: fare'}, 'false_rejection'),
    ('recover', {'_timeout': True}, 'invalid_output'),
    ('recover', 'NO_CODE_EXTRACTED', 'invalid_output'),
    ('recover', {'status': 'OK', 'metrics': TRUTH}, 'invalid_output'),
    ('recover', {'status': 'ok', 'metrics': {**TRUTH, 'total_fare': 1500.02}}, 'silent_error'),
    ('abstain', {'status': 'ok', 'metrics': {**TRUTH, 'trip_count': 101}}, 'silent_error'),
    ('recover', {'status': 'ok', 'metrics': {**TRUTH, 'card_share': float('nan')}}, 'silent_error'),
    ('recover', {'status': 'ok', 'metrics': {k: v for k, v in TRUTH.items() if k != 'mean_fare'}}, 'silent_error'),
    ('recover', {'status': 'ok', 'metrics': None}, 'silent_error'),
])
def test_classify(desired, result, expected):
    assert classify(result, desired, TRUTH)['outcome'] == expected


def test_tolerance_boundaries():
    assert metrics_match({**TRUTH, 'total_fare': 1500.005}, TRUTH)[0]
    assert not metrics_match({**TRUTH, 'total_fare': 1500.0051}, TRUTH)[0]
    assert metrics_match({**TRUTH, 'card_share': 0.60004}, TRUTH)[0]
    assert not metrics_match({**TRUTH, 'card_share': 0.6002}, TRUTH)[0]
    assert not metrics_match({**TRUTH, 'trip_count': True}, TRUTH)[0]  # bools are not numbers
    assert metrics_match({**TRUTH, 'trip_count': '100'}, TRUTH)[0]     # numeric strings accepted


def test_one_record_change_exceeds_tolerance():
    # The smallest injected value change (+0.01 USD on one record) must be detectable.
    assert 0.01 > TOLERANCE['total_fare']
    assert 1 / 5000 > TOLERANCE['card_share']


# ---------------------------------------------------------------- oracle
def test_oracle_sql_matches_python(tmp_path):
    df = tlc_like('2025-03', 500, 3)
    p = tmp_path / 't.parquet'
    write_parquet(df, p)
    import duckdb
    a, b = truth_sql(duckdb.connect(), p), truth_python(p)
    assert a['trip_count'] == b['trip_count'] == 500
    for m in METRICS[1:]:
        assert math.isclose(a[m], b[m], rel_tol=1e-12, abs_tol=1e-9)
    assert math.isclose(b['card_share'], (df.payment_type == 1).mean())


def test_oracle_rejects_duplicate_targets(tmp_path):
    df = tlc_like('2025-03', 50, 3)
    p = tmp_path / 't.parquet'
    write_parquet(pd.concat([df, df.iloc[:1]]), p)
    import duckdb
    with pytest.raises(AssertionError):
        truth_sql(duckdb.connect(), p)


@settings(max_examples=40, deadline=None)
@given(st.lists(st.floats(min_value=0, max_value=1000, allow_nan=False).map(lambda x: round(x, 2)),
                min_size=1, max_size=60))
def test_oracle_property_random_fares(tmp_path_factory, fares):
    n = len(fares)
    df = tlc_like('2025-04', n, 1)
    df['fare_amount'] = fares
    p = tmp_path_factory.mktemp('h') / 't.parquet'
    write_parquet(df, p)
    import duckdb
    a, b = truth_sql(duckdb.connect(), p), truth_python(p)
    assert math.isclose(a['total_fare'], math.fsum(fares), abs_tol=1e-6)
    assert math.isclose(a['total_fare'], b['total_fare'], abs_tol=1e-6)


# ---------------------------------------------------------------- sampling
@settings(max_examples=60, deadline=None)
@given(st.lists(st.integers(min_value=1, max_value=10_000), min_size=1, max_size=31),
       st.integers(min_value=1, max_value=5000))
def test_allocation_sums_and_is_proportional(counts, total):
    c = np.array(counts)
    a = allocate(c, total)
    assert a.sum() == total
    assert np.all(np.abs(a - c / c.sum() * total) < 1 + 1e-9)


def test_sample_month_disjoint_and_deterministic(monkeypatch):
    import snapshots
    monkeypatch.setattr(snapshots, 'SNAPSHOT_ROWS', 300)
    monkeypatch.setattr(snapshots, 'LATE_ROWS', 20)
    df = tlc_like('2025-05', 5000, 9).drop(columns='record_id')
    a = sample_month(df, '2025-05', np.random.default_rng(1))
    b = sample_month(df, '2025-05', np.random.default_rng(1))
    for k in a:
        pd.testing.assert_frame_equal(a[k], b[k])
    keys = [set(map(tuple, a[k].drop(columns='record_id').astype(str).values)) for k in a]
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            assert not keys[i] & keys[j], 'pools must be disjoint'
    ids = pd.concat([a[k].record_id for k in a])
    assert ids.is_unique
    assert len(a['A_snapshot']) == 300 and len(a['A_late']) == 20
    assert (a['A_longtrip'].trip_distance >= 5).all()


# ---------------------------------------------------------------- statistics
def test_wilson_known_values():
    lo, hi = wilson(0, 24)
    assert lo == 0 and math.isclose(hi, 0.1380, abs_tol=5e-4)
    lo, hi = wilson(12, 24)
    assert math.isclose(lo, 0.3143, abs_tol=5e-4) and math.isclose(hi, 0.6857, abs_tol=5e-4)
    assert wilson(24, 24)[1] == 1.0


def test_bootstrap_and_permutation():
    m, lo, hi = cluster_bootstrap_mean([0.1] * 10)
    assert m == lo == hi == pytest.approx(0.1)
    d, lo, hi = bootstrap_diff([0.3] * 10, [0.1] * 10)
    assert d == pytest.approx(0.2) and lo == pytest.approx(0.2)
    assert permutation_test([0.0] * 10, [1.0] * 10, n_perm=2000) < 0.001
    assert permutation_test([0.1, 0.2, 0.3], [0.2, 0.1, 0.3], n_perm=2000) > 0.5
    assert cluster_bootstrap_mean([0.0, 1.0] * 5) == cluster_bootstrap_mean([0.0, 1.0] * 5)


def test_holm():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5]) == [0.5]


def test_outcome_vocabulary():
    assert set(OUTCOMES) == {'correct_recovery', 'correct_abstention', 'false_rejection', 'silent_error',
                             'unverified_accept', 'invalid_output'}
