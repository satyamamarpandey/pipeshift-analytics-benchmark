"""Held-out scenarios H01-H03: injection effects, prompt independence and result integrity."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import tlc_like
from faults import inject
from prompts import CONDITIONS, build_prompt
from scoring import classify
from spec import ALL_FAULT_BY_ID, HELDOUT_FAULTS, METRICS

ROOT = Path(__file__).resolve().parents[1]
CTX = {'month': '2025-02', 'prev_snapshot': tlc_like('2025-01', 400, 5),
       'late': tlc_like('2025-02', 10, 6), 'longtrip': tlc_like('2025-02', 400, 7, min_distance=5)}


def test_heldout_injection_effects_and_determinism():
    base = tlc_like('2025-02', 400, 4)
    h1 = inject('H01', 0.05, base, CTX, np.random.default_rng(3))[0]
    assert len(h1) == 420 and h1.record_id.is_unique
    assert h1.drop(columns='record_id').duplicated().sum() == 20      # content replays, fresh keys
    h2 = inject('H02', None, base, CTX, np.random.default_rng(3))[0]
    assert ((base.payment_type == 1) == (h2.payment_type == 2)).all()
    assert ((base.payment_type == 2) == (h2.payment_type == 1)).all()
    h3 = inject('H03', None, base, CTX, np.random.default_rng(3))[0]
    assert np.allclose(h3.fare_amount, np.round(base.fare_amount * 1.10, 2))
    for f in HELDOUT_FAULTS:
        a = inject(f.fid, f.default_intensity, base, CTX, np.random.default_rng(9))[0]
        b = inject(f.fid, f.default_intensity, base, CTX, np.random.default_rng(9))[0]
        pd.testing.assert_frame_equal(a, b)


def test_heldout_scenarios_are_abstain_and_metric_changing():
    for f in HELDOUT_FAULTS:
        assert f.desired == 'abstain' and f.metric_impact and f.manifest == 'consistent'
        assert ALL_FAULT_BY_ID[f.fid] is f


def test_prompts_do_not_describe_heldout_checks():
    for c in CONDITIONS:
        p = build_prompt(c).lower()
        for phrase in ('re-key', 'rekey', 'fresh record_id', 'swap', '10%', '1.10', 'inflat'):
            assert phrase not in p, (c, phrase)


@pytest.mark.skipif(not (ROOT / 'results_v2' / 'llm' / 'episode_results_heldout.csv').exists()
                    or not (ROOT / 'work' / 'oracle' / 'episodes.csv').exists(),
                    reason='held-out results or rebuilt oracle not present')
def test_heldout_results_complete_and_rescorable():
    ho = pd.read_csv(ROOT / 'results_v2' / 'llm' / 'episode_results_heldout.csv', keep_default_na=False)
    assert ho.gen_id.nunique() == 120 and ho.groupby('gen_id').size().eq(36).all()
    assert set(ho.fid) == {'H01', 'H02', 'H03'}
    eps = pd.read_csv(ROOT / 'work' / 'oracle' / 'episodes.csv', keep_default_na=False)
    m = ho[ho.metrics_json != ''].merge(eps[['episode_id'] + [f'truth_{k}' for k in METRICS]], on='episode_id')
    for r in m.sample(n=min(1000, len(m)), random_state=1).itertuples():
        truth = {k: getattr(r, f'truth_{k}') for k in METRICS}
        assert classify({'status': 'ok', 'metrics': json.loads(r.metrics_json)}, r.desired, truth)['outcome'] == r.outcome
