"""PipeShift v2: generation parsing, sandbox restrictions, prompts and portability."""
import os
import re
from pathlib import Path

import pytest

from llm_evaluate import run_batch
from llm_generate import extract_code
from prompts import CONDITIONS, build_prompt, prompt_hashes
from spec import FAULTS

ROOT = Path(__file__).resolve().parents[1]

GOOD = '''
import pandas as pd
def analyze(data_path, manifest_path):
    df = pd.read_parquet(data_path)
    return {"status": "ok", "metrics": {"trip_count": len(df), "total_fare": df.fare_amount.sum(),
            "mean_fare": df.fare_amount.mean(), "mean_trip_distance": df.trip_distance.mean(),
            "card_share": (df.payment_type == 1).mean()}, "reason": ""}
'''
ATTACKS = {
    'network': 'import urllib.request\ndef analyze(d, m):\n    urllib.request.urlopen("http://example.com", timeout=2)\n',
    'write': 'def analyze(d, m):\n    open("pwned.txt", "w").write("x")\n',
    'subprocess': 'import subprocess\ndef analyze(d, m):\n    subprocess.run(["cmd", "/c", "echo"])\n',
    'read_oracle': 'def analyze(d, m):\n    return open(r"{oracle}").read()\n',
}


def test_extract_code_prefers_block_with_analyze():
    text = 'intro\n```python\nx = 1\n```\nmore\n```python\ndef analyze(a, b):\n    return {}\n```\n'
    assert extract_code(text).startswith('def analyze')
    assert extract_code('no code here') is None
    assert extract_code('```\ndef analyze(a,b): pass\n```').startswith('def analyze')


def test_prompts_frozen_and_free_of_answer_key():
    h = prompt_hashes()
    assert h['A_unassisted'].startswith('834b3886') and h['B_schema_aware'].startswith('0b939c49')
    assert h['C_provenance_guided'].startswith('df680767') and h['system'].startswith('632284d3')
    for c in CONDITIONS:
        p = build_prompt(c)
        for f in FAULTS:
            assert f.name not in p
        assert 'truth' not in p.lower() and 'oracle' not in p.lower()
    assert 'sha-256' not in build_prompt('B_schema_aware').lower()
    assert 'SHA-256' in build_prompt('C_provenance_guided')


def _jobs(world, n=2):
    root, idx = world
    ids = idx[idx.suite == 'main'].episode_id.head(n)
    return [{'episode_id': e, 'data_path': str(root / 'episodes' / e / 'trips.parquet'),
             'manifest_path': str(root / 'episodes' / e / 'manifest.json')} for e in ids]


def test_sandbox_runs_valid_module(mini_world, tmp_path):
    mod = tmp_path / 'good.py'
    mod.write_text(GOOD)
    res = run_batch(mod, _jobs(mini_world), job_timeout=60)
    assert all(r['result']['status'] == 'ok' for r in res.values())


@pytest.mark.parametrize('attack', list(ATTACKS))
def test_sandbox_blocks(mini_world, tmp_path, attack):
    root, _ = mini_world
    code = ATTACKS[attack].replace('{oracle}', str(root / 'registry'))
    mod = tmp_path / f'{attack}.py'
    mod.write_text(code)
    res = run_batch(mod, _jobs(mini_world, 1), job_timeout=60)
    out = next(iter(res.values()))['result']
    assert '_exception' in out and ('sandbox' in out['_exception'] or 'Permission' in out['_exception']), out
    assert not (tmp_path / 'pwned.txt').exists()


def test_sandbox_verbose_stderr_does_not_deadlock(mini_world, tmp_path):
    # Regression: an undrained stderr pipe once blocked a workflow that warned on every episode.
    mod = tmp_path / 'noisy.py'
    mod.write_text('import sys\n' + GOOD.replace('    df = pd.read_parquet', '    sys.stderr.write("w" * 40000)\n'
                                                  '    df = pd.read_parquet'))
    res = run_batch(mod, _jobs(mini_world, 3), job_timeout=10)
    assert len(res) == 3 and all(r['result'].get('status') == 'ok' for r in res.values())


def test_sandbox_timeout_and_crash(mini_world, tmp_path):
    hang = tmp_path / 'hang.py'
    hang.write_text('import time\ndef analyze(d, m):\n    time.sleep(600)\n')
    crash = tmp_path / 'crash.py'
    crash.write_text('import os\ndef analyze(d, m):\n    os._exit(3)\n')
    for mod in (hang, crash):
        res = run_batch(mod, _jobs(mini_world, 2), job_timeout=3)
        assert len(res) == 2 and all(r['result'].get('_timeout') for r in res.values())
    # Defensive guard: a timed-out runner (including any child interpreter) must not keep running.
    if os.name == 'nt':
        import subprocess
        q = subprocess.run(['powershell', '-NoProfile', '-Command',
                            "(Get-CimInstance Win32_Process -Filter \"name='python.exe'\").CommandLine"],
                           capture_output=True, text=True).stdout
        assert str(hang) not in q, 'orphaned sandbox interpreter still running'


def test_no_hardcoded_local_paths_in_sources():
    pat = re.compile(r'[A-Za-z]:\\\\Users|[A-Za-z]:/Users|D:\\\\Research|D:/Research|/home/[a-z]')
    for p in list((ROOT / 'pipeshift').glob('*.py')) + list((ROOT / 'benchmark').glob('*.py')):
        assert not pat.search(p.read_text(encoding='utf-8')), p
