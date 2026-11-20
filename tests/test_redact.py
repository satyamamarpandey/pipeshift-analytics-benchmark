"""Release-hygiene tests: quoted source-data values are redacted; counts, column names and outcomes are untouched."""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pipeshift'))
from redact import redact_values  # noqa: E402


def test_datetime_value_redacted():
    msg = 'time data "02/01/2025 02:32:01 AM" doesn\'t match format "%Y-%m-%d %H:%M:%S", at position 1.'
    out = redact_values(msg)
    assert '02/01/2025' not in out and 'time data "<value>"' in out and '%Y-%m-%d' in out


def test_concatenated_numbers_redacted_even_when_truncated():
    out = redact_values("Could not convert string '9.318.416.310.023.316.313.543.3126.823.334.568.87.914.936.510.05.87.2")
    assert out == "Could not convert string '<value>'"
    assert redact_values("ValueError: could not convert string to float: '930.01840.0163'") == "ValueError: could not convert string to float: '<value>'"


def test_generic_messages_unchanged():
    for m in ("missing required columns: ['fare_amount']", 'row count mismatch: manifest says 5000, file has 4250',
              'SHA-256 mismatch', "'<' not supported between instances of 'str' and 'int'"):
        assert redact_values(m) == m


def test_released_records_contain_no_quoted_values():
    files = list((ROOT / 'results_v2').rglob('episode_results*.csv')) + list((ROOT / 'docs' / 'records').rglob('episode_results*.csv'))
    assert files
    for f in files:
        with open(f, newline='', encoding='utf-8') as fh:
            for row in csv.DictReader(fh):
                r = row.get('reason', '')
                assert redact_values(r) == r, (f.name, r[:80])
