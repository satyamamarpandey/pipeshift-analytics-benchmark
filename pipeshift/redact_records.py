"""Apply pipeshift.redact to the ``reason`` column of the archived episode-level records (release hygiene).

Usage: python redact_records.py [--check]
Only the ``reason`` cells that quote source-data values change; every other cell is left byte-identical.
With --check nothing is written and the exit status is 1 if any record still contains a quoted value."""
from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

from redact import redact_values

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted(list((ROOT / 'results_v2').rglob('episode_results*.csv')) + list((ROOT / 'docs' / 'records').rglob('episode_results*.csv')))


def main() -> int:
    check = '--check' in sys.argv
    total = 0
    for f in FILES:
        raw = f.read_bytes().decode('utf-8')
        nl = '\r\n' if '\r\n' in raw else '\n'
        rows = list(csv.reader(io.StringIO(raw, newline='')))
        if not rows or 'reason' not in rows[0]:
            continue
        i = rows[0].index('reason')
        changed = 0
        for r in rows[1:]:
            new = redact_values(r[i])
            if new != r[i]:
                changed += 1
                r[i] = new
        if changed and not check:
            buf = io.StringIO(newline='')
            csv.writer(buf, lineterminator=nl).writerows(rows)
            f.write_bytes(buf.getvalue().encode('utf-8'))
        print(f'{f.relative_to(ROOT).as_posix()}: {changed} reason cells {"would change" if check else "redacted"}')
        total += changed
    return 1 if (check and total) else 0


if __name__ == '__main__':
    sys.exit(main())
