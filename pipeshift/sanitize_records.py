"""Remove machine-specific absolute paths from captured messages in archived result files."""
from pathlib import Path

import pandas as pd

from llm_evaluate import sanitize

ROOT = Path(__file__).resolve().parents[1]
FILES = list((ROOT / 'results_v2' / 'llm').glob('episode_results_*.csv')) + \
    [ROOT / 'results_v2' / 'baselines' / 'episode_results.csv']

for f in FILES:
    df = pd.read_csv(f, keep_default_na=False, dtype=str)
    changed = 0
    for col in ('reason', 'metrics_json'):
        if col in df:
            new = df[col].map(sanitize)
            changed += int((new != df[col]).sum())
            df[col] = new
    if changed:
        df.to_csv(f, index=False)
    print(f'{f.name}: {changed} cells sanitized')
