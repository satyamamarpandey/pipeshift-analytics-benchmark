"""Independent re-computation of headline numbers using only the standard library.

Reads the episode-level CSVs directly (no pandas, no code shared with analysis.py), recomputes
silent-error / false-rejection rates for baselines and LLM model x condition cells and the
pooled condition rates, and compares them with the LaTeX macros the manuscript prints.
Writes results_v2/verification/number_check.csv and exits non-zero on any mismatch.
"""
from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT / 'results_v2'


def read(path):
    with open(path, newline='', encoding='utf-8') as fh:
        yield from csv.DictReader(fh)


def macros() -> dict:
    text = (ROOT / 'paper' / 'generated_macros.tex').read_text()
    return dict(re.findall(r'\\newcommand\{\\(\w+)\}\{((?:[^{}]|\{[^{}]*\})*)\}', text))


def fmt(x):
    return f'{100 * x:.1f}'


def main() -> int:
    mac = macros()
    checks = []

    def check(name, recomputed):
        printed = mac.get(name)
        checks.append({'macro': name, 'printed': printed, 'recomputed': recomputed,
                       'match': printed == recomputed})

    # Baselines, main suite
    cnt = defaultdict(lambda: defaultdict(int))
    for r in read(R2 / 'baselines' / 'episode_results.csv'):
        if r['suite'] != 'main':
            continue
        c = cnt[r['workflow']]
        c['n'] += 1
        c[r['outcome']] += 1
        if r['desired'] == 'recover':
            c['n_rec'] += 1
    tag = {'B0_naive': 'Bzero', 'B1_schema_aware': 'Bone', 'B2_statistical': 'Btwo', 'B3_contract': 'Bthree',
           'B4_manifest': 'Bfour', 'B5_contract_manifest': 'Bfive', 'B6_all_checks': 'Bsix',
           'B7_contract_registry': 'Bseven'}
    for wf, k in tag.items():
        c = cnt[wf]
        check(f'Sil{k}', str(c['silent_error']))
        check(f'SilPct{k}', fmt(c['silent_error'] / c['n']))
        check(f'FRPct{k}', fmt(c['false_rejection'] / c['n_rec']))
        check(f'Des{k}', fmt((c['correct_recovery'] + c['correct_abstention']) / c['n']))
    check('NMainEpisodes', str(cnt['B0_naive']['n']))

    # LLM: per-generation rates, then mean over generations
    llm = R2 / 'llm' / 'episode_results_test.csv'
    if llm.exists():
        g = defaultdict(lambda: defaultdict(int))
        meta = {}
        for r in read(llm):
            s = g[r['gen_id']]
            s['n'] += 1
            s['sil'] += r['outcome'] == 'silent_error'
            if r['desired'] == 'recover':
                s['rec'] += 1
                s['fr'] += r['outcome'] == 'false_rejection'
            meta[r['gen_id']] = (r['model'], r['condition'])
        cells = defaultdict(list)
        pooled = defaultdict(list)
        for gid, s in g.items():
            assert s['n'] == 384, gid
            cells[meta[gid]].append((s['sil'] / s['n'], s['fr'] / s['rec']))
            pooled[meta[gid][1]].append(s['sil'] / s['n'])
        mt = {'claude-haiku-5-5': 'Haiku', 'claude-sonnet-5-5': 'Sonnet', 'claude-opus-5-5': 'Opus',
              'qwen2.5-coder-7b-instruct-q4_k_m': 'Qwen'}
        ct = {'A_unassisted': 'A', 'B_schema_aware': 'B', 'C_provenance_guided': 'C'}
        for (m, c), v in cells.items():
            k = mt[m] + ct[c]
            check(f'Sil{k}', fmt(sum(x[0] for x in v) / len(v)))
            check(f'FR{k}', fmt(sum(x[1] for x in v) / len(v)))
        for c, v in pooled.items():
            check(f'PooledSil{ct[c]}', fmt(sum(v) / len(v)))
        check('NGenerations', str(len(g)))
        n_exec = sum(s['n'] for s in g.values())
        check('NLLMExecutions', f'{n_exec:,}'.replace(',', '{,}'))
        cost = sum(float(r['cost_usd'] or 0) for f in (R2 / 'llm' / 'logs').glob('generations_test_*.csv')
                   for r in read(f))
        check('LLMCost', f'{cost:.2f}')
    held = R2 / 'llm' / 'episode_results_heldout.csv'
    if held.exists():
        hg = defaultdict(lambda: [0, 0])
        hmeta = {}
        for r in read(held):
            if not r['model'].startswith('claude'):
                continue
            hg[r['gen_id']][0] += 1
            hg[r['gen_id']][1] += r['outcome'] == 'silent_error'
            hmeta[r['gen_id']] = r['condition']
        pc = defaultdict(list)
        for gid, (n, s) in hg.items():
            pc[hmeta[gid]].append(s / n)
        for c, v in pc.items():
            check(f"HeldSil{ {'A_unassisted': 'A', 'B_schema_aware': 'B', 'C_provenance_guided': 'C'}[c]}",
                  fmt(sum(v) / len(v)))
    out = R2 / 'verification'
    out.mkdir(exist_ok=True)
    with open(out / 'number_check.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=['macro', 'printed', 'recomputed', 'match'])
        w.writeheader()
        w.writerows(checks)
    bad = [c for c in checks if not c['match']]
    print(f'{len(checks)} numbers checked independently; {len(bad)} mismatches')
    for b in bad:
        print('MISMATCH', b)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
