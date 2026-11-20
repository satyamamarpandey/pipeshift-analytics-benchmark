"""Collect independent model generations of the analysis workflow.

Backends
  claude-cli : Anthropic models via the Claude Code CLI in headless mode (`claude -p`),
               custom system prompt, all tools disabled, no session persistence, run from
               an empty directory. Each call is a fresh, independent session.
  ollama     : open-weight models served locally by Ollama (temperature and seed set).

Every generation is kept (including failures) and logged to generations.csv with
model, provider, condition, timestamps, prompt hashes, token usage and reported cost.
Credentials are never written. Spending is capped by --budget-usd (reported list cost).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path

import pandas as pd
import requests

from prompts import CONDITIONS, SYSTEM_PROMPT, build_prompt, prompt_hashes

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results_v2' / 'llm'
CODE_RE = re.compile(r'```(?:python|py)?[ \t]*\n(.*?)```', re.S)
RISKY = ('subprocess', 'socket', 'requests', 'urllib', 'os.system', 'shutil.rmtree', 'eval(', 'exec(')


def extract_code(text: str) -> str | None:
    blocks = CODE_RE.findall(text or '')
    if not blocks:
        return None
    with_def = [b for b in blocks if 'def analyze' in b]
    return (with_def or blocks)[-1]


def call_claude(model: str, prompt: str, max_budget: float) -> dict:
    with tempfile.TemporaryDirectory(prefix='psgen_') as tmp:
        cmd = ['claude', '-p', '--model', model, '--system-prompt', SYSTEM_PROMPT, '--tools', '',
               '--output-format', 'json', '--no-session-persistence', '--setting-sources', '',
               '--strict-mcp-config', '--disable-slash-commands', '--max-budget-usd', str(max_budget)]
        t0 = time.time()
        proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding='utf-8',
                              cwd=tmp, timeout=900)
        wall = time.time() - t0
    try:
        js = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {'ok': False, 'error': (proc.stderr or proc.stdout)[:500], 'wall_seconds': wall, 'text': ''}
    usage = js.get('usage', {})
    mu = js.get('modelUsage', {})
    return {'ok': not js.get('is_error', False), 'text': js.get('result', ''), 'wall_seconds': wall,
            'error': '' if not js.get('is_error') else str(js.get('result'))[:500],
            'response_model': ','.join(mu.keys()), 'session_id': js.get('session_id', ''),
            'request_uuid': js.get('uuid', ''), 'stop_reason': js.get('stop_reason', ''),
            'input_tokens': usage.get('input_tokens', 0) + usage.get('cache_creation_input_tokens', 0)
                            + usage.get('cache_read_input_tokens', 0),
            'output_tokens': usage.get('output_tokens', 0),
            'thinking_tokens': usage.get('output_tokens_details', {}).get('thinking_tokens', 0),
            'cost_usd': float(js.get('total_cost_usd', 0.0)),
            'sampling': 'provider default (temperature not settable via CLI)'}


def call_ollama(model: str, prompt: str, seed: int, temperature: float, host: str) -> dict:
    t0 = time.time()
    r = requests.post(f'{host}/api/chat', timeout=1800, json={
        'model': model, 'stream': False,
        'messages': [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': prompt}],
        'options': {'temperature': temperature, 'seed': seed, 'num_ctx': 8192, 'num_predict': 4096}})
    r.raise_for_status()
    js = r.json()
    return {'ok': True, 'text': js['message']['content'], 'wall_seconds': time.time() - t0, 'error': '',
            'response_model': js.get('model', model), 'session_id': '', 'request_uuid': '',
            'stop_reason': js.get('done_reason', ''), 'input_tokens': js.get('prompt_eval_count', 0),
            'output_tokens': js.get('eval_count', 0), 'thinking_tokens': 0, 'cost_usd': 0.0,
            'sampling': f'temperature={temperature}, seed={seed}, num_predict=4096'}


def call_llamacpp(model: str, prompt: str, seed: int, temperature: float, host: str) -> dict:
    t0 = time.time()
    r = requests.post(f'{host}/v1/chat/completions', timeout=3600, json={
        'model': model, 'temperature': temperature, 'seed': seed, 'max_tokens': 4096,
        'messages': [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': prompt}]})
    r.raise_for_status()
    js = r.json()
    usage = js.get('usage', {})
    return {'ok': True, 'text': js['choices'][0]['message']['content'], 'wall_seconds': time.time() - t0,
            'error': '', 'response_model': js.get('model', model), 'session_id': '', 'request_uuid': js.get('id', ''),
            'stop_reason': js['choices'][0].get('finish_reason', ''), 'input_tokens': usage.get('prompt_tokens', 0),
            'output_tokens': usage.get('completion_tokens', 0), 'thinking_tokens': 0, 'cost_usd': 0.0,
            'sampling': f'temperature={temperature}, seed={seed}, max_tokens=4096, llama.cpp defaults otherwise'}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', choices=['claude-cli', 'ollama', 'llamacpp'], required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--conditions', nargs='+', default=list(CONDITIONS))
    ap.add_argument('--n', type=int, default=10, help='generations per condition')
    ap.add_argument('--split', default='test', choices=['test', 'pilot'])
    ap.add_argument('--budget-usd', type=float, default=20.0, help='cap on cumulative reported cost (all models)')
    ap.add_argument('--temperature', type=float, default=0.8)
    ap.add_argument('--ollama-host', default='http://127.0.0.1:11434')
    ap.add_argument('--llamacpp-host', default='http://127.0.0.1:8080')
    args = ap.parse_args()
    gen_dir = OUT / 'generations'
    gen_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r'[^A-Za-z0-9._-]', '_', args.model)
    log_path = OUT / 'logs' / f'generations_{args.split}_{safe}.csv'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame()

    def total_spent() -> float:  # budget is global across all per-model logs
        return float(sum(pd.read_csv(f).cost_usd.sum() for f in log_path.parent.glob('generations_*.csv')))
    hashes = prompt_hashes()
    safe_model = re.sub(r'[^A-Za-z0-9._-]', '_', args.model)
    # Interleave conditions so that time-of-day effects are spread across conditions.
    for i in range(args.n):
        for cond in args.conditions:
            gen_id = f'{args.split}__{safe_model}__{cond}__{i:02d}'
            if not log.empty and gen_id in set(log.gen_id):
                continue
            spent = total_spent() if log_path.parent.exists() else 0.0
            if spent >= args.budget_usd:
                print(f'BUDGET CAP reached: ${spent:.2f} >= ${args.budget_usd:.2f}; stopping.')
                return
            prompt = build_prompt(cond)
            started = dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')
            try:
                if args.backend == 'claude-cli':
                    res = call_claude(args.model, prompt, max_budget=min(2.0, args.budget_usd - spent))
                elif args.backend == 'llamacpp':
                    res = call_llamacpp(args.model, prompt, seed=1000 + i, temperature=args.temperature,
                                        host=args.llamacpp_host)
                else:
                    res = call_ollama(args.model, prompt, seed=1000 + i, temperature=args.temperature,
                                      host=args.ollama_host)
            except Exception as exc:  # noqa: BLE001 - logged as a failed generation
                res = {'ok': False, 'error': f'{type(exc).__name__}: {exc}'[:500], 'text': '', 'wall_seconds': 0}
            code = extract_code(res.get('text', ''))
            (gen_dir / f'{gen_id}.response.txt').write_text(res.get('text', ''), encoding='utf-8')
            code_sha = ''
            if code is not None:
                (gen_dir / f'{gen_id}.py').write_text(code, encoding='utf-8')
                code_sha = hashlib.sha256(code.encode()).hexdigest()
            row = {'gen_id': gen_id, 'split': args.split, 'backend': args.backend,
                   'provider': {'claude-cli': 'Anthropic (via Claude Code CLI)', 'ollama': 'local Ollama',
                                'llamacpp': 'local llama.cpp server'}[args.backend],
                   'model': args.model, 'condition': cond, 'index': i, 'started_utc': started,
                   'prompt_sha256': hashes[cond], 'system_sha256': hashes['system'],
                   'code_extracted': code is not None, 'code_sha256': code_sha,
                   'code_lines': code.count('\n') + 1 if code else 0,
                   'risky_tokens': '|'.join(t for t in RISKY if code and t in code),
                   **{k: res.get(k, '') for k in ('ok', 'error', 'response_model', 'session_id', 'request_uuid',
                                                   'stop_reason', 'input_tokens', 'output_tokens',
                                                   'thinking_tokens', 'cost_usd', 'wall_seconds', 'sampling')}}
            row['cost_usd'] = float(row['cost_usd'] or 0.0)
            log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
            log.to_csv(log_path, index=False)
            print(f"{gen_id}: ok={row['ok']} code={row['code_extracted']} out_tok={row['output_tokens']} "
                  f"${row['cost_usd']:.4f} total=${total_spent():.2f}", flush=True)


if __name__ == '__main__':
    main()
