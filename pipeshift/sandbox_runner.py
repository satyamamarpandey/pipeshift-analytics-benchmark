"""Execute one generated module on a batch of episodes inside a restricted child process.

Invoked as:  python -I sandbox_runner.py <module.py> <jobs.json> <out.jsonl>
Restrictions (best effort, enforced with a Python audit hook after library imports):
  * no network sockets, subprocesses, os.system/exec/spawn, or dynamic library loading;
  * file opens limited to read-only access of the episode input directories and the
    Python installation; no writes anywhere except the runner's own output file.
The parent process applies wall-clock timeouts and a clean environment.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback

import numpy as np  # noqa: F401  (pre-import before restrictions)
import pandas as pd  # noqa: F401
import pyarrow  # noqa: F401
import pyarrow.parquet  # noqa: F401
import hashlib  # noqa: F401
import datetime  # noqa: F401
import zoneinfo  # noqa: F401


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (bool, type(None), str)):
        return x
    if isinstance(x, (int, np.integer)):
        return int(x)
    try:
        f = float(x)
        return f if math.isfinite(f) else str(f)
    except (TypeError, ValueError):
        return str(x)


def main() -> None:
    module_path, jobs_path, out_path = sys.argv[1:4]
    jobs = json.loads(open(jobs_path, encoding='utf-8').read())
    allowed_roots = [os.path.normcase(os.path.abspath(p)) for p in
                     {sys.prefix, sys.base_prefix, sys.exec_prefix, os.path.dirname(os.path.abspath(module_path))}
                     | {os.path.dirname(j['data_path']) for j in jobs}]
    out = open(out_path, 'a', encoding='utf-8')
    src = open(module_path, encoding='utf-8').read()

    blocked_events = ('socket.', 'subprocess.Popen', 'os.system', 'os.exec', 'os.spawn', 'os.posix_spawn',
                      'os.startfile', 'ctypes.dlopen', 'winreg.', 'os.remove', 'os.rename', 'os.rmdir',
                      'shutil.rmtree', 'os.mkdir', 'os.chmod', 'os.unlink')

    def hook(event, args):
        if event.startswith(blocked_events):
            raise PermissionError(f'sandbox: {event} blocked')
        if event == 'open':
            path, mode = args[0], args[1]
            if isinstance(path, int):
                return
            p = os.path.normcase(os.path.abspath(os.fsdecode(path)))
            writing = isinstance(mode, str) and any(c in mode for c in 'wax+')
            if writing or not any(p.startswith(r) for r in allowed_roots):
                raise PermissionError(f'sandbox: open {p!r} mode={mode!r} blocked')

    sys.addaudithook(hook)
    namespace = {'__name__': 'generated_workflow'}
    try:
        exec(compile(src, 'generated_workflow.py', 'exec'), namespace)
        analyze = namespace['analyze']
        load_error = None
    except BaseException as exc:  # noqa: BLE001
        load_error = f'load: {type(exc).__name__}: {exc}'[:300]
    for job in jobs:
        t0 = time.perf_counter()
        if load_error:
            rec = {'_exception': load_error}
        else:
            try:
                rec = analyze(job['data_path'], job['manifest_path'])
                rec = _jsonable(rec)
            except BaseException as exc:  # noqa: BLE001
                tb = traceback.extract_tb(exc.__traceback__)
                rec = {'_exception': f'{type(exc).__name__}: {exc}'[:300],
                       '_where': f'{tb[-1].name}:{tb[-1].lineno}' if tb else ''}
        out.write(json.dumps({'episode_id': job['episode_id'], 'seconds': time.perf_counter() - t0,
                              'result': rec}) + '\n')
        out.flush()
    out.close()


if __name__ == '__main__':
    main()
