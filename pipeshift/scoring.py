"""Episode scoring against the oracle. Shared by baselines and generated workflows.

A workflow returns {'status': 'ok'|'abstain', 'metrics': {...}, 'reason': str} or
raises. Raising and explicit abstention are both non-silent outcomes; they are
recorded separately ('signal') but scored identically.
"""
from __future__ import annotations

import math

from spec import ABSTAIN, METRICS, RECOVER, TOLERANCE


def _num(x):
    if isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def metrics_match(metrics: dict, truth: dict) -> tuple[bool, list[str]]:
    wrong = []
    for m in METRICS:
        v = _num(metrics.get(m)) if isinstance(metrics, dict) else None
        if v is None or abs(v - float(truth[m])) > TOLERANCE[m] + 1e-12:
            wrong.append(m)
    return not wrong, wrong


def classify(result, desired: str, truth: dict) -> dict:
    """result: dict returned by workflow, or {'_exception': str}, or {'_timeout': True}."""
    if not isinstance(result, dict):
        return {'outcome': 'invalid_output', 'signal': 'non_dict', 'wrong_metrics': ''}
    if result.get('_timeout'):
        return {'outcome': 'invalid_output', 'signal': 'timeout', 'wrong_metrics': ''}
    if '_exception' in result:
        signal, accepted = 'exception', False
    else:
        status = result.get('status')
        if status == 'abstain':
            signal, accepted = 'abstain', False
        elif status == 'ok':
            signal, accepted = 'ok', True
        else:
            return {'outcome': 'invalid_output', 'signal': f'bad_status:{str(status)[:20]}', 'wrong_metrics': ''}
    if not accepted:
        outcome = 'correct_abstention' if desired == ABSTAIN else 'false_rejection'
        return {'outcome': outcome, 'signal': signal, 'wrong_metrics': ''}
    ok, wrong = metrics_match(result.get('metrics'), truth)
    if desired == RECOVER:
        outcome = 'correct_recovery' if ok else 'silent_error'
    else:
        outcome = 'unverified_accept' if ok else 'silent_error'
    return {'outcome': outcome, 'signal': signal, 'wrong_metrics': '|'.join(wrong)}
