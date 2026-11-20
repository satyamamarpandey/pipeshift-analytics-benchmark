"""Statistical helpers: Wilson intervals, cluster bootstrap, permutation tests, Holm."""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

SEED = 20261022


def wilson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return (float('nan'), float('nan'))
    z = norm.ppf(1 - (1 - conf) / 2)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    lo = 0.0 if k == 0 else max(0.0, centre - half)
    hi = 1.0 if k == n else min(1.0, centre + half)
    return (float(lo), float(hi))


def cluster_bootstrap_mean(values, n_boot: int = 10_000, seed: int = SEED, conf: float = 0.95):
    """Percentile CI for the mean of per-cluster values (clusters resampled with replacement)."""
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return (float('nan'), float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    boots = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    lo, hi = np.quantile(boots, [(1 - conf) / 2, 1 - (1 - conf) / 2])
    return (float(v.mean()), float(lo), float(hi))


def bootstrap_diff(a, b, n_boot: int = 10_000, seed: int = SEED, conf: float = 0.95):
    """Difference of means of two independent sets of cluster values, percentile CI."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    da = a[rng.integers(0, len(a), size=(n_boot, len(a)))].mean(axis=1)
    db = b[rng.integers(0, len(b), size=(n_boot, len(b)))].mean(axis=1)
    d = da - db
    lo, hi = np.quantile(d, [(1 - conf) / 2, 1 - (1 - conf) / 2])
    return (float(a.mean() - b.mean()), float(lo), float(hi))


def permutation_test(a, b, n_perm: int = 10_000, seed: int = SEED) -> float:
    """Two-sided permutation p-value for difference in means (labels shuffled)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    obs = abs(a.mean() - b.mean())
    pooled = np.concatenate([a, b])
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        if abs(pooled[:len(a)].mean() - pooled[len(a):].mean()) >= obs - 1e-12:
            count += 1
    return (count + 1) / (n_perm + 1)


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj.tolist()
