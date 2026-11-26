"""Set-level diversity metrics of benchmarks in a feature space (#11).

All functions work on one pool ``R`` of problems at one dimension (never pool
dimensions): a feature matrix with one row per problem and a benchmark label per row.
Features are robustly standardized on the pool, distances are Euclidean.

- coverage ``C``: fraction of the pool within ``eps`` of some problem of the benchmark.
  Problems of the benchmark are themselves in the pool, so they count as covered.
- Vendi score ``VS``: effective number of distinct problems of the benchmark.
- novelty ``N``: fraction of the benchmark farther than ``eps`` from every reference
  (BBOB) problem.

``eps`` is the median distance to the 5th nearest neighbour in the pool and the Vendi
bandwidth is the median pairwise distance in the pool. ``C`` and ``VS`` grow with the
benchmark size, so :func:`diversity` reports them averaged over random subsets of the
reference size (``@24``).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.linalg import eigvalsh
from scipy.spatial.distance import cdist, pdist, squareform

__all__ = ["coverage", "diversity", "novelty", "radius", "standardize", "vendi_score"]

NEIGHBOUR = 5
SUBSET_SIZE = 24
SUBSETS = 200


def standardize(pool: np.ndarray) -> np.ndarray:
    """``(x - median) / IQR`` per feature, both computed on the pool.

    Raises ``ValueError`` for a feature with zero IQR: it cannot be scaled, so the
    caller has to drop it explicitly.
    """
    median = np.median(pool, axis=0)
    q1, q3 = np.percentile(pool, [25, 75], axis=0)
    iqr = q3 - q1
    if np.any(iqr == 0):
        raise ValueError(f"Features with zero IQR in the pool: columns {np.flatnonzero(iqr == 0)}.")
    return (pool - median) / iqr


def radius(pool: np.ndarray, k: int = NEIGHBOUR) -> float:
    """Median over the pool of the distance to the ``k``-th nearest other problem."""
    distances = squareform(pdist(pool))
    np.fill_diagonal(distances, np.inf)
    return float(np.median(np.sort(distances, axis=1)[:, k - 1]))


def coverage(pool: np.ndarray, benchmark: np.ndarray, eps: float) -> float:
    """Fraction of the pool within ``eps`` of some problem of ``benchmark``."""
    return float(np.mean(cdist(pool, benchmark).min(axis=1) <= eps))


def vendi_score(benchmark: np.ndarray, bandwidth: float) -> float:
    """``exp`` of the entropy of the eigenvalues of ``K / n`` (Gaussian kernel ``K``)."""
    kernel = np.exp(-squareform(pdist(benchmark, "sqeuclidean")) / (2 * bandwidth**2))
    eigenvalues = np.clip(eigvalsh(kernel / len(benchmark)), 0, None)
    eigenvalues = eigenvalues[eigenvalues > 0]
    return float(np.exp(-np.sum(eigenvalues * np.log(eigenvalues))))


def novelty(benchmark: np.ndarray, reference: np.ndarray, eps: float) -> float:
    """Fraction of ``benchmark`` farther than ``eps`` from every ``reference`` problem."""
    return float(np.mean(cdist(benchmark, reference).min(axis=1) > eps))


def diversity(
    features: np.ndarray,
    benchmarks: np.ndarray,
    reference: str = "bbob",
    subset_size: int = SUBSET_SIZE,
    subsets: int = SUBSETS,
    seed: int = 0,
) -> pd.DataFrame:
    """``C@n``, ``VS@n`` and ``N`` of every benchmark in one pool.

    ``features`` has one row per problem of the pool, ``benchmarks`` its benchmark
    labels. One row per benchmark: ``benchmark``, ``size``, ``subsampled``,
    ``coverage`` and ``vendi`` (means over ``subsets`` random subsets of
    ``subset_size`` problems) with their 2.5 / 97.5 percentiles (``*_low``,
    ``*_high``), and ``novelty`` w.r.t. ``reference``. A benchmark with at most
    ``subset_size`` problems is not subsampled (``subsampled`` is False): its metrics
    are those of the whole benchmark and the interval is a point. The subsets of a
    benchmark depend only on ``seed`` and its label.
    """
    benchmarks = np.asarray(benchmarks)
    z = standardize(np.asarray(features, dtype=float))
    if not np.any(benchmarks == reference):
        raise ValueError(f"The reference benchmark {reference!r} is not in the pool.")
    eps = radius(z)
    bandwidth = float(np.median(pdist(z)))
    metrics: dict[str, Callable[[np.ndarray], float]] = {
        "coverage": lambda b: coverage(z, b, eps),
        "vendi": lambda b: vendi_score(b, bandwidth),
    }
    rows = []
    for name in pd.unique(benchmarks):
        b = z[benchmarks == name]
        subsampled = len(b) > subset_size
        if subsampled:
            rng = np.random.default_rng(_seed(seed, str(name)))
            subsets_ = [b[rng.choice(len(b), subset_size, replace=False)] for _ in range(subsets)]
        else:
            subsets_ = [b]
        row = {"benchmark": name, "size": len(b), "subsampled": subsampled}
        for metric, compute in metrics.items():
            values = [compute(subset) for subset in subsets_]
            low, high = np.percentile(values, [2.5, 97.5])
            row |= {metric: float(np.mean(values)), f"{metric}_low": low, f"{metric}_high": high}
        row["novelty"] = novelty(b, z[benchmarks == reference], eps)
        rows.append(row)
    return pd.DataFrame(rows)


def _seed(seed: int, label: str) -> np.random.SeedSequence:
    digest = hashlib.sha256(label.encode()).digest()
    return np.random.SeedSequence([seed, *np.frombuffer(digest, dtype=np.uint32)])
