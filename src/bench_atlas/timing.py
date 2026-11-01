"""Evaluation cost of benchmark problems."""

from __future__ import annotations

import time

import numpy as np
from threadpoolctl import threadpool_limits

from bench_atlas.problems import Problem

__all__ = ["evaluation_time"]

WARMUP = 10
MIN_BATCHES = 10


def evaluation_time(problem: Problem, batch_size: int = 1, n_points: int = 1000) -> float:
    """Median wall time in seconds per point when evaluating batches of ``batch_size``.

    Times ``max(MIN_BATCHES, n_points // batch_size)`` batches of uniform random
    points on [0, 1]^d, after ``WARMUP`` untimed batches, and divides the median
    batch time by ``batch_size``. BLAS / OpenMP thread pools are limited to one
    thread. The points come from a fixed seed, so every call times the same points.
    """
    rng = np.random.default_rng(0)
    n_batches = max(MIN_BATCHES, n_points // batch_size)
    times = np.empty(n_batches)
    with threadpool_limits(limits=1):
        for _ in range(WARMUP):
            problem(rng.random((batch_size, problem.dimension)))
        for i in range(n_batches):
            x = rng.random((batch_size, problem.dimension))
            start = time.perf_counter()
            problem(x)
            times[i] = time.perf_counter() - start
    return float(np.median(times)) / batch_size
