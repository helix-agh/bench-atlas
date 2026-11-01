"""Evaluation cost of benchmark problems."""

from __future__ import annotations

import time

import numpy as np
from threadpoolctl import threadpool_limits

from bench_atlas.problems import Problem

__all__ = ["evaluation_time"]

WARMUP = 10


def evaluation_time(problem: Problem, n_points: int = 1000) -> float:
    """Median wall time in seconds of a single evaluation of ``problem``.

    Each of ``n_points`` uniform random points on [0, 1]^d is evaluated on its
    own (a batch of one point) and timed, after ``WARMUP`` untimed evaluations.
    BLAS / OpenMP thread pools are limited to one thread during the measurement.
    The points come from a fixed seed, so every call times the same points.
    """
    points = np.random.default_rng(0).random((WARMUP + n_points, 1, problem.dimension))
    times = np.empty(n_points)
    with threadpool_limits(limits=1):
        for x in points[:WARMUP]:
            problem(x)
        for i, x in enumerate(points[WARMUP:]):
            start = time.perf_counter()
            problem(x)
            times[i] = time.perf_counter() - start
    return float(np.median(times))
