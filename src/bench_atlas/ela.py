"""Exploratory landscape analysis (ELA) features computed with orivex."""

from __future__ import annotations

import warnings

import numpy as np
import orivex
from scipy.stats import qmc

from bench_atlas.problems import Problem

__all__ = ["POINTS_PER_DIMENSION", "ela_features"]

# 250·d, not the 50·d of algorithm selection (Kerschke et al. 2016), which matches the
# cost of one EA initial population rather than feature stability. 250·d is the most
# common size in papers that compare suites; at 50·d BBOB functions cannot be told apart
# for d <= 20 (Renau et al. 2021) and features are biased, not only noisy (Belkhir et al.
# 2016). It also gives enough points for a q = 0.1 level-set split at d = 2 and for the
# 861 coefficients of quad_w_interact at d = 40. Details and references: issues #24, #41.
POINTS_PER_DIMENSION = 250


def ela_features(
    problem: Problem, seed: int, points_per_dimension: int = POINTS_PER_DIMENSION
) -> dict[str, float]:
    """All orivex ELA features of ``problem`` on one Sobol sample.

    The sample has ``points_per_dimension * d`` scrambled Sobol points on
    [0, 1]^d drawn with ``seed``; objective values are min-max normalized by
    orivex. Features that orivex cannot compute are NaN. ``f_med`` is the
    median objective value of the (unnormalized) sample.
    """
    d = problem.dimension
    with warnings.catch_warnings():
        # The sample size is usually not a power of 2; the balance warning is expected.
        warnings.simplefilter("ignore", UserWarning)
        x = qmc.Sobol(d, scramble=True, seed=seed).random(points_per_dimension * d)
    y = problem(x)
    sample = orivex.LandscapeSample(x, y, lower=np.zeros(d), upper=np.ones(d))
    result = orivex.compute(sample, "*", y_normalization="minmax")
    features = {
        name: np.nan if value.value is None else float(value.value)
        for name, value in result.values.items()
    }
    return {"f_med": float(np.median(y)), **features}
