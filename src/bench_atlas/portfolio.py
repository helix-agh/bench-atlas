"""Optimizer portfolio run with Nevergrad.

:func:`optimize` runs one optimizer on a problem with Nevergrad's ask/tell interface
and returns its best-so-far curve, stored as the evaluations at which the best value
improved. The curve is lossless and much shorter than one value per evaluation;
:func:`best_so_far` expands it, e.g. to compute AOCC at several budgets.

The portfolio (:data:`PORTFOLIO`) covers different method classes. It is frozen by
its optimizer names and the Nevergrad version, which the executor records in every
row. Two choices differ from Nevergrad's plain names:

- ``CMAstd`` instead of ``CMA``. On a bounded domain Nevergrad's ``CMA`` selects
  ``CMAbounded``, an elitist diagonal CMA-ES with a small population, so it would
  nearly duplicate ``DiagonalCMA``. ``CMAstd`` (what ``CMA`` selects without bounds) is
  the full-covariance, non-elitist CMA-ES with the default population.
- ``RNelderMead`` instead of ``NelderMead``. Nevergrad's ``NelderMead`` ignores the
  seed and, once the simplex has collapsed, restarts from the best point and evaluates
  it over and over (on the sphere in d = 2, 92 % of 2000 evaluations repeat a point).
  ``RNelderMead`` restarts from a random point, like Nevergrad's ``RPowell`` and
  ``RCobyla``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import nevergrad as ng
import numpy as np
from nevergrad.optimization.base import ConfiguredOptimizer, Optimizer
from nevergrad.optimization.recastlib import NonObjectOptimizer

from bench_atlas.problems import Problem

__all__ = ["BUDGET_PER_DIMENSION", "PORTFOLIO", "best_so_far", "optimize", "optimizer_class"]

BUDGET_PER_DIMENSION = 1000

PORTFOLIO: dict[str, ConfiguredOptimizer | type[Optimizer]] = {
    "CMAstd": ng.optimizers.registry["CMAstd"],
    "DiagonalCMA": ng.optimizers.registry["DiagonalCMA"],
    "TwoPointsDE": ng.optimizers.registry["TwoPointsDE"],
    "PSO": ng.optimizers.registry["PSO"],
    "RNelderMead": NonObjectOptimizer(method="Nelder-Mead", random_restart=True).set_name(
        "RNelderMead"
    ),
    "LBFGSB": ng.optimizers.registry["LBFGSB"],
    "OnePlusOne": ng.optimizers.registry["OnePlusOne"],
    "RandomSearch": ng.optimizers.registry["RandomSearch"],
}


def optimize(
    problem: Problem,
    seed: int,
    optimizer: str,
    budget_per_dimension: int = BUDGET_PER_DIMENSION,
) -> dict[str, Any]:
    """One run of ``optimizer`` on ``problem`` with ``budget_per_dimension * d`` evaluations.

    The search space is [0, 1]^d (``ng.p.Array`` with these bounds), and all randomness
    comes from ``seed`` through the parametrization's ``random_state``. Points are
    evaluated one at a time; the run ends early only if the optimizer raises
    ``NevergradEarlyStopping``.

    Args:
        problem: Problem to minimize.
        seed: Seed of the run.
        optimizer: Name in :data:`PORTFOLIO` or in Nevergrad's registry.
        budget_per_dimension: Evaluations per dimension.

    Returns:
        The problem's ``name`` and ``f_opt`` (NaN if unknown), so that a row describes
        its problem (e.g. for IOH logs); ``evaluations`` used; ``best_f``,
        the best evaluated value, and ``x_best``, its point on [0, 1]^d; ``f_last``,
        the value of the last evaluation (IOH logs it when a run ends); and the
        best-so-far curve as ``curve_evaluations``, the evaluation numbers (from 1) at
        which the best value improved, with ``curve_f``, the best value after each of
        them.
    """
    d = problem.dimension
    budget = budget_per_dimension * d
    parametrization = ng.p.Array(shape=(d,), lower=0.0, upper=1.0)
    parametrization.random_state = np.random.RandomState(seed)
    instance = optimizer_class(optimizer)(
        parametrization=parametrization, budget=budget, num_workers=1
    )
    best = math.inf
    f = math.nan
    x_best = np.full(d, np.nan)
    curve_evaluations: list[int] = []
    curve_f: list[float] = []
    evaluations = 0
    for evaluation in range(1, budget + 1):
        try:
            candidate = instance.ask()
        except ng.errors.NevergradEarlyStopping:
            break
        f = float(problem(candidate.value[np.newaxis, :])[0])
        instance.tell(candidate, f)
        evaluations = evaluation
        if f < best:
            best = f
            x_best = np.array(candidate.value, dtype=float)
            curve_evaluations.append(evaluation)
            curve_f.append(f)
    return {
        "name": problem.name,
        "f_opt": math.nan if problem.f_opt is None else problem.f_opt,
        "evaluations": evaluations,
        "best_f": best,
        "x_best": x_best.tolist(),
        "f_last": f,
        "curve_evaluations": curve_evaluations,
        "curve_f": curve_f,
    }


def best_so_far(
    curve_evaluations: Sequence[int], curve_f: Sequence[float], budget: int
) -> np.ndarray:
    """Best value after each of the first ``budget`` evaluations.

    Expands a curve returned by :func:`optimize`. After the last improvement, and after
    the end of a run that stopped early, the curve keeps its last value. Before the
    first improvement it is NaN.
    """
    evaluations = np.asarray(curve_evaluations, dtype=int)
    values = np.asarray(curve_f, dtype=float)
    last = np.searchsorted(evaluations, np.arange(1, budget + 1), side="right") - 1
    if len(values) == 0:
        return np.full(budget, np.nan)
    return np.where(last >= 0, values[np.maximum(last, 0)], np.nan)


def optimizer_class(name: str) -> ConfiguredOptimizer | type[Optimizer]:
    """Optimizer ``name`` from :data:`PORTFOLIO`, else from Nevergrad's registry.

    Raises:
        KeyError: Unknown optimizer.
    """
    if name in PORTFOLIO:
        return PORTFOLIO[name]
    if name in ng.optimizers.registry:
        return ng.optimizers.registry[name]
    raise KeyError(f"Unknown optimizer {name!r}")
