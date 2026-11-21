"""Optimizer portfolio run with Nevergrad.

:func:`optimize` runs one optimizer on a problem with Nevergrad's ask/tell interface
and returns its best-so-far curve, stored as the evaluations at which the best value
improved. The curve is lossless and much shorter than one value per evaluation;
:func:`best_so_far` expands it, e.g. to compute AOCC at several budgets.

The portfolio (:data:`PORTFOLIO`) has one optimizer per method class:

- ``CMAstd``: CMA-ES with full covariance, non-elitist, default population size. It
  starts with a step size of 0.25 of the range (:data:`CMA_SIGMA0`); with Nevergrad's
  default of 0.024 it would be a local search from the centre. (Nevergrad's
  ``CMA`` is not used: on a bounded domain it selects an elitist diagonal CMA-ES with
  a small population.)
- ``LBFGSB``: L-BFGS-B, a quasi-Newton method, with random restarts.
- ``RPowell``: Powell's direction-set method with random restarts.

Each of the three is the best optimizer on a different part of the benchmark
problems. Other Nevergrad optimizers, such as differential evolution, particle swarm,
diagonal CMA-ES or random search, barely improved on the best of these three, so they
are not in the portfolio; they can still be run by name. The portfolio is identified
by its optimizer names and the Nevergrad version, which the executor records in every
row.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import nevergrad as ng
import numpy as np
from nevergrad.optimization.base import ConfiguredOptimizer, Optimizer
from nevergrad.optimization.optimizerlib import ParametrizedCMA

from bench_atlas.problems import Problem

__all__ = [
    "BUDGET_PER_DIMENSION",
    "CMA_SIGMA0",
    "PORTFOLIO",
    "SIGMA",
    "best_so_far",
    "optimize",
    "optimizer_class",
    "unit_box",
]

BUDGET_PER_DIMENSION = 1000
SIGMA = 1 / 6
CMA_SIGMA0 = 0.25

OptimizerClass = ConfiguredOptimizer | type[Optimizer] | Callable[..., Optimizer]


def _cmastd(parametrization: ng.p.Array, budget: int, num_workers: int) -> Optimizer:
    """Nevergrad's ``CMAstd`` with an initial step size of ``CMA_SIGMA0`` on [0, 1]^d.

    On a bounded domain Nevergrad starts CMA-ES with ``scale`` times 0.3 (d < 18) or 0.15
    (d >= 18), in units of the parametrization's sigma (``SIGMA``).
    """
    multiplier = 0.3 if parametrization.dimension < 18 else 0.15
    config = ParametrizedCMA(
        scale=CMA_SIGMA0 / (SIGMA * multiplier),
        popsize_factor=3,
        elitist=False,
        diagonal=False,
        fcmaes=False,
    )
    return config(parametrization, budget=budget, num_workers=num_workers)


PORTFOLIO: dict[str, OptimizerClass] = {
    "CMAstd": _cmastd,
    "LBFGSB": ng.optimizers.registry["LBFGSB"],
    "RPowell": ng.optimizers.registry["RPowell"],
}


def optimize(
    problem: Problem,
    seed: int,
    optimizer: str,
    budget_per_dimension: int = BUDGET_PER_DIMENSION,
) -> dict[str, Any]:
    """One run of ``optimizer`` on ``problem`` with ``budget_per_dimension * d`` evaluations.

    The search space is [0, 1]^d (:func:`unit_box`), and all randomness comes
    from ``seed`` through the parametrization's ``random_state``. Points are
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
    parametrization = unit_box(d)
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


def unit_box(dimension: int) -> ng.p.Array:
    """[0, 1]^d with uniform sampling and Nevergrad's mutation sigma of 1/6, from the centre."""
    return (
        ng.p.Array(init=np.full(dimension, 0.5))
        .set_mutation(sigma=SIGMA)
        .set_bounds(0.0, 1.0, full_range_sampling=True)
    )


def optimizer_class(name: str) -> OptimizerClass:
    """Optimizer ``name`` from :data:`PORTFOLIO`, else from Nevergrad's registry.

    Raises:
        KeyError: Unknown optimizer.
    """
    if name in PORTFOLIO:
        return PORTFOLIO[name]
    if name in ng.optimizers.registry:
        return ng.optimizers.registry[name]
    raise KeyError(f"Unknown optimizer {name!r}")
