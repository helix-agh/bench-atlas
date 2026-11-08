"""Local optima network (LON) features computed with lonkit."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from lonkit import CMLON, BasinHoppingSampler, BasinHoppingSamplerConfig, LONConfig

from bench_atlas.problems import Problem

__all__ = ["cmlon", "lon_features"]

STEP_SIZE = 0.1
N_RUNS = 100
N_ITER_NO_CHANGE = 250
BUDGET_PER_DIMENSION = 10_000_000

_BUDGET_EXHAUSTED = "evaluation budget exhausted"


def lon_features(
    problem: Problem,
    seed: int,
    budget_per_dimension: int = BUDGET_PER_DIMENSION,
    n_runs: int = N_RUNS,
    n_iter_no_change: int = N_ITER_NO_CHANGE,
) -> dict[str, float]:
    """CMLON metrics of ``problem`` from one basin-hopping sample (see :func:`cmlon`).

    ``evaluations`` is the number of objective evaluations used and ``n_runs`` the
    number of runs that started within the budget. ``success`` and ``deviation`` are
    measured against the best fitness found, as in lonkit.
    """
    start = problem.evaluations
    network = cmlon(problem, seed, budget_per_dimension, n_runs, n_iter_no_change)
    source = network.source_lon
    runs = source.final_run_values if source is not None else None
    return {
        "evaluations": problem.evaluations - start,
        "n_runs": 0 if runs is None else len(runs),
        **network.compute_metrics(),
    }


def cmlon(
    problem: Problem,
    seed: int,
    budget_per_dimension: int = BUDGET_PER_DIMENSION,
    n_runs: int = N_RUNS,
    n_iter_no_change: int = N_ITER_NO_CHANGE,
) -> CMLON:
    """Compressed monotonic LON of ``problem`` sampled by lonkit's basin hopping.

    ``n_runs`` runs of L-BFGS-B local searches with uniform perturbations of fixed
    size ``STEP_SIZE`` on [0, 1]^d, each stopping after ``n_iter_no_change``
    non-improving perturbations. At most ``budget_per_dimension * d`` points are
    evaluated: once the budget is spent, the current run stops and the remaining
    runs are skipped. Runs are sequential, so the result depends only on ``seed``.
    Optima are identified by coordinates rounded to 5 decimals; their fitness is the
    minimum over the merged points (lonkit's default, without its warning).
    """
    d = problem.dimension
    objective = _PointObjective(problem, problem.evaluations + budget_per_dimension * d)
    sampler = BasinHoppingSampler(
        BasinHoppingSamplerConfig(
            n_runs=n_runs,
            n_iter_no_change=n_iter_no_change,
            step_mode="fixed",
            step_size=STEP_SIZE,
            minimizer_method="L-BFGS-B",
            seed=seed,
            n_jobs=1,
        )
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=f".*{_BUDGET_EXHAUSTED}")
        sample = sampler.sample(objective, [(0.0, 1.0)] * d)
    return sampler.sample_to_lon(sample, LONConfig(warn_on_duplicates=False)).to_cmlon()


class _BudgetExhausted(ValueError):
    """Raised by the objective once the budget is spent.

    lonkit treats a ``ValueError`` from the local search as a failed search: it
    rejects the perturbation, or skips the run if it was the initial search.
    """


@dataclass(frozen=True, eq=False)
class _PointObjective:
    """Single-point view ``x (d,) -> float`` of a problem, limited to ``limit`` evaluations."""

    problem: Problem
    limit: int

    def __call__(self, x: np.ndarray) -> float:
        if self.problem.evaluations >= self.limit:
            raise _BudgetExhausted(_BUDGET_EXHAUSTED)
        return float(self.problem(x[np.newaxis])[0])
