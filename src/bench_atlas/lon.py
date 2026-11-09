"""Local optima network (LON) features computed with lonkit.

Sampling and network construction are separate. :func:`sample_run` runs one
basin-hopping run and returns the chain of local optima it accepted, at full
precision. :func:`cmlon_from_runs` rounds the optima of many runs to the chosen
precision, builds the LON from them and compresses it to a CMLON. The precision does
not affect the search, so a CMLON can be rebuilt at another precision without
sampling again, and the runs of one problem can be sampled in parallel.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from lonkit import CMLON, LON, BasinHoppingSampler, BasinHoppingSamplerConfig, LONConfig

from bench_atlas.problems import Problem

__all__ = ["cmlon_features", "cmlon_from_runs", "sample_run"]

STEP_SIZE = 0.1
N_RUNS = 100
N_ITER_NO_CHANGE = 250
RUN_BUDGET_PER_DIMENSION = 1_000_000
COORDINATE_PRECISION = 4
FITNESS_PRECISION = 4

_BUDGET_EXHAUSTED = "evaluation budget exhausted"
_TRACE_COLUMNS = ["run", "fit1", "node1", "fit2", "node2"]


def sample_run(
    problem: Problem,
    seed: int,
    run_budget_per_dimension: int = RUN_BUDGET_PER_DIMENSION,
    n_iter_no_change: int = N_ITER_NO_CHANGE,
) -> dict[str, Any]:
    """One lonkit basin-hopping run on ``problem``.

    L-BFGS-B local searches with uniform perturbations of fixed size ``STEP_SIZE`` on
    [0, 1]^d; the run stops after ``n_iter_no_change`` non-improving perturbations or
    when ``run_budget_per_dimension * d`` points have been evaluated.

    Returns:
        ``evaluations``; ``optima_f``, the fitness of the initial local optimum and of
        every accepted one, in order; and ``optima_x``, their coordinates, flattened
        (d values per optimum). Both are empty if the initial local search failed.
    """
    d = problem.dimension
    start = problem.evaluations
    objective = _PointObjective(problem, start + run_budget_per_dimension * d)
    sampler = BasinHoppingSampler(
        BasinHoppingSamplerConfig(
            n_runs=1,
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
    optima_x, optima_f = _accepted_optima(sample.raw_records)
    return {
        "evaluations": problem.evaluations - start,
        "optima_f": optima_f,
        "optima_x": [float(v) for x in optima_x for v in x],
    }


def cmlon_from_runs(
    runs: Iterable[tuple[Sequence[float], Sequence[float]]],
    dimension: int,
    coordinate_precision: int | None = COORDINATE_PRECISION,
    fitness_precision: int | None = FITNESS_PRECISION,
) -> CMLON:
    """CMLON of the runs of one problem, each given as ``(optima_x, optima_f)``.

    Builds lonkit's trace format: optima are identified by their coordinates rounded
    to ``coordinate_precision`` decimals, fitness is rounded to ``fitness_precision``
    decimals (``None``: no rounding, as in lonkit), every pair of consecutive optima of
    a run is a transition, and a run that never moved is a self-loop (as lonkit
    records it). L-BFGS-B stops a little
    short of an optimum, so with lonkit's defaults (5 decimals, full fitness
    precision) runs that end in the same basin become separate optima and funnels.
    A merged optimum takes the minimum fitness of its points (lonkit's default,
    without its warning).
    """
    rows = []
    for run, (optima_x, optima_f) in enumerate(runs, start=1):
        if len(optima_f) == 0:
            continue
        nodes = [
            _node(x, coordinate_precision) for x in np.reshape(optima_x, (len(optima_f), dimension))
        ]
        fitness = np.asarray(optima_f, dtype=float)
        if fitness_precision is not None:
            fitness = np.round(fitness, fitness_precision)
        if len(nodes) == 1:
            rows.append((run, fitness[0], nodes[0], fitness[0], nodes[0]))
        rows += [
            (run, fitness[i], nodes[i], fitness[i + 1], nodes[i + 1]) for i in range(len(nodes) - 1)
        ]
    if not rows:
        return LON().to_cmlon()
    trace = pd.DataFrame(rows, columns=_TRACE_COLUMNS)
    return LON.from_trace_data(trace, config=LONConfig(warn_on_duplicates=False)).to_cmlon()


def cmlon_features(cmlon: CMLON) -> dict[str, float]:
    """``n_runs`` (runs in the network) and all lonkit CMLON metrics.

    ``success`` and ``deviation`` are measured against the best fitness found, as in
    lonkit.
    """
    source = cmlon.source_lon
    runs = source.final_run_values if source is not None else None
    return {"n_runs": 0 if runs is None else len(runs), **cmlon.compute_metrics()}


def _accepted_optima(records: list[dict]) -> tuple[list[np.ndarray], list[float]]:
    """Initial local optimum and every accepted one, in order.

    Moves are accepted from the current optimum, which changes only on acceptance,
    so consecutive optima are exactly lonkit's accepted transitions. A run without
    accepted moves keeps its initial optimum (lonkit's preserved record).
    """
    if not records:
        return [], []
    optima_x = [records[0]["current_x"]]
    optima_f = [float(records[0]["current_f"])]
    for record in records:
        if record["accepted"]:
            optima_x.append(record["new_x"])
            optima_f.append(float(record["new_f"]))
    return optima_x, optima_f


def _node(x: np.ndarray, precision: int | None) -> str:
    """Node id of a point, formatted like lonkit's: rounded, with ``-0.0`` as ``0.0``."""
    if precision is None:
        return "_".join(str(v) for v in x + 0.0)
    return "_".join(f"{v:.{precision}f}" for v in np.round(x, precision) + 0.0)


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
