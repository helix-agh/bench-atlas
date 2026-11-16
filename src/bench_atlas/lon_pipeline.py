"""LON analysis in two stages: basin-hopping runs, then CMLONs.

Stage 1 samples every run of every problem and replicate as its own executor task
(table ``lon_runs``), so the runs of one problem are spread over all workers. Stage 2
builds, for each problem and replicate, the CMLON from its runs at the requested
precision, saves it to ``<results>/cmlon/`` and adds its metrics to table ``lon``.
The precision is not part of the stage-1 cache key, so another precision reuses
all stored runs.
"""

from __future__ import annotations

import functools
import hashlib
import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from bench_atlas import executor, lon, store
from bench_atlas.problems import Problem

__all__ = ["Settings", "run"]

_Chains = dict[tuple[str, int], tuple[int, list[tuple[Sequence[float], Sequence[float]]], int]]


@dataclass(frozen=True)
class Settings:
    """Sampler and network settings of a LON analysis."""

    runs: int = lon.N_RUNS
    iter_no_change: int = lon.N_ITER_NO_CHANGE
    coordinate_precision: int = lon.COORDINATE_PRECISION
    fitness_precision: int = lon.FITNESS_PRECISION

    def sampling(self) -> dict[str, Any]:
        """Cache key of one run: the settings that change the search."""
        return {"iter_no_change": self.iter_no_change, "step_size": lon.STEP_SIZE}

    def network(self) -> dict[str, Any]:
        """Cache key of one CMLON: all settings."""
        return {**asdict(self), "step_size": lon.STEP_SIZE}


def run(
    problems: Sequence[Problem],
    replicates: Iterable[int],
    settings: Settings,
    results: Path,
    n_jobs: int = 1,
    global_seed: int = 0,
    log: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Rows of table ``lon`` for every problem and replicate, computing only missing ones.

    A row holds the problem id, ``replicate``, ``seed``, the settings, library
    versions, ``evaluations`` (summed over the runs), ``n_runs``, the lonkit CMLON
    metrics and ``cmlon``, the name of the saved graph (see
    :func:`bench_atlas.store.load_cmlon`). ``n_jobs`` parallelizes the runs; the
    CMLONs are built serially.
    """
    replicates = list(replicates)
    runs = executor.run(
        "lon_runs",
        problems,
        replicates,
        functools.partial(lon.sample_run, n_iter_no_change=settings.iter_no_change),
        settings.sampling(),
        results,
        n_jobs=n_jobs,
        global_seed=global_seed,
        log=log,
        runs=settings.runs,
    )
    chains = _chains(runs, problems, replicates, global_seed)
    return executor.run(
        "lon",
        problems,
        replicates,
        functools.partial(_network, chains=chains, results=results, settings=settings),
        settings.network(),
        results,
        global_seed=global_seed,
        log=log,
    )


def _chains(
    runs: pd.DataFrame, problems: Sequence[Problem], replicates: list[int], global_seed: int
) -> _Chains:
    """Runs of every (problem, replicate seed): replicate, run optima, evaluations."""
    grouped = {
        key: (
            list(zip(group["optima_x"], group["optima_f"], strict=True)),
            int(group["evaluations"].sum()),
        )
        for key, group in runs.groupby(["problem", "replicate"], sort=False)
    }
    chains: _Chains = {}
    for problem in problems:
        for replicate in replicates:
            optima, evaluations = grouped[(str(problem.id), replicate)]
            seed = executor.task_seed(global_seed, problem.id, replicate)
            chains[(str(problem.id), seed)] = (replicate, optima, evaluations)
    return chains


def _network(
    problem: Problem, seed: int, chains: _Chains, results: Path, settings: Settings
) -> dict[str, Any]:
    """Build, save and describe the CMLON of one problem and replicate."""
    replicate, optima, evaluations = chains[(str(problem.id), seed)]
    cmlon = lon.cmlon_from_runs(
        optima, problem.dimension, settings.coordinate_precision, settings.fitness_precision
    )
    key = json.dumps({"seed": seed, **settings.network()}, sort_keys=True)
    name = f"{problem.id}_r{replicate}_{hashlib.sha256(key.encode()).hexdigest()[:8]}"
    store.save_cmlon(results, name, cmlon)
    return {"evaluations": evaluations, **lon.cmlon_features(cmlon), "cmlon": name}
