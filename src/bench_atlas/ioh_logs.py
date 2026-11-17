"""IOH logs of stored portfolio runs.

IOH logs are the IOHprofiler format that IOHanalyzer and IOHinspector read. They are
written by IOH itself: each run of table ``runs`` is replayed through a problem that
computes nothing. Evaluation by evaluation it returns values that reproduce the run's
improvements and its last value, and an ``ioh.logger.Analyzer`` attached to it logs
them as during a live run. The ``.dat`` files therefore equal those of a live IOH run
with the same optimizer and seed.

Like IOH's BBOB problems, which log the objective before adding the optimum, the logged
values are ``f - f_opt`` when the optimum is known, and ``f`` otherwise. The best point
in the JSON files is on [0, 1]^d, the domain of bench-atlas problems, not in the
problem's original box. Each run also records its ``replicate`` and ``seed``.
"""

from __future__ import annotations

import math
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import ioh
import joblib
import numpy as np
import pandas as pd

from bench_atlas import store
from bench_atlas.portfolio import best_so_far

__all__ = ["select", "write", "write_all"]

_PROBLEM_COLUMNS = ["function", "dimension", "instance"]
_RUN_ORDER = [*_PROBLEM_COLUMNS, "replicate", "seed"]


def select(
    runs: pd.DataFrame,
    budget_per_dimension: int,
    suites: Sequence[str] | None = None,
    optimizers: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Rows of ``runs`` with the given budget, and of the given suites and optimizers
    (all if ``None`` or empty)."""
    if runs.empty:
        return runs
    keep = runs["budget_per_dimension"] == budget_per_dimension
    if suites:
        keep &= runs["suite"].isin(suites)
    if optimizers:
        keep &= runs["optimizer"].isin(optimizers)
    return runs[keep]


def write_all(
    runs: pd.DataFrame,
    results: Path,
    n_jobs: int = 1,
    log: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Write the IOH logs of ``runs`` to ``<results>/ioh-logs/<suite>/<optimizer>/``.

    The (suite, optimizer) groups are written in parallel with joblib; each group has its
    own folder and logger. Earlier logs of the same groups are replaced.

    Returns:
        One row per group: ``suite``, ``optimizer``, ``runs`` and ``folder``.
    """
    groups = [
        (suite, optimizer, group, store.ioh_logs_path(results, suite, optimizer))
        for (suite, optimizer), group in runs.groupby(["suite", "optimizer"], sort=True)
    ]
    counts = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(write)(group, folder) for _, _, group, folder in groups
    )
    summary = pd.DataFrame(
        [
            {"suite": suite, "optimizer": optimizer, "runs": count, "folder": str(folder)}
            for (suite, optimizer, _, folder), count in zip(groups, counts, strict=True)
        ],
        columns=["suite", "optimizer", "runs", "folder"],
    )
    if log is not None:
        log(f"ioh-logs: {summary['runs'].sum()} runs written to {len(summary)} folders")
    return summary


def write(runs: pd.DataFrame, folder: Path) -> int:
    """Write the IOH logs of ``runs`` (one optimizer and one budget) to ``folder``.

    ``folder`` is replaced as a whole: the files are written to a temporary folder next
    to it, which then takes its place. The algorithm name is the optimizer; its info
    gives the Nevergrad version and the budget per dimension.

    Returns:
        The number of logged runs.

    Raises:
        ValueError: ``runs`` mixes optimizers, budgets or Nevergrad versions.
    """
    folder = Path(folder)
    optimizer = _single(runs, "optimizer")
    budget = _single(runs, "budget_per_dimension")
    version = _single(runs, "version_nevergrad")
    folder.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=folder.parent, prefix=f".{folder.name}.") as tmp:
        logger = ioh.logger.Analyzer(
            root=tmp,
            folder_name=folder.name,
            algorithm_name=str(optimizer),
            algorithm_info=f"nevergrad {version}, budget_per_dimension {budget}",
        )
        run_info = _RunInfo()
        logger.add_run_attributes(run_info, ["replicate", "seed"])
        for _, group in runs.sort_values(_RUN_ORDER).groupby(_PROBLEM_COLUMNS, sort=False):
            problem = _ReplayProblem(group.iloc[0])
            problem.attach_logger(logger)
            for row in group.to_dict("records"):
                run_info.replicate = float(row["replicate"])
                run_info.seed = float(row["seed"])
                problem.replay(row)
                problem.reset()
            problem.detach_logger()
        logger.close()
        if folder.exists():
            folder.rename(Path(tmp) / ".previous")
        (Path(tmp) / folder.name).rename(folder)
    return len(runs)


class _ReplayProblem(ioh.problem.RealSingleObjective):
    """IOH problem that replays stored runs of one bench-atlas problem.

    Its id, name, instance and dimension are those of the problem, its bounds are
    [0, 1]^d, and its optimum is 0 (``f - f_opt`` is replayed) or NaN if ``f_opt`` is
    unknown.
    """

    def __init__(self, row: pd.Series) -> None:
        d = int(row["dimension"])
        known = not math.isnan(row["f_opt"])
        super().__init__(
            str(row["name"]),
            d,
            int(row["instance"]),
            True,
            ioh.RealBounds([0.0] * d, [1.0] * d),
            [],
            ioh.RealSolution([math.nan] * d, 0.0 if known else math.nan),
        )
        self.set_id(int(row["function"]))
        self._offset = float(row["f_opt"]) if known else 0.0
        self._values = np.empty(0)
        self._next = 0

    def replay(self, run: dict[str, Any]) -> None:
        """Evaluate the run again, so that the attached logger records it.

        Before and after its best point the run evaluates the centre of the box;
        IOH keeps no other positions.
        """
        evaluations = int(run["evaluations"])
        if evaluations == 0:
            return
        values = best_so_far(run["curve_evaluations"], run["curve_f"], evaluations)
        values[-1] = run["f_last"]
        self._values = values - self._offset
        self._next = 0
        best = int(run["curve_evaluations"][-1])
        d = self.meta_data.n_variables
        self._evaluate_centre(best - 1, d)
        self(np.asarray(run["x_best"], dtype=float))
        self._evaluate_centre(evaluations - best, d)

    def evaluate(self, _x: Sequence[float]) -> float:
        value = self._values[self._next]
        self._next += 1
        return float(value)

    def _evaluate_centre(self, n: int, d: int) -> None:
        if n > 0:
            self(np.full((n, d), 0.5))


class _RunInfo:
    """Run attributes that the logger reads at the end of each run."""

    replicate: float = 0.0
    seed: float = 0.0


def _single(runs: pd.DataFrame, column: str) -> Any:
    values = runs[column].unique()
    if len(values) != 1:
        raise ValueError(f"Expected one {column} per log folder, got {list(values)}")
    return values[0]
