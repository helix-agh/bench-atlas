"""Normalized area over the convergence curve (AOCC) of the portfolio runs.

With ``f*`` the known optimum (or the best value found by any portfolio run) and
``f_med`` the median of f on the ELA sample, a run's error after ``t`` evaluations is

    e(t) = (f_best(t) - f*) / (f_med - f*)

and its AOCC over ``t_max`` evaluations is

    AOCC = 1/t_max * sum_{t=1}^{t_max} [1 - clip((log10 e(t) + 8) / 8, 0, 1)].

AOCC = 0 means "no better than a typical random point"; AOCC = 1 means reaching
``1e-8`` at once. This is the standard AOCC on ``f_best - f*`` with bounds
``[1e-8 * delta, delta]``, ``delta = f_med - f*`` ([`aocc`][bench_atlas.aocc]). It is computed after
the runs, from their stored best-so-far curves, because for problems without a known
optimum ``f*`` is known only once all runs are done. Runs that stopped early keep their
last best value; shorter budgets use prefixes of the same curves.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from bench_atlas.ela import POINTS_PER_DIMENSION as ELA_POINTS_PER_DIMENSION
from bench_atlas.portfolio import best_so_far

__all__ = [
    "BUDGETS_PER_DIMENSION",
    "ELA_POINTS_PER_DIMENSION",
    "PRECISION",
    "aocc",
    "aocc_table",
    "normalized_aocc",
]

PRECISION = 1e-8
BUDGETS_PER_DIMENSION = (100, 300, 1000)

_RUN_COLUMNS = [
    "problem",
    "suite",
    "function",
    "instance",
    "dimension",
    "replicate",
    "global_seed",
    "seed",
    "optimizer",
    "budget_per_dimension",
]


def aocc_table(
    runs: pd.DataFrame,
    ela: pd.DataFrame,
    budgets_per_dimension: Iterable[int] = BUDGETS_PER_DIMENSION,
    ela_points_per_dimension: int = ELA_POINTS_PER_DIMENSION,
) -> pd.DataFrame:
    """Normalized AOCC of every run of table ``runs``, at several budgets.

    ``f_med`` of a problem is the median of ``f_med`` over its ELA replicates computed
    with ``ela_points_per_dimension``. ``f*`` is the run's ``f_opt``, or, if it is
    unknown, the best ``best_f`` of all runs of the problem (``f_opt_known`` is then
    False).

    Returns:
        One row per run, in the order of ``runs``: its id columns, ``f_opt`` (the
        ``f*`` used), ``f_opt_known``, ``f_med`` and ``aocc_<m>d`` for each budget of
        ``m * d`` evaluations (once per budget, in the order first given); NaN where
        ``m`` exceeds the run's budget per dimension.

    Raises:
        ValueError: A problem of ``runs`` has no ELA rows.
    """
    budgets = list(dict.fromkeys(budgets_per_dimension))
    f_med = _f_med(runs, ela, ela_points_per_dimension)
    f_opt, known = _f_opt(runs)
    rows = []
    for run in runs.to_dict("records"):
        problem = run["problem"]
        row = {
            **{column: run[column] for column in _RUN_COLUMNS},
            "f_opt": float(f_opt[problem]),
            "f_opt_known": bool(known[problem]),
            "f_med": float(f_med[problem]),
        }
        for m in budgets:
            row[f"aocc_{m}d"] = (
                math.nan
                if m > run["budget_per_dimension"]
                else normalized_aocc(
                    run["curve_evaluations"],
                    run["curve_f"],
                    row["f_opt"],
                    row["f_med"],
                    m * run["dimension"],
                )
            )
        rows.append(row)
    columns = [*_RUN_COLUMNS, "f_opt", "f_opt_known", "f_med", *(f"aocc_{m}d" for m in budgets)]
    return pd.DataFrame(rows, columns=columns)


def normalized_aocc(
    curve_evaluations: Sequence[int],
    curve_f: Sequence[float],
    f_opt: float,
    f_med: float,
    budget: int,
) -> float:
    """AOCC of a best-so-far curve over its first ``budget`` evaluations.

    The curve is given as by [`bench_atlas.portfolio.optimize`][bench_atlas.portfolio.optimize] and expanded with
    [`bench_atlas.portfolio.best_so_far`][bench_atlas.portfolio.best_so_far], so a run that stopped early keeps its
    last best value. The error is normalized by ``f_med - f_opt``; NaN if that is not
    positive.
    """
    delta = f_med - f_opt
    if not math.isfinite(delta) or delta <= 0:
        return math.nan
    errors = (best_so_far(curve_evaluations, curve_f, budget) - f_opt) / delta
    return aocc(errors, PRECISION, 1.0)


def aocc(errors: Sequence[float], lower: float, upper: float) -> float:
    """Area over the convergence curve of ``errors``, one per evaluation, in [0, 1].

    Each evaluation contributes ``1 - clip((log10 e - log10 lower) / (log10 upper -
    log10 lower), 0, 1)``: 1 at or below ``lower``, 0 at or above ``upper``. Errors of
    zero or below count as ``lower``. NaN if there are no errors or any is NaN.
    """
    errors = np.asarray(errors, dtype=float)
    if errors.size == 0 or np.isnan(errors).any():
        return math.nan
    logs = np.log10(np.where(errors > 0, errors, lower))
    low, high = math.log10(lower), math.log10(upper)
    return float(np.mean(1 - np.clip((logs - low) / (high - low), 0, 1)))


def _f_opt(runs: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """``f*`` of every problem of ``runs``, and whether it is the known optimum."""
    problems = runs.groupby("problem")
    known = problems["f_opt"].first()
    return known.fillna(problems["best_f"].min()), known.notna()


def _f_med(runs: pd.DataFrame, ela: pd.DataFrame, points_per_dimension: int) -> pd.Series:
    """Median ELA ``f_med`` of every problem of ``runs``."""
    if ela.empty or not {"problem", "points_per_dimension", "f_med"} <= set(ela.columns):
        medians = pd.Series(dtype=float)
    else:
        sample = ela[ela["points_per_dimension"] == points_per_dimension]
        medians = sample.groupby("problem")["f_med"].median()
    missing = sorted(set(runs["problem"]) - set(medians.index))
    if missing:
        shown = ", ".join(missing[:5]) + (", ..." if len(missing) > 5 else "")
        raise ValueError(
            f"No ELA rows with points_per_dimension={points_per_dimension} for "
            f"{len(missing)} problems ({shown}); run `bench-atlas ela` first."
        )
    return medians
