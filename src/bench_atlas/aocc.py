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

[`sbs_vbs_gap`][bench_atlas.aocc.sbs_vbs_gap] turns the AOCC table into one complementarity number per suite.
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
    "sbs_vbs_gap",
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


def sbs_vbs_gap(table: pd.DataFrame, resamples: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Gap between the virtual best and the single best optimizer of every suite.

    With ``s_p(a)`` the AOCC of optimizer ``a`` on problem ``p``, averaged over
    replicates, ``VBS = mean_p max_a s_p(a)``, ``SBS = max_a mean_p s_p(a)`` and
    ``gap = VBS - SBS`` in [0, 1]. A large gap means that different problems of the
    suite favour different optimizers; 0 means one optimizer is best on all of them.
    The gap depends on the portfolio, and it is biased low for small suites because
    the SBS is chosen on the problems it is scored on.

    Args:
        table: AOCC table as returned by [`aocc_table`][bench_atlas.aocc.aocc_table].
        resamples: Bootstrap resamples of the problems for the 95 % interval.
        seed: Seed of the bootstrap.

    Returns:
        One row per suite, dimension and budget: ``suite``, ``dimension``,
        ``budget_per_dimension``, ``problems``, ``sbs`` (its name), ``sbs_aocc``,
        ``vbs_aocc``, ``gap``, ``gap_low`` and ``gap_high``. Problems without an AOCC
        for every optimizer are left out.

    Raises:
        ValueError: ``table`` mixes runs of several budgets per dimension; filter it to
            one first, since a run's budget changes the optimizer's behaviour.
    """
    run_budgets = sorted(int(b) for b in table["budget_per_dimension"].unique())
    if len(run_budgets) > 1:
        raise ValueError(
            f"Runs with several budgets per dimension {run_budgets}; select one, e.g. "
            "table[table['budget_per_dimension'] == 1000]."
        )
    rng = np.random.default_rng(seed)
    budgets = [c for c in table.columns if c.startswith("aocc_") and c.endswith("d")]
    means = table.groupby(["suite", "dimension", "problem", "optimizer"])[budgets].mean()
    rows = []
    for column in budgets:
        wide = means[column].unstack("optimizer")
        for (suite, dimension), scores in wide.groupby(level=["suite", "dimension"]):
            scores = scores.dropna()
            if scores.empty:
                continue
            values = scores.to_numpy()
            optimizer_means = values.mean(axis=0)
            vbs, sbs = values.max(axis=1).mean(), optimizer_means.max()
            samples = values[rng.integers(len(values), size=(resamples, len(values)))]
            gaps = samples.max(axis=2).mean(axis=1) - samples.mean(axis=1).max(axis=1)
            low, high = np.percentile(gaps, [2.5, 97.5])
            rows.append(
                {
                    "suite": suite,
                    "dimension": dimension,
                    "budget_per_dimension": int(column[len("aocc_") : -1]),
                    "problems": len(values),
                    "sbs": scores.columns[optimizer_means.argmax()],
                    "sbs_aocc": float(sbs),
                    "vbs_aocc": float(vbs),
                    "gap": float(vbs - sbs),
                    "gap_low": float(low),
                    "gap_high": float(high),
                }
            )
    return pd.DataFrame(rows)


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
