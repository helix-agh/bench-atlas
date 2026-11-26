"""Command-line interface for bench-atlas."""

import enum
import functools
import socket
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Any

import joblib
import pandas as pd
import typer

from bench_atlas import __version__, benchmarks, executor, lon_pipeline, portfolio, store
from bench_atlas import aocc as aocc_module
from bench_atlas import ioh_logs as ioh_logs_module
from bench_atlas import lon as lon_module
from bench_atlas.ela import POINTS_PER_DIMENSION, ela_features
from bench_atlas.problems import Problem
from bench_atlas.timing import timing_features

app = typer.Typer(
    name="bench-atlas",
    help="Analyse and compare continuous black-box optimization benchmarks.",
    no_args_is_help=True,
)


def _not_implemented(command: str) -> None:
    typer.echo(f"`bench-atlas {command}` is not implemented yet.", err=True)
    raise typer.Exit(code=1)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the version and exit.",
    ),
) -> None:
    """Analyse and compare continuous black-box optimization benchmarks."""


def _check_jobs(value: int) -> int:
    if value == 0:
        raise typer.BadParameter("0 is not a valid number of jobs.")
    return value


class Suite(enum.StrEnum):
    bbob = "bbob"
    ma_bbob = "ma_bbob"
    gnbg_i = "gnbg_i"
    gnbg_ii = "gnbg_ii"
    gnbg_iii = "gnbg_iii"
    iohclustering = "iohclustering"
    portal = "portal"
    zigzag = "zigzag"
    cec2022 = "cec2022"
    random_trees = "random_trees"


# The atlas problem set: every command defaults to it, so features join per problem.
DIMENSIONS = [2, 5, 10, 20, 40]
INSTANCES = [1, 2, 3, 4, 5]
# Protocol decision: LON is too expensive above d = 10, so the LON atlas covers
# d <= 10 and instance 1 only.
LON_DIMENSIONS = [d for d in DIMENSIONS if d <= 10]
LON_INSTANCES = INSTANCES[:1]

Suites = Annotated[list[Suite], typer.Argument(help="Benchmark suites.")]
Dimensions = Annotated[
    list[int],
    typer.Option(
        "--dimension", "-d", help="Dimensions of BBOB, MA-BBOB, random trees, Zigzag and CEC 2022."
    ),
]
Instances = Annotated[
    list[int],
    typer.Option(
        "--instance",
        "-i",
        help="Instances of BBOB, MA-BBOB and random trees, parametrizations of Zigzag.",
    ),
]
Results = Annotated[
    Path, typer.Option("--results", "-o", help="Results folder, one Parquet file per analysis.")
]
Jobs = Annotated[
    int,
    typer.Option("--jobs", "-j", callback=_check_jobs, help="Parallel jobs; -1 uses all CPUs."),
]
Seed = Annotated[
    int, typer.Option("--seed", "-s", help="Global seed from which the task seeds are derived.")
]


@app.command()
def ela(
    suite: Suites,
    dimension: Dimensions = DIMENSIONS,
    instance: Instances = INSTANCES,
    points_per_dimension: int = typer.Option(
        POINTS_PER_DIMENSION,
        "--points-per-dimension",
        "-n",
        min=1,
        help="Sample size per dimension (250·d by default, see issue #41).",
    ),
    replicates: int = typer.Option(
        10, "--replicates", "-r", min=1, help="Independent Sobol samples per problem."
    ),
    results: Results = Path("results"),
    jobs: Jobs = 1,
    seed: Seed = 0,
) -> None:
    """Compute ELA features.

    Adds one row per problem and replicate (0 to ``--replicates`` - 1) to
    ``<results>/ela.parquet``, skipping rows that are already there, and prints the
    requested rows as CSV. Each replicate's Sobol sample seed is derived from
    ``--seed``, the dimension and the replicate, not from the problem or the number of
    replicates: replicate r of every problem of dimension d is evaluated on the same
    sample, so differences between problems are not sampling noise, and raising
    ``--replicates`` computes only the new ones. Rows hold the problem id,
    ``replicate``, ``seed``, ``points_per_dimension``, library versions, ``f_med``
    and all orivex features. GNBG suites have a fixed d = 30 and one instance.

    Stored rows are reused by problem, seed and settings only, not by the version
    of the analysis code: after changing the code, use a fresh ``--results`` folder
    or delete ``<results>/ela.parquet``. Tables written before the sample was shared
    across problems (one sample per problem id) are invalid.
    """
    table = _run(
        "ela",
        _problems(suite, dimension, instance),
        range(replicates),
        functools.partial(ela_features, points_per_dimension=points_per_dimension),
        {"points_per_dimension": points_per_dimension},
        results,
        jobs,
        seed,
        seed_per_dimension=True,
    )
    typer.echo(table.to_csv(index=False), nl=False)


def _run(
    analysis: str,
    problems: list[Problem],
    replicates: Sequence[int],
    compute: Callable[[Problem, int], Mapping[str, Any]],
    params: Mapping[str, Any],
    results: Path,
    jobs: int,
    seed: int = 0,
    variants: Sequence[Mapping[str, Any]] | None = None,
    seed_per_dimension: bool = False,
) -> pd.DataFrame:
    """[`executor.run`][bench_atlas.executor.run] with progress on stderr and a clean error for a locked table."""
    with _report_locked():
        return executor.run(
            analysis,
            problems,
            replicates,
            compute,
            params,
            results,
            n_jobs=jobs,
            global_seed=seed,
            log=functools.partial(typer.echo, err=True),
            variants=variants,
            seed_per_dimension=seed_per_dimension,
        )


@contextmanager
def _report_locked() -> Generator[None, None, None]:
    """Turn [`store.LockedError`][bench_atlas.store.LockedError] into a one-line message and exit code 1."""
    try:
        yield
    except store.LockedError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from None


def _problems(suite: list[Suite], dimension: list[int], instance: list[int]) -> list[Problem]:
    """Problems of the suites.

    BBOB, MA-BBOB, random trees and Zigzag take dimensions and instances (for Zigzag:
    parametrization IDs), CEC 2022 only dimensions; the other suites take neither.
    """
    problems = []
    for name in suite:
        if name in (Suite.bbob, Suite.ma_bbob, Suite.random_trees, Suite.zigzag):
            problems += getattr(benchmarks, name.value)(dimension, instance)
        elif name == Suite.cec2022:
            problems += benchmarks.cec2022(dimension)
        else:
            problems += getattr(benchmarks, name.value)()
    return problems


@app.command()
def lon(
    suite: Suites,
    dimension: Dimensions = LON_DIMENSIONS,
    instance: Instances = LON_INSTANCES,
    replicates: int = typer.Option(
        1, "--replicates", "-r", min=1, help="Basin-hopping samples per problem."
    ),
    runs: int = typer.Option(lon_module.N_RUNS, "--runs", min=1, help="Basin-hopping runs."),
    iter_no_change: int = typer.Option(
        lon_module.N_ITER_NO_CHANGE,
        "--iter-no-change",
        min=1,
        help="Non-improving perturbations that stop a run.",
    ),
    coordinate_precision: int = typer.Option(
        lon_module.COORDINATE_PRECISION,
        "--coordinate-precision",
        min=0,
        help="Decimals of the coordinates that identify an optimum.",
    ),
    fitness_precision: int = typer.Option(
        lon_module.FITNESS_PRECISION,
        "--fitness-precision",
        min=0,
        help="Decimals of the fitness of an optimum.",
    ),
    results: Results = Path("results"),
    jobs: Jobs = 1,
    seed: Seed = 0,
) -> None:
    """Build local optima networks and compute LON metrics.

    Every basin-hopping run is its own task, stored in ``<results>/lon_runs.parquet``
    with the full-precision optima it accepted, so ``--jobs`` spreads the runs of all
    problems over the workers. Then, for each problem and replicate, the CMLON is
    built at the requested precision, saved to ``<results>/cmlon/`` and its metrics
    are added to ``<results>/lon.parquet``; stored runs and rows are skipped, so
    another precision reuses the runs. Prints the requested ``lon`` rows as CSV: the
    problem id, ``replicate``, ``global_seed``, ``seed``, the settings, library versions, the
    evaluations used, the number of runs, the lonkit CMLON metrics and ``cmlon``, the
    graph name. GNBG suites have a fixed d = 30 and one instance.

    By default only d <= 10 and instance 1 (the LON atlas): LON is too expensive
    for the larger dimensions of the other analyses.
    """
    settings = lon_pipeline.Settings(
        runs=runs,
        iter_no_change=iter_no_change,
        coordinate_precision=coordinate_precision,
        fitness_precision=fitness_precision,
    )
    with _report_locked():
        table = lon_pipeline.run(
            _problems(suite, dimension, instance),
            range(replicates),
            settings,
            results,
            n_jobs=jobs,
            global_seed=seed,
            log=functools.partial(typer.echo, err=True),
        )
    typer.echo(table.to_csv(index=False), nl=False)


@app.command()
def timing(
    suite: Suites,
    dimension: Dimensions = DIMENSIONS,
    batch_size: list[int] = typer.Option(
        [1], "--batch-size", "-b", min=1, help="Points evaluated per call."
    ),
    results: Results = Path("results"),
    jobs: Jobs = 1,
) -> None:
    """Measure evaluation cost of benchmark problems.

    Adds one row per problem and batch size to ``<results>/timing.parquet``,
    skipping rows that are already there, and prints the requested rows as CSV.
    ``time_us`` is the median evaluation time per point in microseconds. BBOB and
    MA-BBOB use instance 1; GNBG suites have a fixed d = 30.

    Timings depend on the machine and on the load: with many parallel jobs the
    CPU runs at a lower all-core clock (about 3 % slower for a short run on a
    112-core node, about 11 % under sustained full load). The host and the
    effective number of jobs are therefore part of every row and of the cache
    key, so timings measured under different conditions are never mixed. After
    changing the timing code, use a fresh ``--results`` folder or delete
    ``<results>/timing.parquet``.
    """
    problems = _problems(suite, dimension, [1])
    params = {"n_jobs": joblib.effective_n_jobs(jobs), "host": socket.gethostname()}
    tables = [
        _run(
            "timing",
            problems,
            [0],
            functools.partial(timing_features, batch_size=size),
            {"batch_size": size, **params},
            results,
            jobs,
        )
        for size in batch_size
    ]
    order = {str(problem.id): i for i, problem in enumerate(problems)}
    table = pd.concat(tables, ignore_index=True).sort_values(
        "problem", key=lambda column: column.map(order), kind="stable"
    )
    typer.echo(table.to_csv(index=False), nl=False)


RUN_REPLICATES = 5
_LIST_COLUMNS = ["x_best", "curve_evaluations", "curve_f"]


def _check_optimizers(value: list[str]) -> list[str]:
    for name in value:
        try:
            portfolio.optimizer_class(name)
        except KeyError:
            raise typer.BadParameter(
                f"{name!r} is neither in the portfolio nor in Nevergrad's registry."
            ) from None
    if len(set(value)) != len(value):
        raise typer.BadParameter("An optimizer is given more than once.")
    return value


@app.command()
def run(
    suite: Suites,
    dimension: Dimensions = DIMENSIONS,
    instance: Instances = INSTANCES,
    replicates: int = typer.Option(
        RUN_REPLICATES, "--replicates", min=1, help="Runs per problem and optimizer."
    ),
    optimizer: list[str] = typer.Option(
        list(portfolio.PORTFOLIO),
        "--optimizer",
        "-a",
        callback=_check_optimizers,
        help="Optimizers: names in the portfolio or in Nevergrad's registry.",
    ),
    budget_per_dimension: int = typer.Option(
        portfolio.BUDGET_PER_DIMENSION,
        "--budget-per-dimension",
        "-B",
        min=1,
        help="Evaluation budget of one run, per dimension.",
    ),
    results: Results = Path("results"),
    jobs: Jobs = 1,
    seed: Seed = 0,
) -> None:
    """Run the optimizer portfolio on benchmark problems.

    Adds one row per problem, replicate (0 to ``--replicates`` - 1) and optimizer to
    ``<results>/runs.parquet``, skipping rows that are already there. Every optimizer
    gets ``--budget-per-dimension`` times d evaluations. The seed of a replicate is
    derived from ``--seed``, the problem id and the replicate, and is shared by all
    optimizers. Rows hold the problem id, ``replicate``, ``global_seed``, ``seed``, the
    budget, the optimizer, library versions, the problem's ``name`` and ``f_opt``,
    ``evaluations``, ``best_f``, its point ``x_best`` on [0, 1]^d, ``f_last`` (the
    value of the last evaluation) and the best-so-far curve (``curve_evaluations``,
    ``curve_f``: the evaluations at which the best value improved, and the best value
    after each). Prints the requested rows as CSV
    without ``x_best`` and the curves. GNBG suites have a fixed d = 30 and one
    instance.
    """
    table = _run(
        "runs",
        _problems(suite, dimension, instance),
        range(replicates),
        functools.partial(portfolio.optimize, budget_per_dimension=budget_per_dimension),
        {"budget_per_dimension": budget_per_dimension},
        results,
        jobs,
        seed,
        variants=[{"optimizer": name} for name in optimizer],
    )
    typer.echo(table.drop(columns=_LIST_COLUMNS).to_csv(index=False), nl=False)


@app.command()
def ioh_logs(
    results: Results = Path("results"),
    suite: list[Suite] | None = typer.Option(
        None, "--suite", help="Suites to log; all suites in the runs by default."
    ),
    optimizer: list[str] | None = typer.Option(
        None, "--optimizer", "-a", help="Optimizers to log; all by default."
    ),
    budget_per_dimension: int = typer.Option(
        portfolio.BUDGET_PER_DIMENSION,
        "--budget-per-dimension",
        "-B",
        min=1,
        help="Budget per dimension of the runs to log.",
    ),
    jobs: Jobs = 1,
) -> None:
    """Write IOH logs of the optimizer portfolio runs.

    Replays the runs stored in ``<results>/runs.parquet`` through IOH's Analyzer logger
    and writes ``<results>/ioh-logs/<suite>/<optimizer>/``, replacing earlier logs of
    the same suite and optimizer. IOHanalyzer and IOHinspector read this format; zip
    ``<results>/ioh-logs/<suite>`` to load all optimizers of a suite at once. Logged
    values are ``f - f_opt``, as in IOH's BBOB logs. Prints one CSV row per folder:
    ``suite``, ``optimizer``, ``runs`` and ``folder``.
    """
    runs = ioh_logs_module.select(
        store.load(results, "runs"),
        budget_per_dimension,
        [name.value for name in suite or []],
        optimizer,
    )
    if runs.empty:
        typer.echo(f"No runs to log in {store.path(results, 'runs')}.", err=True)
        raise typer.Exit(code=1)
    summary = ioh_logs_module.write_all(
        runs, results, n_jobs=jobs, log=functools.partial(typer.echo, err=True)
    )
    typer.echo(summary.to_csv(index=False), nl=False)


@app.command()
def aocc(
    results: Results = Path("results"),
    at: list[int] = typer.Option(
        list(aocc_module.BUDGETS_PER_DIMENSION),
        "--at",
        min=1,
        help="Budgets, per dimension, at which to compute AOCC.",
    ),
    ela_points_per_dimension: int = typer.Option(
        aocc_module.ELA_POINTS_PER_DIMENSION,
        "--ela-points-per-dimension",
        min=1,
        help="Sample size per dimension of the ELA rows that give f_med.",
    ),
) -> None:
    """Compute the normalized AOCC of the optimizer portfolio runs.

    Reads ``<results>/runs.parquet`` and the ``f_med`` of ``<results>/ela.parquet`` (run
    ``bench-atlas ela`` on the same problems first) and replaces
    ``<results>/aocc.parquet``. The error ``(f_best - f*) / (f_med - f*)`` is clipped to
    [1e-8, 1] on a log scale, with ``f*`` the known optimum or else the best value of
    any run of the problem, and ``f_med`` the median over the ELA replicates. Prints
    one CSV row per run: its id columns, ``f_opt`` (the ``f*`` used), ``f_opt_known``,
    ``f_med`` and ``aocc_<m>d`` for every ``--at m`` (NaN beyond the run's budget).
    """
    runs = store.load(results, "runs")
    if runs.empty:
        typer.echo(f"No runs in {store.path(results, 'runs')}.", err=True)
        raise typer.Exit(code=1)
    try:
        table = aocc_module.aocc_table(
            runs, store.load(results, "ela"), at, ela_points_per_dimension
        )
    except ValueError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from None
    with _report_locked(), store.lock(results, "aocc"):
        store.save(results, "aocc", table)
    typer.echo(table.to_csv(index=False), nl=False)


@app.command()
def metrics() -> None:
    """Compute benchmark comparison metrics."""
    _not_implemented("metrics")


@app.command()
def report() -> None:
    """Generate figures and tables from stored results."""
    _not_implemented("report")
