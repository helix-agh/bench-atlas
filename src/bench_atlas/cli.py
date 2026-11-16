"""Command-line interface for bench-atlas."""

import enum
import functools
import socket
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import typer

from bench_atlas import __version__, executor, store, suites
from bench_atlas.ela import ela_features
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


ELA_REPLICATES = 10


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


@app.command()
def ela(
    suite: list[Suite] = typer.Argument(..., help="Suites to analyse."),
    dimension: list[int] = typer.Option(
        [2, 5, 10, 20, 40], "--dimension", "-d", help="Dimensions of BBOB and MA-BBOB."
    ),
    instance: list[int] = typer.Option(
        [1, 2, 3, 4, 5], "--instance", "-i", help="Instances of BBOB and MA-BBOB."
    ),
    points_per_dimension: int = typer.Option(
        50, "--points-per-dimension", "-n", min=1, help="Sample size per dimension."
    ),
    results: Path = typer.Option(
        Path("results"), "--results", "-o", help="Results folder, one Parquet file per analysis."
    ),
    jobs: int = typer.Option(
        1, "--jobs", "-j", callback=_check_jobs, help="Parallel jobs; -1 uses all CPUs."
    ),
    seed: int = typer.Option(0, "--seed", "-s", help="Global seed of the sample seeds."),
) -> None:
    """Compute ELA features.

    Adds one row per problem and replicate (0-9) to ``<results>/ela.parquet``,
    skipping rows that are already there, and prints the requested rows as CSV.
    Each replicate's Sobol sample seed is derived from ``--seed``, the problem id
    and the replicate. Rows hold the problem id, ``replicate``, ``seed``,
    ``points_per_dimension``, library versions, ``f_med`` and all orivex
    features. GNBG suites have a fixed d = 30 and one instance.
    """
    table = _run(
        "ela",
        _problems(suite, dimension, instance),
        range(ELA_REPLICATES),
        functools.partial(ela_features, points_per_dimension=points_per_dimension),
        {"points_per_dimension": points_per_dimension},
        results,
        jobs,
        seed,
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
) -> pd.DataFrame:
    """:func:`executor.run` with progress on stderr and a clean error for a locked table."""
    try:
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
        )
    except store.LockedError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from None


def _problems(suite: list[Suite], dimension: list[int], instance: list[int]) -> list[Problem]:
    """Problems of the suites; GNBG suites ignore dimensions and instances."""
    problems = []
    for name in suite:
        if name in (Suite.bbob, Suite.ma_bbob):
            problems += getattr(suites, name.value)(dimension, instance)
        else:
            problems += getattr(suites, name.value)()
    return problems


@app.command()
def lon() -> None:
    """Build local optima networks and compute LON metrics."""
    _not_implemented("lon")


@app.command()
def timing(
    suite: list[Suite] = typer.Argument(..., help="Suites to measure."),
    dimension: list[int] = typer.Option(
        [2, 5, 10, 20, 40], "--dimension", "-d", help="Dimensions of BBOB and MA-BBOB."
    ),
    batch_size: list[int] = typer.Option(
        [1], "--batch-size", "-b", min=1, help="Points evaluated per call."
    ),
    results: Path = typer.Option(
        Path("results"), "--results", "-o", help="Results folder, one Parquet file per analysis."
    ),
    jobs: int = typer.Option(
        1, "--jobs", "-j", callback=_check_jobs, help="Parallel jobs; -1 uses all CPUs."
    ),
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
    key, so timings measured under different conditions are never mixed.
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


@app.command()
def run() -> None:
    """Run the optimizer portfolio on benchmark problems."""
    _not_implemented("run")


@app.command()
def metrics() -> None:
    """Compute benchmark comparison metrics."""
    _not_implemented("metrics")


@app.command()
def report() -> None:
    """Generate figures and tables from stored results."""
    _not_implemented("report")
