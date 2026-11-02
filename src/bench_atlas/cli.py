"""Command-line interface for bench-atlas."""

import enum

import pandas as pd
import typer

from bench_atlas import __version__, suites
from bench_atlas.ela import ela_features

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
) -> None:
    """Compute ELA features.

    Prints CSV: one row per problem and sample seed (0-9) with ``f_med`` and all
    orivex features. GNBG suites have a fixed d = 30 and one instance.
    """
    rows = []
    for name in suite:
        if name in (Suite.bbob, Suite.ma_bbob):
            problems = getattr(suites, name.value)(dimension, instance)
        else:
            problems = getattr(suites, name.value)()
        for problem in problems:
            for seed in range(10):
                rows.append(
                    {
                        "problem": str(problem.id),
                        "seed": seed,
                        **ela_features(problem, seed, points_per_dimension),
                    }
                )
    typer.echo(pd.DataFrame(rows).to_csv(index=False), nl=False)


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
) -> None:
    """Measure evaluation cost of benchmark problems.

    Prints CSV: problem id, batch size and the median evaluation time per point
    in microseconds. BBOB and MA-BBOB use instance 1; GNBG suites have a fixed
    d = 30.
    """
    from bench_atlas.timing import evaluation_time

    typer.echo("problem,batch_size,time_us")
    for name in suite:
        if name in (Suite.bbob, Suite.ma_bbob):
            problems = getattr(suites, name.value)(dimension, [1])
        else:
            problems = getattr(suites, name.value)()
        for problem in problems:
            for size in batch_size:
                time_us = evaluation_time(problem, batch_size=size) * 1e6
                typer.echo(f"{problem.id},{size},{time_us:.4g}")


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
