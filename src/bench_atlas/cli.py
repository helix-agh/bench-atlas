"""Command-line interface for bench-atlas."""

import enum

import typer

from bench_atlas import __version__, suites

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


@app.command()
def ela() -> None:
    """Compute ELA features."""
    _not_implemented("ela")


@app.command()
def lon() -> None:
    """Build local optima networks and compute LON metrics."""
    _not_implemented("lon")


class Suite(str, enum.Enum):
    bbob = "bbob"
    ma_bbob = "ma_bbob"
    gnbg_i = "gnbg_i"
    gnbg_ii = "gnbg_ii"
    gnbg_iii = "gnbg_iii"


@app.command()
def timing(
    suite: list[Suite] = typer.Argument(..., help="Suites to measure."),
    dimension: list[int] = typer.Option(
        [2, 5, 10, 20, 40], "--dimension", "-d", help="Dimensions of BBOB and MA-BBOB."
    ),
) -> None:
    """Measure evaluation cost of benchmark problems.

    Prints CSV: problem id and the median time of one evaluation in microseconds.
    BBOB and MA-BBOB use instance 1; GNBG suites have a fixed d = 30.
    """
    from bench_atlas.timing import evaluation_time

    typer.echo("problem,time_us")
    for name in suite:
        if name in (Suite.bbob, Suite.ma_bbob):
            problems = getattr(suites, name.value)(dimension, [1])
        else:
            problems = getattr(suites, name.value)()
        for problem in problems:
            typer.echo(f"{problem.id},{evaluation_time(problem) * 1e6:.2f}")


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
