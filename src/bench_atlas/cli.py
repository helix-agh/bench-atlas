"""Command-line interface for bench-atlas."""

import typer

from bench_atlas import __version__

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


@app.command()
def timing() -> None:
    """Measure evaluation cost of benchmark problems."""
    _not_implemented("timing")


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
