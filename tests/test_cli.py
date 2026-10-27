import pytest
from typer.testing import CliRunner

from bench_atlas import __version__
from bench_atlas.cli import app

runner = CliRunner()

SUBCOMMANDS = ["ela", "lon", "timing", "run", "metrics", "report"]


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def test_help_lists_subcommands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in SUBCOMMANDS:
        assert command in result.stdout


@pytest.mark.parametrize("command", SUBCOMMANDS)
def test_placeholder_subcommands_exit_with_error(command: str) -> None:
    result = runner.invoke(app, [command])
    assert result.exit_code == 1
    assert "not implemented" in result.output


def test_core_dependencies_import() -> None:
    import ioh  # noqa: F401
    import joblib  # noqa: F401
    import lonkit  # noqa: F401
    import matplotlib  # noqa: F401
    import numpy  # noqa: F401
    import orivex  # noqa: F401
    import pandas  # noqa: F401
    import pyarrow  # noqa: F401
    import umap  # noqa: F401
