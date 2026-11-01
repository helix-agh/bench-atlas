import io

import pandas as pd
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


@pytest.mark.parametrize("command", [c for c in SUBCOMMANDS if c not in ("ela", "timing")])
def test_placeholder_subcommands_exit_with_error(command: str) -> None:
    result = runner.invoke(app, [command])
    assert result.exit_code == 1
    assert "not implemented" in result.output


def test_timing() -> None:
    result = runner.invoke(app, ["timing", "bbob", "-d", "2", "-b", "1", "-b", "10"])
    assert result.exit_code == 0
    lines = result.stdout.splitlines()
    assert lines[0] == "problem,batch_size,time_us"
    assert len(lines) == 1 + 24 * 2
    assert lines[1].startswith("bbob_f1_i1_d2,1,")
    assert lines[2].startswith("bbob_f1_i1_d2,10,")


def test_ela_on_bbob() -> None:
    result = runner.invoke(app, ["ela", "bbob", "-d", "2"])
    assert result.exit_code == 0
    table = pd.read_csv(io.StringIO(result.stdout))
    assert len(table) == 24 * 5 * 10
    assert table["problem"].iloc[0] == "bbob_f1_i1_d2"
    assert not table.isna().any().any()


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
