import io
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from bench_atlas import __version__, store
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


def test_timing(tmp_path: Path) -> None:
    args = ["timing", "bbob", "-d", "2", "-b", "1", "-b", "10", "-o", str(tmp_path)]
    result = runner.invoke(app, args)
    assert result.exit_code == 0
    table = pd.read_csv(io.StringIO(result.stdout))
    assert len(table) == 24 * 2
    assert list(table["problem"][:3]) == ["bbob_f1_i1_d2", "bbob_f1_i1_d2", "bbob_f2_i1_d2"]
    assert list(table["batch_size"][:3]) == [1, 10, 1]
    assert (table["time_us"] > 0).all()
    assert (table["n_jobs"] == 1).all()
    assert table["host"].nunique() == 1
    assert len(pd.read_parquet(tmp_path / "timing.parquet")) == 24 * 2


def test_timing_cache_separates_job_counts(tmp_path: Path) -> None:
    args = ["timing", "bbob", "-d", "2", "-o", str(tmp_path)]
    first = runner.invoke(app, args)
    again = runner.invoke(app, args)
    assert "24 cached, 0 to compute" in again.stderr
    assert again.stdout == first.stdout
    parallel = runner.invoke(app, [*args, "-j", "2"])
    assert parallel.exit_code == 0
    assert "0 cached, 24 to compute" in parallel.stderr
    assert len(pd.read_parquet(tmp_path / "timing.parquet")) == 24 * 2


def test_ela_on_bbob(tmp_path: Path) -> None:
    result = runner.invoke(app, ["ela", "bbob", "-d", "2", "-n", "30", "-o", str(tmp_path)])
    assert result.exit_code == 0
    table = pd.read_csv(io.StringIO(result.stdout))
    assert len(table) == 24 * 5 * 10
    assert table["problem"].iloc[0] == "bbob_f1_i1_d2"
    assert not table.isna().any().any()


def test_ela_instances(tmp_path: Path) -> None:
    result = runner.invoke(app, ["ela", "bbob", "-d", "2", "-i", "3", "-o", str(tmp_path)])
    assert result.exit_code == 0
    table = pd.read_csv(io.StringIO(result.stdout))
    assert set(table["problem"].str.split("_").str[2]) == {"i3"}


def test_ela_rejects_empty_sample(tmp_path: Path) -> None:
    result = runner.invoke(app, ["ela", "bbob", "-d", "2", "-n", "0", "-o", str(tmp_path)])
    assert result.exit_code == 2
    assert not any(tmp_path.iterdir())


def test_ela_rejects_zero_jobs(tmp_path: Path) -> None:
    result = runner.invoke(app, ["ela", "bbob", "-d", "2", "-j", "0", "-o", str(tmp_path)])
    assert result.exit_code == 2


def test_ela_stores_and_reuses_results(tmp_path: Path) -> None:
    args = ["ela", "bbob", "-d", "2", "-i", "1", "-n", "10", "-o", str(tmp_path)]
    first = runner.invoke(app, args)
    assert first.exit_code == 0
    stored = pd.read_parquet(tmp_path / "ela.parquet")
    assert len(stored) == 24 * 10
    again = runner.invoke(app, args)
    assert again.exit_code == 0
    assert "240 cached, 0 to compute" in again.stderr
    assert again.stdout == first.stdout
    parallel = runner.invoke(app, [*args[:-1], str(tmp_path / "parallel"), "-j", "2"])
    assert parallel.exit_code == 0
    assert parallel.stdout == first.stdout


def test_ela_runs_repeated_problems_once(tmp_path: Path) -> None:
    args = ["ela", "bbob", "-d", "2", "-d", "2", "-i", "1", "-n", "10", "-o", str(tmp_path)]
    result = runner.invoke(app, args)
    assert result.exit_code == 0
    assert "0 cached, 240 to compute" in result.stderr
    assert len(pd.read_csv(io.StringIO(result.stdout))) == 24 * 10
    assert len(pd.read_parquet(tmp_path / "ela.parquet")) == 24 * 10


def test_ela_replicates_extend_the_cache(tmp_path: Path) -> None:
    base = ["ela", "bbob", "-d", "2", "-i", "1", "-n", "10", "-o", str(tmp_path)]
    two = runner.invoke(app, [*base, "-r", "2"])
    assert two.exit_code == 0
    three = runner.invoke(app, [*base, "-r", "3"])
    assert "48 cached, 24 to compute" in three.stderr
    one = runner.invoke(app, [*base, "-r", "1"])
    assert "24 cached, 0 to compute" in one.stderr
    table = pd.read_csv(io.StringIO(one.stdout))
    assert len(table) == 24
    assert set(table["replicate"]) == {0}
    assert runner.invoke(app, [*base, "-r", "0"]).exit_code == 2


def test_ela_refuses_a_locked_table(tmp_path: Path) -> None:
    with store.lock(tmp_path, "ela"):
        result = runner.invoke(app, ["ela", "bbob", "-d", "2", "-o", str(tmp_path)])
    assert result.exit_code == 1
    assert "ela.parquet.lock is held by" in result.stderr
    assert "Traceback" not in result.output


def test_ela_seed_changes_samples(tmp_path: Path) -> None:
    base = ["ela", "bbob", "-d", "2", "-i", "1", "-n", "10", "-o", str(tmp_path)]
    seeds = [
        pd.read_csv(io.StringIO(runner.invoke(app, [*base, "-s", s]).stdout))["seed"]
        for s in ("0", "1")
    ]
    assert not seeds[0].isin(seeds[1]).any()
    assert len(pd.read_parquet(tmp_path / "ela.parquet")) == 2 * 24 * 10


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
