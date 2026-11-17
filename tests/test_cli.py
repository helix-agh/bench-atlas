import io
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from bench_atlas import __version__, store
from bench_atlas.cli import app

runner = CliRunner()

SUBCOMMANDS = ["ela", "lon", "timing", "run", "ioh-logs", "aocc", "metrics", "report"]


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def test_help_lists_subcommands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in SUBCOMMANDS:
        assert command in result.stdout


@pytest.mark.parametrize("command", ["metrics", "report"])
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


def test_lon_on_bbob(tmp_path: Path) -> None:
    args = ["lon", "bbob", "-d", "2", "--runs", "2", "--iter-no-change", "5", "-o", str(tmp_path)]
    result = runner.invoke(app, args)
    assert result.exit_code == 0
    assert "lon_runs: 0 cached, 48 to compute" in result.stderr
    table = pd.read_csv(io.StringIO(result.stdout))
    assert len(table) == 24
    assert table["problem"].iloc[0] == "bbob_f1_i1_d2"
    assert (table["n_runs"] == 2).all()
    assert (table["coordinate_precision"] == 4).all()
    assert not table.drop(columns="cmlon").isna().any().any()
    assert len(pd.read_parquet(tmp_path / "lon_runs.parquet")) == 48
    assert len(pd.read_parquet(tmp_path / "lon.parquet")) == 24
    assert store.load_cmlon(tmp_path, table["cmlon"].iloc[0]).n_vertices >= 1
    again = runner.invoke(app, args)
    assert "lon_runs: 48 cached, 0 to compute" in again.stderr
    assert "lon: 24 cached, 0 to compute" in again.stderr
    assert again.stdout == result.stdout


def test_lon_other_precision_reuses_the_runs(tmp_path: Path) -> None:
    args = ["lon", "bbob", "-d", "2", "--runs", "2", "--iter-no-change", "5", "-o", str(tmp_path)]
    runner.invoke(app, args)
    result = runner.invoke(app, [*args, "--coordinate-precision", "2"])
    assert result.exit_code == 0
    assert "lon_runs: 48 cached, 0 to compute" in result.stderr
    assert "lon: 0 cached, 24 to compute" in result.stderr
    assert len(list((tmp_path / "cmlon").glob("*.graphml"))) == 48


def test_lon_replicates_and_jobs(tmp_path: Path) -> None:
    base = ["lon", "bbob", "-d", "2", "-r", "2", "--runs", "3", "--iter-no-change", "5"]
    serial = runner.invoke(app, [*base, "-o", str(tmp_path / "serial")])
    assert serial.exit_code == 0
    table = pd.read_csv(io.StringIO(serial.stdout))
    assert len(table) == 24 * 2
    assert list(table["replicate"][:2]) == [0, 1]
    assert (table["n_runs"] == 3).all()
    parallel = runner.invoke(app, [*base, "-o", str(tmp_path / "parallel"), "-j", "2"])
    assert parallel.stdout == serial.stdout


def test_lon_rejects_zero_runs(tmp_path: Path) -> None:
    result = runner.invoke(app, ["lon", "bbob", "-d", "2", "--runs", "0", "-o", str(tmp_path)])
    assert result.exit_code == 2


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


def test_run(tmp_path: Path) -> None:
    args = ["run", "bbob", "-d", "2", "-i", "1", "--replicates", "2", "-B", "10"]
    args += ["-a", "CMAstd", "-a", "RandomSearch", "-o", str(tmp_path)]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    table = pd.read_csv(io.StringIO(result.stdout))
    assert len(table) == 24 * 2 * 2
    assert list(table["optimizer"][:4]) == ["CMAstd", "RandomSearch"] * 2
    assert list(table["replicate"][:4]) == [0, 0, 1, 1]
    assert (table["evaluations"] == 20).all()
    assert (table["budget_per_dimension"] == 10).all()
    assert "curve_f" not in table
    assert "x_best" not in table
    stored = pd.read_parquet(tmp_path / "runs.parquet")
    assert len(stored) == 24 * 2 * 2
    assert (stored["curve_f"].map(len) > 0).all()
    assert not table.isna().any().any()
    again = runner.invoke(app, args)
    assert "96 cached, 0 to compute" in again.stderr
    assert again.stdout == result.stdout


def test_run_rejects_unknown_optimizer(tmp_path: Path) -> None:
    result = runner.invoke(app, ["run", "bbob", "-d", "2", "-a", "Nope", "-o", str(tmp_path)])
    assert result.exit_code == 2


def test_ioh_logs(tmp_path: Path) -> None:
    run_args = ["run", "bbob", "-d", "2", "-i", "1", "--replicates", "2", "-B", "10"]
    run_args += ["-a", "CMAstd", "-a", "PSO", "-o", str(tmp_path)]
    assert runner.invoke(app, run_args).exit_code == 0
    result = runner.invoke(app, ["ioh-logs", "-o", str(tmp_path), "-B", "10", "-a", "PSO"])
    assert result.exit_code == 0, result.output
    summary = pd.read_csv(io.StringIO(result.stdout))
    assert summary.to_dict("records") == [
        {
            "suite": "bbob",
            "optimizer": "PSO",
            "runs": 48,
            "folder": str(tmp_path / "ioh-logs" / "bbob" / "PSO"),
        }
    ]
    assert len(list((tmp_path / "ioh-logs" / "bbob" / "PSO").glob("*.json"))) == 24
    assert not (tmp_path / "ioh-logs" / "bbob" / "CMAstd").exists()


def test_ioh_logs_without_runs(tmp_path: Path) -> None:
    result = runner.invoke(app, ["ioh-logs", "-o", str(tmp_path)])
    assert result.exit_code == 1
    assert "No runs to log" in result.stderr


def test_aocc(tmp_path: Path) -> None:
    out = ["-o", str(tmp_path)]
    run_args = ["run", "bbob", "-d", "2", "-i", "1", "--replicates", "2", "-B", "10", "-a", "PSO"]
    assert runner.invoke(app, [*run_args, *out]).exit_code == 0
    assert runner.invoke(app, ["aocc", *out]).exit_code == 1
    ela_args = ["ela", "bbob", "-d", "2", "-i", "1", "-n", "10"]
    assert runner.invoke(app, [*ela_args, *out]).exit_code == 0
    args = ["aocc", "--at", "5", "--at", "10", "--at", "20", "--ela-points-per-dimension", "10"]
    result = runner.invoke(app, [*args, *out])
    assert result.exit_code == 0, result.output
    table = pd.read_csv(io.StringIO(result.stdout))
    assert len(table) == 24 * 2
    assert table["f_opt_known"].all()
    for column in ("aocc_5d", "aocc_10d"):
        assert table[column].between(0, 1).all()
    assert table["aocc_20d"].isna().all()
    stored = pd.read_parquet(tmp_path / "aocc.parquet")
    assert len(stored) == 24 * 2
    repeated = ["aocc", "--at", "5", "--at", "5", "--ela-points-per-dimension", "10", *out]
    result = runner.invoke(app, repeated)
    assert result.exit_code == 0, result.output
    assert list(pd.read_csv(io.StringIO(result.stdout)).columns).count("aocc_5d") == 1
    assert list(pd.read_parquet(tmp_path / "aocc.parquet").columns).count("aocc_5d") == 1


def test_aocc_without_runs(tmp_path: Path) -> None:
    result = runner.invoke(app, ["aocc", "-o", str(tmp_path)])
    assert result.exit_code == 1
    assert "No runs" in result.stderr
