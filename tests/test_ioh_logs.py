import json
from pathlib import Path

import ioh
import numpy as np
import pandas as pd
import pytest

from bench_atlas import ioh_logs, portfolio, store
from bench_atlas.problems import Problem, ProblemId
from bench_atlas.suites import bbob
from tests.test_portfolio import _StopAfter


def _row(problem: Problem, optimizer: str, replicate: int, seed: int, budget: int) -> dict:
    return {
        "problem": str(problem.id),
        "suite": problem.id.suite,
        "function": problem.id.function,
        "instance": problem.id.instance,
        "dimension": problem.id.dimension,
        "replicate": replicate,
        "seed": seed,
        "budget_per_dimension": budget,
        "optimizer": optimizer,
        "version_nevergrad": "1.0.12",
        **portfolio.optimize(problem, seed, optimizer, budget_per_dimension=budget),
    }


def _live(
    problem: Problem, optimizer: str, seeds: list[int], root: Path, budget: int = 30
) -> pd.DataFrame:
    inner = problem.function.base._problem
    logger = ioh.logger.Analyzer(root=str(root), folder_name="live", algorithm_name=optimizer)
    inner.attach_logger(logger)
    rows = []
    for replicate, seed in enumerate(seeds):
        rows.append(_row(problem, optimizer, replicate, seed, budget))
        inner.reset()
    inner.detach_logger()
    logger.close()
    return pd.DataFrame(rows)


def _files(folder: Path) -> dict[str, Path]:
    return {str(file.relative_to(folder)): file for file in folder.rglob("*") if file.is_file()}


def _without_run_details(info: dict) -> dict:
    info = {key: value for key, value in info.items() if key not in ("algorithm", "run_attributes")}
    for scenario in info["scenarios"]:
        for run in scenario["runs"]:
            for key in ("replicate", "seed"):
                run.pop(key, None)
            run["best"].pop("x")
            run["best"].pop("y")
    return info


def _best_y(info: dict) -> list[float]:
    return [run["best"]["y"] for scenario in info["scenarios"] for run in scenario["runs"]]


@pytest.mark.parametrize("optimizer", list(portfolio.PORTFOLIO))
@pytest.mark.parametrize("function", [1, 8, 15, 21])
def test_replay_equals_live_ioh_log(tmp_path: Path, function: int, optimizer: str) -> None:
    _assert_replay_equals_live(tmp_path, bbob([2], [3])[function - 1], optimizer, 30)


@pytest.mark.parametrize(("function", "optimizer"), [(8, "CMAstd"), (2, "LBFGSB")])
def test_replay_equals_live_ioh_log_after_the_target(
    tmp_path: Path, function: int, optimizer: str
) -> None:
    problem = bbob([2], [1])[function - 1]
    runs = _assert_replay_equals_live(tmp_path, problem, optimizer, 1000)
    assert (runs["best_f"] - runs["f_opt"] < 1e-8).all()


def _assert_replay_equals_live(
    tmp_path: Path, problem: Problem, optimizer: str, budget: int
) -> pd.DataFrame:
    runs = _live(problem, optimizer, [11, 12], tmp_path, budget)
    ioh_logs.write(runs, tmp_path / "replay")
    live, replay = _files(tmp_path / "live"), _files(tmp_path / "replay")
    assert live.keys() == replay.keys()
    for name in live:
        if name.endswith(".dat"):
            assert live[name].read_bytes() == replay[name].read_bytes()
        else:
            live_info, replay_info = (
                json.loads(files[name].read_text()) for files in (live, replay)
            )
            assert _best_y(replay_info) == pytest.approx(_best_y(live_info), rel=1e-12)
            assert _without_run_details(live_info) == _without_run_details(replay_info)
    return runs


def test_json_metadata(tmp_path: Path) -> None:
    problem = bbob([2], [1])[0]
    runs = pd.DataFrame([_row(problem, "PSO", r, 100 + r, 20) for r in range(3)])
    assert ioh_logs.write(runs, tmp_path / "PSO") == 3
    info = json.loads((tmp_path / "PSO" / "IOHprofiler_f1_Sphere.json").read_text())
    assert info["function_id"] == 1
    assert info["algorithm"] == {"name": "PSO", "info": "nevergrad 1.0.12, budget_per_dimension 20"}
    [scenario] = info["scenarios"]
    assert scenario["dimension"] == 2
    assert [run["seed"] for run in scenario["runs"]] == [100, 101, 102]
    assert [run["replicate"] for run in scenario["runs"]] == [0, 1, 2]
    for run, row in zip(scenario["runs"], runs.itertuples(), strict=True):
        assert run["evals"] == 40
        assert run["best"]["y"] == pytest.approx(row.best_f - row.f_opt)
        assert run["best"]["x"] == pytest.approx(row.x_best)


def test_dat_lines_follow_the_curve(tmp_path: Path) -> None:
    problem = Problem(ProblemId("test", 1, 1, 2), lambda x: np.sum(x**2, axis=1), name="square")
    runs = pd.DataFrame([_row(problem, "RandomSearch", r, r, 50) for r in range(2)])
    ioh_logs.write(runs, tmp_path / "out")
    text = (tmp_path / "out" / "data_f1_square" / "IOHprofiler_f1_DIM2.dat").read_text()
    blocks = [block.strip().splitlines() for block in text.split("evaluations raw_y")[1:]]
    for lines, row in zip(blocks, runs.itertuples(), strict=True):
        evaluations = [int(line.split()[0]) for line in lines]
        values = [float(line.split()[1]) for line in lines]
        improvements = list(row.curve_evaluations)
        assert evaluations[: len(improvements)] == improvements
        np.testing.assert_allclose(values[: len(improvements)], row.curve_f, rtol=1e-9, atol=1e-10)
        if improvements[-1] != 100:
            assert evaluations[-1] == 100
            assert values[-1] == pytest.approx(row.f_last)


def test_early_stop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(portfolio.PORTFOLIO, "Stop", _StopAfter(7))
    problem = bbob([2], [1])[0]
    runs = pd.DataFrame([_row(problem, "Stop", 0, 0, 50)])
    ioh_logs.write(runs, tmp_path / "Stop")
    info = json.loads((tmp_path / "Stop" / "IOHprofiler_f1_Sphere.json").read_text())
    assert info["scenarios"][0]["runs"][0]["evals"] == 7


def test_write_replaces_the_folder(tmp_path: Path) -> None:
    problem = bbob([2], [1])[0]
    runs = pd.DataFrame([_row(problem, "PSO", r, r, 10) for r in range(2)])
    ioh_logs.write(runs, tmp_path / "PSO")
    ioh_logs.write(runs.iloc[:1], tmp_path / "PSO")
    assert [path.name for path in tmp_path.iterdir()] == ["PSO"]
    info = json.loads((tmp_path / "PSO" / "IOHprofiler_f1_Sphere.json").read_text())
    assert len(info["scenarios"][0]["runs"]) == 1


def test_write_refuses_mixed_optimizers(tmp_path: Path) -> None:
    problem = bbob([2], [1])[0]
    runs = pd.DataFrame([_row(problem, name, 0, 0, 10) for name in ("PSO", "CMAstd")])
    with pytest.raises(ValueError, match="optimizer"):
        ioh_logs.write(runs, tmp_path / "out")


def test_write_all(tmp_path: Path) -> None:
    problems = bbob([2, 3], [1])[:4]
    runs = pd.DataFrame([_row(p, name, 0, 0, 10) for p in problems for name in ("PSO", "CMAstd")])
    summary = ioh_logs.write_all(runs, tmp_path, n_jobs=2)
    assert list(summary["optimizer"]) == ["CMAstd", "PSO"]
    assert list(summary["runs"]) == [4, 4]
    for optimizer in ("CMAstd", "PSO"):
        folder = store.ioh_logs_path(tmp_path, "bbob", optimizer)
        assert folder == tmp_path / "ioh-logs" / "bbob" / optimizer
        info = json.loads((folder / "IOHprofiler_f1_Sphere.json").read_text())
        assert [scenario["dimension"] for scenario in info["scenarios"]] == [2, 3]
