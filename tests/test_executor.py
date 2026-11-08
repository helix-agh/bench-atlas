import functools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bench_atlas import executor, store
from bench_atlas.ela import ela_features
from bench_atlas.problems import Problem, ProblemId
from bench_atlas.suites import bbob


def _sphere(x: np.ndarray) -> np.ndarray:
    return np.sum((x - 0.5) ** 2, axis=1)


def _problems(n: int = 3) -> list[Problem]:
    return [Problem(ProblemId("test", f, 1, 2), _sphere) for f in range(1, n + 1)]


def _seed_feature(problem: Problem, seed: int) -> dict[str, float]:
    return {"value": float(seed % 1000)}


class _InterruptAfter:
    def __init__(self, n: int) -> None:
        self.n = n
        self.calls = 0

    def __call__(self, problem: Problem, seed: int) -> dict[str, float]:
        self.calls += 1
        if self.calls > self.n:
            raise KeyboardInterrupt
        return _seed_feature(problem, seed)


class _FailOn:
    def __init__(self, seeds: set[int]) -> None:
        self.seeds = seeds

    def __call__(self, problem: Problem, seed: int) -> dict[str, float]:
        assert seed not in self.seeds, "recomputed a stored task"
        return _seed_feature(problem, seed)


def _run(results: Path, compute=_seed_feature, params=None, **kwargs) -> pd.DataFrame:
    return executor.run(
        "test", _problems(), range(2), compute, params or {"n": 1}, results, **kwargs
    )


def test_task_seed_is_pinned_and_distinct() -> None:
    problem_id = ProblemId("bbob", 1, 1, 2)
    assert executor.task_seed(0, problem_id, 0) == 1860061545
    seeds = {
        executor.task_seed(g, ProblemId("bbob", f, 1, 2), r)
        for g in range(2)
        for f in range(1, 25)
        for r in range(10)
    }
    assert len(seeds) == 2 * 24 * 10


def test_rows_and_order(tmp_path: Path) -> None:
    table = _run(tmp_path)
    assert (
        list(table["problem"])
        == ["test_f1_i1_d2"] * 2 + ["test_f2_i1_d2"] * 2 + ["test_f3_i1_d2"] * 2
    )
    assert list(table["replicate"]) == [0, 1] * 3
    assert (table["n"] == 1).all()
    assert {"version_bench_atlas", "version_orivex", "version_lonkit", "version_ioh"} <= set(
        table.columns
    )
    pd.testing.assert_frame_equal(table, store.load(tmp_path, "test"))


def test_second_run_is_cached(tmp_path: Path) -> None:
    first = _run(tmp_path)
    messages: list[str] = []
    again = _run(tmp_path, _FailOn(set(first["seed"])), log=messages.append)
    pd.testing.assert_frame_equal(first, again)
    assert messages == ["test: 6 cached, 0 to compute"]


def test_other_params_are_recomputed(tmp_path: Path) -> None:
    first = _run(tmp_path, params={"n": 1})
    with pytest.raises(AssertionError, match="recomputed"):
        _run(tmp_path, _FailOn(set(first["seed"])), params={"n": 2})


def test_interrupted_run_resumes(tmp_path: Path) -> None:
    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path, _InterruptAfter(2), flush_seconds=0)
    saved = store.load(tmp_path, "test")
    assert len(saved) == 2
    assert not (tmp_path / "test.parquet.lock").exists()
    messages: list[str] = []
    table = _run(tmp_path, _FailOn(set(saved["seed"])), log=messages.append)
    assert len(table) == 6
    assert messages[0] == "test: 2 cached, 4 to compute"


def test_locked_table_is_refused(tmp_path: Path) -> None:
    with store.lock(tmp_path, "test"), pytest.raises(store.LockedError):
        _run(tmp_path)


def test_results_do_not_depend_on_n_jobs(tmp_path: Path) -> None:
    problems = bbob([2], [1])[:6]
    compute = functools.partial(ela_features, points_per_dimension=10)
    serial, parallel = (
        executor.run(
            "ela",
            problems,
            range(3),
            compute,
            {"points_per_dimension": 10},
            tmp_path / str(n_jobs),
            n_jobs=n_jobs,
        )
        for n_jobs in (1, 4)
    )
    pd.testing.assert_frame_equal(serial, parallel, check_exact=True)
