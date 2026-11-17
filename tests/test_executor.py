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
    assert (table["global_seed"] == 0).all()
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


class _SaveFailsOnce:
    def __init__(self, call: int) -> None:
        self.call = call
        self.calls = 0
        self.save = store.save

    def __call__(self, results: Path, analysis: str, table: pd.DataFrame) -> None:
        self.calls += 1
        if self.calls == self.call:
            raise SystemExit(143)
        self.save(results, analysis, table)


def test_save_retried_after_interrupt_adds_no_duplicates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(store, "save", _SaveFailsOnce(2))
    with pytest.raises(SystemExit):
        _run(tmp_path, _InterruptAfter(2), flush_seconds=0)
    monkeypatch.undo()
    saved = store.load(tmp_path, "test")
    assert len(saved) == 2
    assert not saved.duplicated(["problem", "seed"]).any()
    table = _run(tmp_path, _FailOn(set(saved["seed"])))
    assert len(table) == 6
    assert len(store.load(tmp_path, "test")) == 6


class _Count:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, problem: Problem, seed: int) -> dict[str, float]:
        self.calls += 1
        return _seed_feature(problem, seed)


def test_duplicate_tasks_run_once(tmp_path: Path) -> None:
    compute = _Count()
    table = executor.run("test", _problems() * 2, [0, 1, 1], compute, {"n": 1}, tmp_path)
    assert compute.calls == 6
    assert list(table["replicate"]) == [0, 1] * 3
    pd.testing.assert_frame_equal(table, store.load(tmp_path, "test"))


class _CallNumber:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, problem: Problem, seed: int) -> dict[str, float]:
        self.calls += 1
        return {"value": float(self.calls)}


@pytest.mark.parametrize(("runs", "n_tasks"), [(None, 6), (3, 18)])
def test_tasks_with_the_same_seed_are_distinct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runs: int | None, n_tasks: int
) -> None:
    """Tasks are identified by problem, replicate and run, not by their seed."""
    monkeypatch.setattr(executor, "task_seed", lambda *_args: 7)
    compute = _CallNumber()
    table = _run(tmp_path, compute, runs=runs)
    assert compute.calls == n_tasks
    assert (table["seed"] == 7).all()
    assert sorted(table["value"]) == list(range(1, n_tasks + 1))
    pd.testing.assert_frame_equal(table, store.load(tmp_path, "test"))
    messages: list[str] = []
    again = _run(tmp_path, _CallNumber(), runs=runs, log=messages.append)
    assert messages == [f"test: {n_tasks} cached, 0 to compute"]
    pd.testing.assert_frame_equal(table, again)


def test_other_global_seed_is_a_new_experiment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(executor, "task_seed", lambda *_args: 7)
    first = _run(tmp_path, _CallNumber())
    messages: list[str] = []
    other = _run(tmp_path, _CallNumber(), global_seed=1, log=messages.append)
    assert messages[0] == "test: 0 cached, 6 to compute"
    assert (other["global_seed"] == 1).all()
    assert len(store.load(tmp_path, "test")) == 12
    pd.testing.assert_frame_equal(first, _run(tmp_path, _CallNumber()))


def test_table_without_global_seed_is_recomputed(tmp_path: Path) -> None:
    first = _run(tmp_path)
    store.save(tmp_path, "test", first.drop(columns="global_seed"))
    messages: list[str] = []
    _run(tmp_path, log=messages.append)
    assert messages[0] == "test: 0 cached, 6 to compute"


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


def test_task_seed_of_runs() -> None:
    problem_id = ProblemId("bbob", 1, 1, 2)
    assert executor.task_seed(0, problem_id, 0, None) == executor.task_seed(0, problem_id, 0)
    seeds = [executor.task_seed(0, problem_id, r, run) for r in range(2) for run in range(50)]
    assert len(set(seeds)) == 100
    assert executor.task_seed(0, problem_id, 0) not in seeds


def test_runs_are_separate_tasks(tmp_path: Path) -> None:
    messages: list[str] = []
    table = _run(tmp_path, runs=3, log=messages.append)
    assert messages[0] == "test: 0 cached, 18 to compute"
    assert list(table["run"][:4]) == [0, 1, 2, 0]
    assert list(table["replicate"][:4]) == [0, 0, 0, 1]
    assert table["seed"].nunique() == 18
    again = _run(tmp_path, _FailOn(set(table["seed"])), runs=3)
    pd.testing.assert_frame_equal(table, again)


def _variant_feature(problem: Problem, seed: int, scale: int) -> dict[str, float]:
    return {"value": float(seed % 1000) * scale}


def test_variants_are_separate_tasks(tmp_path: Path) -> None:
    messages: list[str] = []
    variants = [{"scale": 1}, {"scale": 2}]
    table = _run(tmp_path, _variant_feature, variants=variants, log=messages.append)
    assert messages[0] == "test: 0 cached, 12 to compute"
    assert list(table["scale"][:4]) == [1, 2, 1, 2]
    assert list(table["replicate"][:4]) == [0, 0, 1, 1]
    assert (table["n"] == 1).all()
    assert table["seed"].nunique() == 6
    assert (table["seed"][::2].to_numpy() == table["seed"][1::2].to_numpy()).all()
    assert (table["value"][1::2].to_numpy() == 2 * table["value"][::2].to_numpy()).all()


def test_new_variant_computes_only_its_tasks(tmp_path: Path) -> None:
    first = _run(tmp_path, _variant_feature, variants=[{"scale": 1}])
    messages: list[str] = []
    table = _run(
        tmp_path, _variant_feature, variants=[{"scale": 2}, {"scale": 1}], log=messages.append
    )
    assert messages[0] == "test: 6 cached, 6 to compute"
    assert list(table["scale"][:2]) == [2, 1]
    pd.testing.assert_frame_equal(
        table[table["scale"] == 1].reset_index(drop=True), first, check_dtype=False
    )


@pytest.mark.parametrize(
    "variants",
    [
        [{"scale": 1}, {"other": 1}],
        [{"n": 2}],
        [{"scale": 1}, {"scale": 1}],
    ],
)
def test_invalid_variants_are_refused(tmp_path: Path, variants: list[dict[str, int]]) -> None:
    with pytest.raises(ValueError):
        _run(tmp_path, _variant_feature, variants=variants)
