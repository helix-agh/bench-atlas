import pickle

import numpy as np
import pandas as pd
import pytest

from bench_atlas.problems import Problem, ProblemId, from_unit

dispersion_bench = pytest.importorskip("dispersion_bench")

from bench_atlas.dispersion import dispersion  # noqa: E402

PENALTY = 1e6


@pytest.fixture(scope="module")
def problems() -> list[Problem]:
    return dispersion()


def test_ids(problems: list[Problem]) -> None:
    dimensions = [2] * 8 + [4] * 8 + [6] * 8
    assert [problem.id for problem in problems] == [
        ProblemId("dispersion", function, 1, d) for function, d in enumerate(dimensions, start=1)
    ]
    assert problems[8].name == "dispersion_p09"


def test_optimum(problems: list[Problem]) -> None:
    assert all(problem.f_opt == 0.0 and problem.x_opt is None for problem in problems)


def test_matches_native(problems: list[Problem]) -> None:
    rng = np.random.default_rng(0)
    for problem in problems:
        native = dispersion_bench.Problem(problem.id.function)
        u = rng.random((3, problem.dimension))
        x = from_unit(u, native.lower_bounds, native.upper_bounds)
        expected = np.array([native(point) for point in x.tolist()])
        np.testing.assert_array_equal(problem(u), expected)


def test_box_corners_are_inside(problems: list[Problem]) -> None:
    # The corners of [0, 1]^d must map into the native bounds, not onto the penalty.
    for problem in problems[::8]:
        corners = np.stack([np.zeros(problem.dimension), np.ones(problem.dimension)])
        assert np.all(problem(corners) < PENALTY)


def test_pickle_round_trip(problems: list[Problem]) -> None:
    # The restored problem is evaluated first, so it cannot reuse the original's state.
    u = np.random.default_rng(1).random((2, 4))
    problem = problems[9]
    restored = pickle.loads(pickle.dumps(problem))
    assert restored.id == problem.id
    assert restored.name == problem.name
    np.testing.assert_array_equal(restored(u), problem(u))


def _values(problem: Problem, seed: int) -> dict[str, float]:
    u = np.random.default_rng(seed).random((2, problem.dimension))
    return {"f0": float(problem(u)[0]), "f1": float(problem(u)[1])}


def test_parallel_workers_rebuild_the_problems(problems: list[Problem], tmp_path) -> None:
    # Problems are pickled into loky worker processes and rebuilt there from their index.
    from bench_atlas import executor

    chosen = [problems[0], problems[9]]
    serial, parallel = (
        executor.run("check", chosen, [0], _values, {}, tmp_path / str(n_jobs), n_jobs=n_jobs)
        for n_jobs in (1, 2)
    )
    pd.testing.assert_frame_equal(serial, parallel, check_exact=True)
