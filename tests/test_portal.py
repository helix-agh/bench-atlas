import pickle
from pathlib import Path

import numpy as np
import pytest

from bench_atlas.benchmarks._portal.fitness import fitness
from bench_atlas.benchmarks.portal import S50, portal
from bench_atlas.problems import Problem, ProblemId, to_unit

REFERENCE = np.load(Path(__file__).parent / "data" / "portal_reference.npz")


@pytest.fixture(scope="module")
def problems() -> list[Problem]:
    return portal()


def test_ids(problems: list[Problem]) -> None:
    assert len(problems) == len(S50) == 50
    assert len({problem.id for problem in problems}) == 50
    assert problems[0].id == ProblemId("portal", 1, 7, 30)
    assert problems[-1].id == ProblemId("portal", 4, 98, 30)
    assert problems[0].name.startswith("f01_r007_")
    assert [sum(p.id.function == f for p in problems) for f in (1, 2, 3, 4)] == [8, 11, 11, 20]


def test_vendored_code_matches_official_code(problems: list[Problem]) -> None:
    # Values of the official PORTAL code; see tests/data/make_portal_reference.py.
    for recipe, problem in zip(S50, problems, strict=True):
        x, y = REFERENCE[f"{recipe}_x"], REFERENCE[f"{recipe}_y"]
        values = [fitness(row, problem.function.instance) for row in x]
        # Stored values can differ in the last bits across platforms/BLAS builds.
        np.testing.assert_allclose(values, y, rtol=1e-12, atol=1e-12)


def test_matches_official_code(problems: list[Problem]) -> None:
    for recipe, problem in zip(S50, problems, strict=True):
        x, y = REFERENCE[f"{recipe}_x"], REFERENCE[f"{recipe}_y"]
        assert REFERENCE[f"{recipe}_f_opt"] == problem.f_opt == y[0]
        np.testing.assert_array_equal(to_unit(x[0], -100.0, 100.0), problem.x_opt)
        # Mapping x to [0, 1]^d and back changes it by rounding only.
        np.testing.assert_allclose(problem(to_unit(x, -100.0, 100.0)), y, rtol=1e-11)


def test_value_at_optimum(problems: list[Problem]) -> None:
    for problem in problems:
        assert problem(problem.x_opt[None, :])[0] == problem.f_opt


def test_value_does_not_depend_on_the_batch(problems: list[Problem]) -> None:
    u = np.random.default_rng(0).random((4, 30))
    for problem in problems[::5]:
        batch = problem(u)
        np.testing.assert_array_equal([problem(row[None, :])[0] for row in u], batch)


def test_picklable(problems: list[Problem]) -> None:
    u = np.random.default_rng(1).random((3, 30))
    for problem in problems[::7]:
        np.testing.assert_array_equal(pickle.loads(pickle.dumps(problem))(u), problem(u))
