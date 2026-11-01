import pickle
from pathlib import Path

import ioh
import numpy as np
import pytest

from bench_atlas.problems import Problem, ProblemId, from_unit
from bench_atlas.suites import bbob, gnbg_ii, gnbg_iii, ma_bbob

DIMENSIONS = [2, 5]
INSTANCES = [1, 2, 3]


@pytest.fixture(scope="module", params=["bbob", "ma_bbob"])
def suite(request) -> list[Problem]:
    return {"bbob": bbob, "ma_bbob": ma_bbob}[request.param](DIMENSIONS, INSTANCES)


def test_sizes() -> None:
    assert len(bbob(DIMENSIONS, INSTANCES)) == 24 * len(INSTANCES) * len(DIMENSIONS)
    assert len(ma_bbob(DIMENSIONS, INSTANCES)) == len(INSTANCES) * len(DIMENSIONS)


def test_unique_ids(suite: list[Problem]) -> None:
    assert len({problem.id for problem in suite}) == len(suite)


def test_value_at_optimum(suite: list[Problem]) -> None:
    for problem in suite:
        assert problem(problem.x_opt[None, :])[0] == pytest.approx(problem.f_opt, abs=1e-8)


def test_picklable(suite: list[Problem]) -> None:
    for problem in (suite[0], suite[-1]):
        x = np.random.default_rng(0).random((5, problem.dimension))
        clone = pickle.loads(pickle.dumps(problem))
        np.testing.assert_array_equal(clone(x), problem(x))


def test_ma_bbob_identical_to_ioh() -> None:
    (problem,) = ma_bbob([5], [7])
    raw = ioh.problem.ManyAffine(7, 5)
    points = np.random.default_rng(0).random((20, 5))
    mapped = from_unit(points, raw.bounds.lb, raw.bounds.ub)
    assert problem.id == ProblemId("ma_bbob", 0, 7, 5)
    assert problem(points).tolist() == [raw(x) for x in mapped]


GNBG = {"gnbg_ii": (gnbg_ii, 24), "gnbg_iii": (gnbg_iii, 22)}


@pytest.fixture(scope="module", params=list(GNBG))
def gnbg(request) -> tuple[str, list[Problem]]:
    load, _ = GNBG[request.param]
    return request.param, load()


class TestGNBG:
    def test_ids(self, gnbg: tuple[str, list[Problem]]) -> None:
        # GNBG-III: only the static problems F1-F22.
        suite, problems = gnbg
        assert [problem.id for problem in problems] == [
            ProblemId(suite, function, 1, 30) for function in range(1, GNBG[suite][1] + 1)
        ]

    def test_matches_official_code(self, gnbg: tuple[str, list[Problem]]) -> None:
        # Reference values from the official GNBG code, in original coordinates;
        # see tests/data/make_gnbg_reference.py.
        suite, problems = gnbg
        reference = np.load(Path(__file__).parent / "data" / f"{suite}_reference.npz")
        for problem in problems:
            x = reference[f"f{problem.id.function}_x"]
            y = reference[f"f{problem.id.function}_y"]
            values = problem.function.base(x)
            np.testing.assert_allclose(values, y, rtol=1e-12)
            assert values[0] == y[0] == problem.f_opt  # the first point is the optimum

    def test_picklable(self, gnbg: tuple[str, list[Problem]]) -> None:
        x = np.random.default_rng(0).random((5, 30))
        for problem in gnbg[1]:
            np.testing.assert_array_equal(pickle.loads(pickle.dumps(problem))(x), problem(x))
