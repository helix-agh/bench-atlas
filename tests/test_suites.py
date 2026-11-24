import pickle

import ioh
import numpy as np
import pytest

from bench_atlas.benchmarks import bbob, ma_bbob
from bench_atlas.problems import Problem, ProblemId, from_unit

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
