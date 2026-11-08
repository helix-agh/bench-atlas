import warnings

import numpy as np
import pytest
from lonkit import CMLON

from bench_atlas.lon import cmlon, lon_features
from bench_atlas.problems import Problem, ProblemId
from bench_atlas.suites import bbob

SMALL = {"n_runs": 4, "n_iter_no_change": 10}


def _sphere(x: np.ndarray) -> np.ndarray:
    return np.sum((x - 0.5) ** 2, axis=1)


def test_sphere_has_one_funnel() -> None:
    problem = Problem(ProblemId("test", 1, 1, 3), _sphere)
    features = lon_features(problem, seed=0, **SMALL)
    assert features["evaluations"] == problem.evaluations
    assert features["n_runs"] == SMALL["n_runs"]
    assert features["n_optima"] == 1
    assert features["n_funnels"] == 1
    assert features["success"] == 1


@pytest.mark.parametrize("function", [2, 8])
def test_unimodal_bbob_has_one_funnel(function: int) -> None:
    features = lon_features(bbob([2], [1])[function - 1], 0, **SMALL)
    assert features["n_funnels"] == 1
    assert features["success"] == 1


def test_budget_limits_evaluations_and_runs() -> None:
    problem = bbob([2], [1])[14]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        features = lon_features(problem, 0, budget_per_dimension=2000, n_runs=20)
    assert problem.evaluations == features["evaluations"] <= 2000 * 2
    assert 0 < features["n_runs"] < 20
    assert not [w for w in caught if "budget" in str(w.message)]


def test_budget_counts_from_the_current_evaluations() -> None:
    problem = bbob([2], [1])[14]
    problem(np.full((10, 2), 0.5))
    features = lon_features(problem, 0, budget_per_dimension=2000, n_runs=20)
    assert features["evaluations"] == problem.evaluations - 10 <= 2000 * 2


def test_deterministic_for_a_seed() -> None:
    problem = bbob([2], [1])[2]
    first, again, other = (lon_features(problem, s, **SMALL) for s in (0, 0, 1))
    assert first == again
    assert first != other


def test_cmlon_graph() -> None:
    network = cmlon(bbob([2], [1])[2], 0, **SMALL)
    assert isinstance(network, CMLON)
    assert network.n_vertices >= 1
    assert network.source_lon is not None


@pytest.mark.parametrize("function", [1, 21])
def test_metrics_are_finite(function: int) -> None:
    features = lon_features(bbob([2], [1])[function - 1], 0, **SMALL)
    assert np.all(np.isfinite(list(features.values())))
