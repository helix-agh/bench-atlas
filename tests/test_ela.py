import numpy as np
import pytest

from bench_atlas.ela import POINTS_PER_DIMENSION, ela_features
from bench_atlas.problems import Problem, ProblemId
from bench_atlas.suites import bbob


def _sphere(x: np.ndarray) -> np.ndarray:
    return np.sum((x - 0.5) ** 2, axis=1)


def test_sample_size_and_f_med() -> None:
    problem = Problem(ProblemId("test", 1, 1, 3), _sphere)
    features = ela_features(problem, seed=0)
    assert problem.evaluations == POINTS_PER_DIMENSION * 3
    assert 0 < features["f_med"] < 0.75
    assert features["ela_meta.quad_simple.adj_r2"] == pytest.approx(1)


def test_deterministic_for_a_seed() -> None:
    problem = bbob([2], [1])[6]
    first, again, other = (ela_features(problem, s) for s in (0, 0, 1))
    assert first == again
    assert first != other


def test_all_feature_groups() -> None:
    groups = {name.split(".")[0] for name in ela_features(bbob([2], [1])[0], 0)}
    assert groups == {"f_med", "ela_distr", "ela_meta", "nbc", "ic", "fitness_distance"}
