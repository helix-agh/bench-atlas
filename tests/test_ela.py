import numpy as np
import pytest

from bench_atlas.benchmarks import bbob
from bench_atlas.ela import ela_features
from bench_atlas.problems import Problem, ProblemId


def _sphere(x: np.ndarray) -> np.ndarray:
    return np.sum((x - 0.5) ** 2, axis=1)


def test_sample_size_and_f_med() -> None:
    problem = Problem(ProblemId("test", 1, 1, 3), _sphere)
    features = ela_features(problem, seed=0, points_per_dimension=20)
    assert problem.evaluations == 20 * 3
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


def test_objective_values_are_minmax_normalized() -> None:
    problem = bbob([2], [1])[6]
    scaled = Problem(problem.id, lambda x: 1000 * problem(x) + 7)
    features, features_scaled = ela_features(problem, 0), ela_features(scaled, 0)
    assert features_scaled.pop("f_med") == pytest.approx(1000 * features.pop("f_med") + 7)
    assert features_scaled == pytest.approx(features, rel=1e-9, abs=1e-12)


def test_sample_depends_only_on_the_seed_and_dimension() -> None:
    samples: list[np.ndarray] = []

    def record(x: np.ndarray) -> np.ndarray:
        samples.append(x.copy())
        return _sphere(x)

    for function in (1, 2):
        ela_features(Problem(ProblemId("test", function, 1, 3), record), seed=5)
    ela_features(Problem(ProblemId("test", 1, 1, 3), record), seed=6)
    assert np.array_equal(samples[0], samples[1])
    assert not np.array_equal(samples[0], samples[2])
