import numpy as np
import pytest
from lonkit import BasinHoppingSampler, BasinHoppingSamplerConfig, LONConfig

from bench_atlas.lon import (
    STEP_SIZE,
    _accepted_optima,
    _PointObjective,
    cmlon_features,
    cmlon_from_runs,
    sample_run,
)
from bench_atlas.problems import Problem, ProblemId
from bench_atlas.suites import bbob


def _sphere(x: np.ndarray) -> np.ndarray:
    return np.sum((x - 0.5) ** 2, axis=1)


def _runs(problem: Problem, n: int = 4, n_iter_no_change: int = 10):
    samples = (sample_run(problem, seed, n_iter_no_change=n_iter_no_change) for seed in range(n))
    return [(s["optima_x"], s["optima_f"]) for s in samples]


def test_sample_run_returns_the_accepted_optima() -> None:
    problem = Problem(ProblemId("test", 1, 1, 3), _sphere)
    run = sample_run(problem, seed=0, n_iter_no_change=10)
    assert run["evaluations"] == problem.evaluations
    assert len(run["optima_f"]) >= 1
    assert len(run["optima_x"]) == 3 * len(run["optima_f"])
    assert np.all(np.diff(run["optima_f"]) <= 0)
    assert np.allclose(run["optima_x"][-3:], 0.5, atol=1e-4)


def test_sample_run_is_deterministic_for_a_seed() -> None:
    problem = bbob([2], [1])[2]
    first, again, other = (sample_run(problem, s, n_iter_no_change=10) for s in (0, 0, 1))
    assert first["optima_f"] == again["optima_f"]
    assert first["optima_f"] != other["optima_f"]


def test_evaluations_count_only_the_run() -> None:
    problem = bbob([2], [1])[14]
    problem(np.full((10, 2), 0.5))
    run = sample_run(problem, 0, n_iter_no_change=10)
    assert run["evaluations"] == problem.evaluations - 10 > 0


def test_accepted_optima_of_special_runs() -> None:
    assert _accepted_optima([]) == ([], [])
    x = np.array([0.1, 0.2])
    preserved = {"current_x": x, "current_f": 3.0, "new_x": x, "new_f": 3.0, "accepted": False}
    optima_x, optima_f = _accepted_optima([preserved])
    assert optima_f == [3.0]
    assert np.array_equal(optima_x[0], x)


@pytest.mark.parametrize(
    ("function", "precisions"), [(3, (4, 4)), (22, (2, 3)), (15, (5, None)), (7, (None, None))]
)
def test_cmlon_matches_lonkit(function: int, precisions: tuple) -> None:
    """The CMLON built from the runs equals lonkit's own for the same raw records."""
    coordinate_precision, fitness_precision = precisions
    problem = bbob([2], [1])[function - 1]
    sampler = BasinHoppingSampler(
        BasinHoppingSamplerConfig(
            n_runs=5,
            n_iter_no_change=20,
            step_mode="fixed",
            step_size=STEP_SIZE,
            seed=3,
            coordinate_precision=coordinate_precision,
            fitness_precision=fitness_precision,
        )
    )
    sample = sampler.sample(_PointObjective(problem), [(0.0, 1.0)] * 2)
    expected = sampler.sample_to_lon(sample, LONConfig(warn_on_duplicates=False)).to_cmlon()
    runs = []
    for run in range(1, 6):
        optima_x, optima_f = _accepted_optima([r for r in sample.raw_records if r["run"] == run])
        runs.append(([v for x in optima_x for v in x], optima_f))
    actual = cmlon_from_runs(runs, 2, coordinate_precision, fitness_precision)
    assert actual.compute_metrics() == expected.compute_metrics()
    assert sorted(actual.graph.vs["name"]) == sorted(expected.graph.vs["name"])
    assert actual.n_edges == expected.n_edges


def test_sphere_has_one_funnel() -> None:
    features = cmlon_features(cmlon_from_runs(_runs(Problem(ProblemId("t", 1, 1, 3), _sphere)), 3))
    assert features["n_runs"] == 4
    assert features["n_optima"] == 1
    assert features["n_funnels"] == 1
    assert features["success"] == 1


@pytest.mark.parametrize("function", [2, 8])
def test_unimodal_bbob_has_one_funnel(function: int) -> None:
    features = cmlon_features(cmlon_from_runs(_runs(bbob([2], [1])[function - 1]), 2))
    assert features["n_funnels"] == 1
    assert features["success"] == 1


def test_lonkit_default_precision_splits_the_optimum() -> None:
    runs = _runs(bbob([2], [1])[1])
    assert cmlon_features(cmlon_from_runs(runs, 2, 5, None))["success"] < 1


def test_runs_without_optima_are_skipped() -> None:
    runs = [([], []), *_runs(bbob([2], [1])[0], n=2)]
    assert cmlon_features(cmlon_from_runs(runs, 2))["n_runs"] == 2
    empty = cmlon_from_runs([([], [])], 2)
    assert empty.n_vertices == 0
    assert cmlon_features(empty)["n_runs"] == 0


@pytest.mark.parametrize("function", [1, 21])
def test_metrics_are_finite(function: int) -> None:
    features = cmlon_features(cmlon_from_runs(_runs(bbob([2], [1])[function - 1]), 2))
    assert np.all(np.isfinite(list(features.values())))
