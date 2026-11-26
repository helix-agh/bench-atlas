import pickle

import numpy as np
import pytest
from scipy.optimize import minimize

from bench_atlas.benchmarks import msg
from bench_atlas.problems import ProblemId

DIMENSIONS = [2, 5, 10, 20, 40]
INSTANCES = [1, 2, 3, 4, 5]


def test_ids() -> None:
    problems = msg(DIMENSIONS, INSTANCES)
    assert len({problem.id for problem in problems}) == len(problems) == 25
    assert problems[0].id == ProblemId("msg", 1, 1, 2)
    sizes = [len(problem.function.centers) for problem in problems[::5]]
    assert sizes == [50 * d for d in DIMENSIONS]


def test_reproducible() -> None:
    # Pinned values: catch changes of scipy's Sobol' scrambling or numpy's Generator.
    function = msg([2], [1])[0].function
    assert function.centers[0].tolist() == [0.2536503244191408, 0.5298733506351709]
    assert function.w[0] == 0.44841514333227195
    assert function.sigma[0] == 0.17501943591045985
    function = msg([40], [3])[0].function
    assert function.centers[-1, -1] == 0.3834801046177745
    assert function.w[-1] == 0.3687879130963847
    assert function.sigma[-1] == 1.7326934263485656
    # Centers depend on d only; the instance changes w and sigma.
    a, b = msg([5], [1, 2])
    assert np.array_equal(a.function.centers, b.function.centers)
    assert not np.array_equal(a.function.w, b.function.w)


def test_value_at_optimum() -> None:
    for problem in msg(DIMENSIONS, INSTANCES):
        assert problem(problem.x_opt[None, :])[0] == problem.f_opt == -problem.function.w.max()
        x = np.random.default_rng(0).random((1000, problem.dimension))
        assert np.all(problem(x) >= problem.f_opt)


@pytest.mark.parametrize("d", [1, 2, 3])
def test_local_optima_brute_force(d: int) -> None:
    # Local search from every center ends at itself exactly for the enumerated local
    # optima and at another enumerated optimum for the dominated centers.
    rng = np.random.default_rng(d)
    for problem in msg([d], [1, 2, 3]):
        centers = problem.function.centers
        optima = {tuple(c) for c in problem.function.local_optima()}
        for c in centers:
            f = problem.function
            result = minimize(
                lambda x, f=f: f(x[None, :])[0], c, method="L-BFGS-B", bounds=[(0, 1)] * d
            )
            end = centers[np.argmin(np.linalg.norm(centers - result.x, axis=1))]
            assert np.linalg.norm(result.x - end) < 1e-3
            assert tuple(end) in optima
            if tuple(c) in optima:
                assert np.array_equal(end, c)
                step = rng.normal(size=(100, d))
                step *= 1e-6 / np.linalg.norm(step, axis=1, keepdims=True)
                assert np.all(problem(c + step) > problem(c[None, :])[0])
            else:
                assert not np.array_equal(end, c)


def test_picklable() -> None:
    for problem in msg([2, 40], [1]):
        x = np.random.default_rng(0).random((5, problem.dimension))
        np.testing.assert_array_equal(pickle.loads(pickle.dumps(problem))(x), problem(x))
