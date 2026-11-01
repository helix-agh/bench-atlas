import time

import numpy as np
import pytest

from bench_atlas.problems import Problem, ProblemId
from bench_atlas.timing import MIN_BATCHES, WARMUP, evaluation_time


def _sleep(x: np.ndarray) -> np.ndarray:
    time.sleep(1e-3)
    return np.zeros(len(x))


@pytest.mark.parametrize("batch_size", [1, 4])
def test_measures_a_known_cost_per_point(batch_size: int) -> None:
    problem = Problem(ProblemId("test", 1, 1, 3), _sleep)
    t = evaluation_time(problem, batch_size=batch_size, n_points=50)
    assert t == pytest.approx(1e-3 / batch_size, rel=0.5)


@pytest.mark.parametrize(("batch_size", "n_batches"), [(1, 20), (5, MIN_BATCHES)])
def test_evaluates_batches(batch_size: int, n_batches: int) -> None:
    sizes = []

    def record(x: np.ndarray) -> np.ndarray:
        sizes.append(len(x))
        return np.zeros(len(x))

    problem = Problem(ProblemId("test", 1, 1, 3), record)
    evaluation_time(problem, batch_size=batch_size, n_points=20)
    assert sizes == [batch_size] * (WARMUP + n_batches)
