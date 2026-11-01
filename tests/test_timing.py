import time

import numpy as np
import pytest

from bench_atlas.problems import Problem, ProblemId
from bench_atlas.timing import WARMUP, evaluation_time


def _sleep(x: np.ndarray) -> np.ndarray:
    time.sleep(1e-3)
    return np.zeros(len(x))


def test_measures_a_known_cost() -> None:
    problem = Problem(ProblemId("test", 1, 1, 3), _sleep)
    assert evaluation_time(problem, n_points=50) == pytest.approx(1e-3, rel=0.5)


def test_evaluates_single_points() -> None:
    sizes = []

    def record(x: np.ndarray) -> np.ndarray:
        sizes.append(len(x))
        return np.zeros(len(x))

    problem = Problem(ProblemId("test", 1, 1, 3), record)
    evaluation_time(problem, n_points=20)
    assert sizes == [1] * (WARMUP + 20)
