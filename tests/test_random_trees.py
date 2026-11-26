import json
import pickle
import random
from pathlib import Path

import numpy as np
import pytest

from bench_atlas.benchmarks import random_trees
from bench_atlas.benchmarks.random_trees import _draw
from bench_atlas.problems import ProblemId

DIMENSIONS = [2, 5, 20]
INSTANCES = range(1, 21)


@pytest.fixture(scope="module")
def suite():
    return random_trees(DIMENSIONS, INSTANCES)


def test_matches_doe2vec() -> None:
    # Expressions of doe2vec's generator; see tests/data/make_random_trees_reference.py.
    path = Path(__file__).parent / "data" / "random_trees_reference.json"
    for key, expected in json.loads(path.read_text()).items():
        seed, d = map(int, key.split("_"))
        assert _draw(random.Random(seed), np.random.RandomState(seed), d) == expected, key


def test_ids(suite) -> None:
    assert [problem.id for problem in suite] == [
        ProblemId("random_trees", 1, instance, d) for instance in INSTANCES for d in DIMENSIONS
    ]
    assert all(problem.x_opt is None and problem.f_opt is None for problem in suite)


def test_reproducible_and_picklable(suite) -> None:
    again = random_trees(DIMENSIONS, INSTANCES)
    rng = np.random.default_rng(0)
    for problem, other in zip(suite, again, strict=True):
        assert problem.function.expression == other.function.expression
        x = rng.random((100, problem.dimension))
        y = problem(x)
        assert np.array_equal(y, other(x), equal_nan=True)
        assert np.array_equal(y, pickle.loads(pickle.dumps(problem))(x), equal_nan=True)


def test_points_are_independent(suite) -> None:
    # A value must not depend on the other points of the batch (doe2vec's noise
    # operand does, so such trees are rejected). The rotated vector (np.dot) differs by
    # 1 ULP between batch sizes, and sin / log can amplify that to ~1e-12.
    rng = np.random.default_rng(1)
    for problem in suite:
        x = rng.random((50, problem.dimension))
        with np.errstate(all="ignore"):
            batch = problem(x)
            single = np.concatenate([problem(x[i : i + 1]) for i in range(len(x))])
        np.testing.assert_allclose(single, batch, rtol=1e-9, atol=0)
