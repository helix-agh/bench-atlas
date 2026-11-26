import json
import pickle
import random
from pathlib import Path

import numpy as np
import pytest

from bench_atlas.benchmarks import random_trees
from bench_atlas.benchmarks.random_trees import _draw, _RandomTree
from bench_atlas.problems import ProblemId

DIMENSIONS = [1, 2, 5, 20]
INSTANCES = range(1, 21)


@pytest.fixture(scope="module")
def suite():
    return random_trees(DIMENSIONS, INSTANCES)


# Our fixed operands and doe2vec's originals (docs/random-trees.md).
FIXED_OPERANDS = {
    "array_x[:,:1]": "array_x[:,0]",
    "(np.hstack((array_x[:,1:], np.zeros((len(array_x), 1)))))": (
        "(np.vstack((array_x[:,1:].ravel(), np.zeros((len(array_x), 1)).ravel())).T)"
    ),
    "(dot(": "(np.dot(",
    ", axis=1, keepdims=True)": ", axis=1)",
}


def test_matches_doe2vec() -> None:
    # Expressions of doe2vec's generator, up to the fixed operands; see
    # tests/data/make_random_trees_reference.py.
    path = Path(__file__).parent / "data" / "random_trees_reference.json"
    for key, expected in json.loads(path.read_text()).items():
        seed, d = map(int, key.split("_"))
        expression = _draw(random.Random(seed), np.random.RandomState(seed), d)
        for ours, doe2vec in FIXED_OPERANDS.items():
            expression = expression and expression.replace(ours, doe2vec)
        assert expression == expected, key


def test_fixed_operands() -> None:
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    m = np.arange(9.0).reshape(3, 3)
    for ours, expected in [
        ("np.mean(array_x[:,:1]+array_x, axis=1, keepdims=True)", [3.0, 9.0]),
        ("np.mean(np.sum(array_x, axis=1, keepdims=True)-array_x, axis=1, keepdims=True)", [4, 10]),
        (
            "np.sum(np.hstack((array_x[:,1:], np.zeros((len(array_x), 1))))*(np.array(range(1, 4))), axis=1, keepdims=True)",
            [8, 17],
        ),
        (f"np.sum(dot(array_x, np.array({m.tolist()})), axis=1, keepdims=True)", (x @ m).sum(1)),
    ]:
        np.testing.assert_array_equal(_RandomTree(ours)(x), expected)


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
    # A value must not depend on the other points of the batch, bitwise. doe2vec's x1
    # and reductions mixed rows at d = 1 and d = 256 (the filter's sample size), its
    # noise depends on the batch (such trees are rejected), np.dot differs by 1 ULP.
    rng = np.random.default_rng(1)
    for problem in suite + random_trees([256], range(5)):
        x = rng.random((50, problem.dimension))
        with np.errstate(all="ignore"):
            batch = problem(x)
            single = np.concatenate([problem(x[i : i + 1]) for i in range(len(x))])
        np.testing.assert_array_equal(single, batch)
