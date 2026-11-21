import pickle
from pathlib import Path

import numpy as np
import pytest

from bench_atlas.problems import Problem, ProblemId, to_unit
from bench_atlas.zigzag import zigzag

DIMENSIONS = [2, 5, 10, 15, 20]
REFERENCE = np.load(Path(__file__).parent / "data" / "zigzag_reference.npz")


def uniform(function: int, d: int, n: int) -> np.ndarray:
    """Points on [0, 1)^d from raw PCG64 output; must match make_zigzag_reference.py."""
    seed = np.random.SeedSequence([int.from_bytes(b"zigzag", "little"), function, d])
    return (np.random.PCG64(seed).random_raw((n, d)) >> 11) * 2.0**-53


@pytest.fixture(scope="module")
def problems() -> list[Problem]:
    return zigzag(DIMENSIONS, range(1, 101))


def test_ids(problems: list[Problem]) -> None:
    assert len(problems) == 4 * 100 * len(DIMENSIONS)
    assert len({problem.id for problem in problems}) == len(problems)
    assert problems[0].id == ProblemId("zigzag", 1, 1, 2)
    # Table 1 of the paper: ID 37 is k = 2, m = 0.5, lambda = 0.5.
    problem = zigzag([10], [37])[0]
    assert problem.name == "F1_k2_m0.5_lambda0.5"


def test_matches_official_code(problems: list[Problem]) -> None:
    # Values of the official MATLAB code in Octave; see tests/data/make_zigzag_reference.py.
    for problem in problems:
        key = f"f{problem.id.function}_d{problem.dimension}"
        p = problem.id.instance - 1
        x = REFERENCE[f"{key}_x"]
        assert REFERENCE[f"{key}_y"][p, 0] == problem.f_opt  # the first point is x = -s
        np.testing.assert_array_equal(to_unit(x[0], -100.0, 100.0), problem.x_opt)
        uniform_y = REFERENCE[f"{key}_uniform_y"][p]
        u = uniform(problem.id.function, problem.dimension, len(uniform_y))
        # rtol: the steep polynomial of F1/F2 and the compositions F2/F4 amplify rounding:
        # the same formula on the original box differs by up to 7e-12, mapping the points
        # to [0, 1]^d raises it to 4e-10 (F4).
        np.testing.assert_allclose(
            problem(to_unit(x, -100.0, 100.0)), REFERENCE[f"{key}_y"][p], rtol=1e-9
        )
        np.testing.assert_allclose(problem(u), uniform_y, rtol=1e-9)


def test_value_at_optimum(problems: list[Problem]) -> None:
    for problem in problems:
        assert problem(problem.x_opt[None, :])[0] == problem.f_opt == 0.0


def test_rejects_unsupported_settings() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        zigzag([3], [1])
    with pytest.raises(ValueError, match="parametrizations"):
        zigzag([2], [0, 101])


def test_picklable(problems: list[Problem]) -> None:
    for problem in problems[::7]:
        x = np.random.default_rng(0).random((5, problem.dimension))
        np.testing.assert_array_equal(pickle.loads(pickle.dumps(problem))(x), problem(x))
