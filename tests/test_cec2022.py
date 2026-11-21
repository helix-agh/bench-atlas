import pickle
from pathlib import Path

import ioh
import numpy as np
import pytest

from bench_atlas.problems import Problem, ProblemId, from_unit
from bench_atlas.suites import cec2022

REFERENCE = np.load(Path(__file__).parent / "data" / "cec2022_reference.npz")
DIFFER_FROM_CODE = (3, 5, 9)  # IOH follows the technical report, not the code


def uniform(function: int, d: int, n: int) -> np.ndarray:
    """Points on [0, 1)^d from raw PCG64 output; must match make_cec2022_reference.py."""
    seed = np.random.SeedSequence([int.from_bytes(b"cec2022", "little"), function, d])
    return (np.random.PCG64(seed).random_raw((n, d)) >> 11) * 2.0**-53


def raw(function: int, d: int) -> ioh.problem.RealSingleObjective:
    return ioh.get_problem(1000 + function, 1, d, ioh.ProblemClass.CEC2022)


@pytest.fixture(scope="module")
def problems() -> list[Problem]:
    return cec2022([10, 20])


def test_ids(problems: list[Problem]) -> None:
    assert [problem.id for problem in problems] == [
        ProblemId("cec2022", 1000 + f, 1, d) for f in range(1, 13) for d in (10, 20)
    ]
    with pytest.raises(ValueError, match="dimensions"):
        cec2022([2, 10])


def test_identical_to_ioh(problems: list[Problem]) -> None:
    for problem in problems:
        points = np.random.default_rng(0).random((20, problem.dimension))
        expected = raw(problem.id.function - 1000, problem.dimension)
        assert problem(points).tolist() == [expected(x) for x in from_unit(points, -100.0, 100.0)]


@pytest.mark.parametrize("d", [10, 20])
@pytest.mark.parametrize("function", range(1, 13))
def test_ioh_matches_official_code(function: int, d: int) -> None:
    # Values of the official C code; see tests/data/make_cec2022_reference.py. For F3, F5
    # and F9 ioh 0.3.22 follows the technical report, not the code (see cec2022()); if
    # that branch fails, IOH has changed and the docstring needs updating.
    problem, key = raw(function, d), f"f{function}_d{d}"
    x = np.vstack([REFERENCE[f"{key}_x"], -100 + 200 * uniform(function, d, 100)])
    y = np.concatenate([REFERENCE[f"{key}_y"], REFERENCE[f"{key}_uniform_y"]])
    values = np.array([problem(xi) for xi in x])
    assert np.array_equal(x[0], problem.optimum.x) and values[0] == y[0] == problem.optimum.y
    if function not in DIFFER_FROM_CODE:
        np.testing.assert_allclose(values, y, rtol=1e-12)
    else:
        assert np.max(np.abs(values - y) / y) > 1e-3


def test_value_at_optimum(problems: list[Problem]) -> None:
    # Mapping x_opt to [0, 1]^d and back moves it by ~1e-14, which changes F8 at
    # d = 10 by 1.5e-4.
    for problem in problems:
        assert problem(problem.x_opt[None, :])[0] == pytest.approx(problem.f_opt, abs=1e-3)


def test_picklable(problems: list[Problem]) -> None:
    for problem in problems:
        x = np.random.default_rng(0).random((5, problem.dimension))
        clone = pickle.loads(pickle.dumps(problem))
        np.testing.assert_array_equal(clone(x), problem(x))
