import pickle
from pathlib import Path

import numpy as np
import pytest

from bench_atlas.gnbg import gnbg_ii, gnbg_iii
from bench_atlas.problems import Problem, ProblemId, to_unit

GNBG = {"gnbg_ii": (gnbg_ii, 24), "gnbg_iii": (gnbg_iii, 22)}


@pytest.fixture(scope="module", params=list(GNBG))
def gnbg(request) -> tuple[str, list[Problem]]:
    load, _ = GNBG[request.param]
    return request.param, load()


def test_ids(gnbg: tuple[str, list[Problem]]) -> None:
    # GNBG-III: only the static problems F1-F22.
    suite, problems = gnbg
    assert [problem.id for problem in problems] == [
        ProblemId(suite, function, 1, 30) for function in range(1, GNBG[suite][1] + 1)
    ]


def test_matches_official_code(gnbg: tuple[str, list[Problem]]) -> None:
    # Reference values from the official GNBG code at points in the original box
    # [-100, 100]^30; see tests/data/make_gnbg_reference.py.
    suite, problems = gnbg
    reference = np.load(Path(__file__).parent / "data" / f"{suite}_reference.npz")
    for problem in problems:
        x = reference[f"f{problem.id.function}_x"]
        y = reference[f"f{problem.id.function}_y"]
        # rtol: mapping the points to [0, 1]^d alone changes f by up to ~2e-11.
        np.testing.assert_allclose(problem(to_unit(x, -100.0, 100.0)), y, rtol=1e-10)
        assert y[0] == problem.f_opt  # the first reference point is the optimum


def test_value_at_optimum(gnbg: tuple[str, list[Problem]]) -> None:
    for problem in gnbg[1]:
        assert problem(problem.x_opt[None, :])[0] == problem.f_opt


def test_picklable(gnbg: tuple[str, list[Problem]]) -> None:
    x = np.random.default_rng(0).random((5, 30))
    for problem in gnbg[1]:
        np.testing.assert_array_equal(pickle.loads(pickle.dumps(problem))(x), problem(x))
