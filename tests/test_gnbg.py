import pickle
from pathlib import Path

import numpy as np
import pytest

from bench_atlas.gnbg import gnbg_ii, gnbg_iii
from bench_atlas.problems import Problem, ProblemId

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
    # Reference values from the official GNBG code, in original coordinates;
    # see tests/data/make_gnbg_reference.py.
    suite, problems = gnbg
    reference = np.load(Path(__file__).parent / "data" / f"{suite}_reference.npz")
    for problem in problems:
        x = reference[f"f{problem.id.function}_x"]
        y = reference[f"f{problem.id.function}_y"]
        values = problem.function.base(x)
        np.testing.assert_allclose(values, y, rtol=1e-12)
        assert values[0] == y[0] == problem.f_opt  # the first point is the optimum

def test_picklable(gnbg: tuple[str, list[Problem]]) -> None:
    x = np.random.default_rng(0).random((5, 30))
    for problem in gnbg[1]:
        np.testing.assert_array_equal(pickle.loads(pickle.dumps(problem))(x), problem(x))
