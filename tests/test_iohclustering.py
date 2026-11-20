import pickle

import numpy as np
import pytest
from iohclustering import CLUSTER_METRICS, get_dataset

from bench_atlas.problems import ProblemId
from bench_atlas.suites import iohclustering

PROBLEMS = iohclustering()


def test_ids() -> None:
    assert len(PROBLEMS) == 40
    assert len({problem.id for problem in PROBLEMS}) == 40
    assert PROBLEMS[0].id == ProblemId("iohclustering", 1, 1, 4)
    assert str(PROBLEMS[-1].id) == "iohclustering_f10_i1_d20"
    assert {problem.dimension for problem in PROBLEMS} == {4, 6, 10, 20}
    assert all(problem.f_opt is None and problem.x_opt is None for problem in PROBLEMS)


@pytest.mark.parametrize("problem", PROBLEMS, ids=lambda problem: str(problem.id))
def test_matches_iohclustering_metric(problem) -> None:
    # IOHClustering's (loop-based) mse_euclidean metric on the min-max normalization of
    # its create_cluster_problem, copied here: calling it would use ioh.wrap_problem.
    data = get_dataset(problem.id.function)
    normalized = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
    k = problem.dimension // data.shape[1]
    x = np.random.default_rng(problem.id.function).random((5, problem.dimension))
    expected = [CLUSTER_METRICS["mse_euclidean"](normalized, p.reshape(k, -1)) for p in x]
    np.testing.assert_allclose(problem(x), expected, rtol=1e-12)


def test_picklable() -> None:
    for problem in PROBLEMS:
        x = np.random.default_rng(0).random((5, problem.dimension))
        clone = pickle.loads(pickle.dumps(problem))
        np.testing.assert_array_equal(clone(x), problem(x))
