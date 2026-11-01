import pickle

import ioh
import numpy as np
import pytest

from bench_atlas.problems import (
    ProblemId,
    from_function,
    from_ioh,
    from_unit,
    to_unit,
)


def bbob(function: int, instance: int, dimension: int):
    return ioh.get_problem(function, instance, dimension, problem_class=ioh.ProblemClass.BBOB)


def sphere(x: np.ndarray) -> np.ndarray:
    return np.sum(x**2, axis=1)


def test_problem_id_str() -> None:
    assert str(ProblemId("ma_bbob", 12, 3, 5)) == "ma_bbob_f12_i3_d5"


def test_unit_mapping_roundtrip() -> None:
    lb, ub = np.array([-5.0, 0.0]), np.array([5.0, 2.0])
    x = np.array([[1.5, 0.25], [-5.0, 2.0]])
    np.testing.assert_allclose(from_unit(to_unit(x, lb, ub), lb, ub), x)


class TestProblem:
    def test_evaluates_batch(self) -> None:
        problem = from_function(sphere, id=ProblemId("test", 1, 1, 2), lb=0.0, ub=1.0)
        np.testing.assert_array_equal(problem([[0.0, 0.0], [1.0, 1.0]]), [0.0, 2.0])

    def test_counts_evaluations(self) -> None:
        problem = from_function(sphere, id=ProblemId("test", 1, 1, 2), lb=0.0, ub=1.0)
        problem([[0.5, 0.5]])
        problem(np.zeros((4, 2)))
        assert problem.evaluations == 5

    def test_rejects_wrong_output_shape(self) -> None:
        problem = from_function(
            lambda x: sphere(x)[:, None], id=ProblemId("test", 1, 1, 2), lb=0.0, ub=1.0
        )
        with pytest.raises(ValueError):
            problem(np.zeros((3, 2)))

    def test_rejects_wrong_dimension(self) -> None:
        problem = from_function(sphere, id=ProblemId("test", 1, 1, 2), lb=0.0, ub=1.0)
        with pytest.raises(ValueError):
            problem([[0.5, 0.5, 0.5]])

    def test_rejects_single_point(self) -> None:
        problem = from_function(sphere, id=ProblemId("test", 1, 1, 2), lb=0.0, ub=1.0)
        with pytest.raises(ValueError):
            problem([0.5, 0.5])


class TestFromFunction:
    def test_maps_domain(self) -> None:
        problem = from_function(
            sphere,
            id=ProblemId("test", 1, 1, 2),
            lb=[-5.0, 0.0],
            ub=[5.0, 10.0],
            x_opt=[0.0, 0.0],
            f_opt=0.0,
        )
        np.testing.assert_array_equal(problem([[0.5, 0.0], [1.0, 1.0]]), [0.0, 125.0])
        np.testing.assert_allclose(problem.x_opt, [0.5, 0.0])

    def test_rejects_empty_box(self) -> None:
        with pytest.raises(ValueError):
            from_function(sphere, id=ProblemId("test", 1, 1, 2), lb=1.0, ub=1.0)


class TestFromIOH:
    @pytest.mark.parametrize("dimension", [2, 5])
    @pytest.mark.parametrize("function", range(1, 25))
    def test_bbob_identical_at_mapped_points(self, function: int, dimension: int) -> None:
        raw = bbob(function, 1, dimension)
        problem = from_ioh(raw, "bbob")
        points = np.random.default_rng(function).random((20, dimension))
        mapped = from_unit(points, raw.bounds.lb, raw.bounds.ub)

        expected = [raw(x) for x in mapped]

        assert problem(points).tolist() == expected

    def test_id_name_and_optimum(self) -> None:
        raw = bbob(3, 2, 5)
        problem = from_ioh(raw, "bbob")
        assert problem.id == ProblemId("bbob", 3, 2, 5)
        assert problem.name == "Rastrigin"
        np.testing.assert_allclose(
            problem.x_opt, to_unit(raw.optimum.x, raw.bounds.lb, raw.bounds.ub)
        )
        assert problem.f_opt == raw.optimum.y
        assert problem(problem.x_opt[None, :])[0] == pytest.approx(problem.f_opt)

    def test_picklable(self) -> None:
        problem = from_ioh(bbob(3, 2, 5), "bbob")
        points = np.random.default_rng(0).random((5, 5))
        clone = pickle.loads(pickle.dumps(problem))
        assert clone.id == problem.id
        np.testing.assert_array_equal(clone(points), problem(points))
