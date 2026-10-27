import pickle

import ioh
import numpy as np
import pytest

from bench_atlas.problems import (
    Problem,
    ProblemId,
    from_function,
    from_ioh,
    from_unit,
    make_instance,
    to_unit,
)


def bbob(function: int, instance: int, dimension: int):
    return ioh.get_problem(function, instance, dimension, problem_class=ioh.ProblemClass.BBOB)


def sphere(x: np.ndarray) -> np.ndarray:
    return np.sum(x**2, axis=1)


def rastrigin(x: np.ndarray) -> np.ndarray:
    return 10 * x.shape[1] + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=1)


def sphere_instance(seed: int = 0, instance: int = 1, dimension: int = 3, **kwargs) -> Problem:
    return make_instance(
        sphere,
        id=ProblemId("test", 1, instance, dimension),
        lb=-5.0,
        ub=5.0,
        x_opt=np.zeros(dimension),
        seed=seed,
        **kwargs,
    )


class TestProblemId:
    def test_str_roundtrip(self) -> None:
        pid = ProblemId("ma_bbob", 12, 3, 5)
        assert str(pid) == "ma_bbob_f12_i3_d5"
        assert ProblemId.parse(str(pid)) == pid

    def test_parse_rejects_garbage(self) -> None:
        with pytest.raises(ValueError):
            ProblemId.parse("bbob-1-2-3")


def test_unit_mapping_roundtrip() -> None:
    lb, ub = np.array([-5.0, 0.0]), np.array([5.0, 2.0])
    x = np.array([[1.5, 0.25], [-5.0, 2.0]])
    np.testing.assert_allclose(from_unit(to_unit(x, lb, ub), lb, ub), x)


class TestProblem:
    def test_single_point_and_batch(self) -> None:
        problem = from_function(sphere, id=ProblemId("test", 1, 1, 2), lb=0.0, ub=1.0)
        assert problem([0.5, 0.5]) == 0.5
        np.testing.assert_array_equal(problem([[0.0, 0.0], [1.0, 1.0]]), [0.0, 2.0])

    def test_counts_evaluations(self) -> None:
        problem = from_function(sphere, id=ProblemId("test", 1, 1, 2), lb=0.0, ub=1.0)
        problem([0.5, 0.5])
        problem(np.zeros((4, 2)))
        assert problem.evaluations == 5

    def test_rejects_wrong_dimension(self) -> None:
        problem = from_function(sphere, id=ProblemId("test", 1, 1, 2), lb=0.0, ub=1.0)
        with pytest.raises(ValueError):
            problem([0.5, 0.5, 0.5])


class TestFromFunction:
    def test_maps_domain(self) -> None:
        problem = from_function(
            sphere,
            id=ProblemId("test", 1, 1, 2),
            lb=[-5.0, 0.0],
            ub=[5.0, 10.0],
            optimum=([0.0, 0.0], 0.0),
        )
        assert problem([0.5, 0.0]) == 0.0
        assert problem([1.0, 1.0]) == 125.0
        np.testing.assert_allclose(problem.optimum[0], [0.5, 0.0])

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
        assert [problem(u) for u in points] == expected

    def test_id_name_and_optimum(self) -> None:
        raw = bbob(3, 2, 5)
        problem = from_ioh(raw, "bbob")
        assert problem.id == ProblemId("bbob", 3, 2, 5)
        assert problem.name == "Rastrigin"
        x_opt, f_opt = problem.optimum
        np.testing.assert_allclose(x_opt, to_unit(raw.optimum.x, raw.bounds.lb, raw.bounds.ub))
        assert f_opt == raw.optimum.y
        assert problem(x_opt) == pytest.approx(f_opt)

    def test_picklable(self) -> None:
        problem = from_ioh(bbob(3, 2, 5), "bbob")
        points = np.random.default_rng(0).random((5, 5))
        clone = pickle.loads(pickle.dumps(problem))
        assert clone.id == problem.id
        np.testing.assert_array_equal(clone(points), problem(points))


class TestMakeInstance:
    def test_reproducible_from_seed(self) -> None:
        points = np.random.default_rng(1).random((10, 3))
        a, b = sphere_instance(seed=42), sphere_instance(seed=42)
        np.testing.assert_array_equal(a(points), b(points))
        np.testing.assert_array_equal(a.optimum[0], b.optimum[0])
        assert a.optimum[1] == b.optimum[1]

    @pytest.mark.parametrize(
        "other", [{"seed": 43}, {"seed": 42, "instance": 2}], ids=["seed", "instance"]
    )
    def test_differs_across_seeds_and_instances(self, other: dict) -> None:
        a, b = sphere_instance(seed=42), sphere_instance(**other)
        assert not np.allclose(a.optimum[0], b.optimum[0])

    @pytest.mark.parametrize("seed", range(20))
    def test_optimum_inside_domain(self, seed: int) -> None:
        problem = sphere_instance(seed=seed, dimension=5)
        x_opt, f_opt = problem.optimum
        assert np.all((x_opt >= 0.1) & (x_opt <= 0.9))
        assert problem(x_opt) == pytest.approx(f_opt)

    @pytest.mark.parametrize("function", [sphere, rastrigin])
    def test_optimum_is_minimum(self, function) -> None:
        problem = make_instance(
            function,
            id=ProblemId("test", 1, 1, 4),
            lb=-5.12,
            ub=5.12,
            x_opt=np.zeros(4),
            seed=7,
        )
        points = np.random.default_rng(0).random((200, 4))
        assert problem(points).min() > problem.optimum[1]

    def test_rotation_preserves_sphere(self) -> None:
        problem = sphere_instance(seed=3, dimension=4)
        u = np.random.default_rng(2).random((1, 4))
        x_star = from_unit(problem.optimum[0], -5.0, 5.0)
        expected = sphere(from_unit(u, -5.0, 5.0) - x_star) + problem.optimum[1]
        np.testing.assert_allclose(problem(u), expected)

    def test_identity_when_disabled(self) -> None:
        problem = sphere_instance(seed=3, shift=False, rotate=False, offset=False)
        u = np.random.default_rng(2).random((5, 3))
        np.testing.assert_allclose(problem(u), sphere(from_unit(u, -5.0, 5.0)))
        assert problem.optimum[1] == 0.0

    def test_picklable(self) -> None:
        problem = sphere_instance(seed=3)
        points = np.random.default_rng(0).random((5, 3))
        np.testing.assert_array_equal(pickle.loads(pickle.dumps(problem))(points), problem(points))

    def test_rejects_optimum_outside_domain(self) -> None:
        with pytest.raises(ValueError):
            make_instance(sphere, id=ProblemId("test", 1, 1, 2), lb=0.0, ub=1.0, x_opt=[2.0, 0.5])
