import gc

import ioh
import numpy as np
import pytest

from bench_atlas.problems import (
    ProblemId,
    from_unit,
    make_instance,
    normalize,
    problem_id,
    to_unit,
    wrap,
)


def bbob(function: int, instance: int, dimension: int):
    return ioh.get_problem(function, instance, dimension, problem_class=ioh.ProblemClass.BBOB)


def sphere(x: np.ndarray) -> float:
    return float(np.sum(np.asarray(x) ** 2))


def rastrigin(x: np.ndarray) -> float:
    x = np.asarray(x)
    return float(10 * x.size + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))


def sphere_instance(seed: int = 0, instance: int = 1, dimension: int = 3, **kwargs):
    return make_instance(
        sphere,
        suite="test",
        name="Sphere",
        function_id=1,
        dimension=dimension,
        instance=instance,
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

    def test_from_ioh_meta_data(self) -> None:
        assert problem_id(bbob(7, 4, 5), "bbob") == ProblemId("bbob", 7, 4, 5)

    def test_normalization_preserves_id(self) -> None:
        assert problem_id(normalize(bbob(7, 4, 5), "bbob"), "bbob") == ProblemId("bbob", 7, 4, 5)


def test_unit_mapping_roundtrip() -> None:
    lb, ub = np.array([-5.0, 0.0]), np.array([5.0, 2.0])
    x = np.array([[1.5, 0.25], [-5.0, 2.0]])
    np.testing.assert_allclose(from_unit(to_unit(x, lb, ub), lb, ub), x)


class TestNormalize:
    @pytest.mark.parametrize("dimension", [2, 5])
    @pytest.mark.parametrize("function", range(1, 25))
    def test_bbob_identical_at_mapped_points(self, function: int, dimension: int) -> None:
        raw = bbob(function, 1, dimension)
        unit = normalize(raw, "bbob")
        points = np.random.default_rng(function).random((20, dimension))
        mapped = from_unit(points, raw.bounds.lb, raw.bounds.ub)

        expected = [raw(x) for x in mapped]
        actual = [unit(u) for u in points]

        assert actual == expected

    def test_bounds_and_optimum(self) -> None:
        raw = bbob(3, 2, 5)
        unit = normalize(raw, "bbob")
        np.testing.assert_array_equal(unit.bounds.lb, np.zeros(5))
        np.testing.assert_array_equal(unit.bounds.ub, np.ones(5))
        np.testing.assert_allclose(
            unit.optimum.x, to_unit(raw.optimum.x, raw.bounds.lb, raw.bounds.ub)
        )
        assert unit.optimum.y == raw.optimum.y
        assert unit(unit.optimum.x) == pytest.approx(unit.optimum.y)

    def test_keeps_meta_data(self) -> None:
        raw = bbob(3, 2, 5)
        unit = normalize(raw, "bbob")
        assert unit.meta_data.name == raw.meta_data.name
        assert unit.meta_data.problem_id == 3
        assert unit.meta_data.instance == 2

    def test_unit_problem_is_returned_unchanged(self) -> None:
        problem = wrap(sphere, suite="test", name="Sphere", function_id=1, dimension=2, lb=0, ub=1)
        assert normalize(problem, "test") is problem

    def test_counts_evaluations(self) -> None:
        unit = normalize(bbob(1, 1, 2), "bbob")
        unit([0.5, 0.5])
        unit(np.random.default_rng(0).random((4, 2)))
        assert unit.state.evaluations == 5


class TestWrap:
    def test_maps_domain_through_ioh(self) -> None:
        problem = wrap(
            sphere,
            suite="test",
            name="Sphere",
            function_id=1,
            dimension=2,
            lb=[-5.0, 0.0],
            ub=[5.0, 10.0],
            optimum=([0.0, 0.0], 0.0),
        )
        assert problem([0.5, 0.0]) == 0.0
        assert problem([1.0, 1.0]) == 125.0
        np.testing.assert_allclose(problem.optimum.x, [0.5, 0.0])
        assert problem.meta_data.problem_id == 1
        assert problem.meta_data.name == "Sphere"

    def test_no_optimum(self) -> None:
        problem = wrap(sphere, suite="test", name="Sphere", function_id=1, dimension=2, lb=-1, ub=1)
        assert not np.isfinite(problem.optimum.y)

    def test_rejects_empty_box(self) -> None:
        with pytest.raises(ValueError):
            wrap(sphere, suite="test", name="Sphere", function_id=1, dimension=2, lb=1, ub=1)


class TestMakeInstance:
    def test_reproducible_from_seed(self) -> None:
        points = np.random.default_rng(1).random((10, 3))
        a, b = sphere_instance(seed=42), sphere_instance(seed=42)
        assert [a(u) for u in points] == [b(u) for u in points]
        np.testing.assert_array_equal(a.optimum.x, b.optimum.x)
        assert a.optimum.y == b.optimum.y

    @pytest.mark.parametrize(
        "other", [{"seed": 43}, {"seed": 42, "instance": 2}], ids=["seed", "instance"]
    )
    def test_differs_across_seeds_and_instances(self, other: dict) -> None:
        a, b = sphere_instance(seed=42), sphere_instance(**other)
        assert not np.allclose(a.optimum.x, b.optimum.x)

    @pytest.mark.parametrize("seed", range(20))
    def test_optimum_inside_domain(self, seed: int) -> None:
        problem = sphere_instance(seed=seed, dimension=5)
        x_opt = np.asarray(problem.optimum.x)
        assert np.all((x_opt >= 0.1) & (x_opt <= 0.9))
        assert problem(x_opt) == pytest.approx(problem.optimum.y)

    @pytest.mark.parametrize("function", [sphere, rastrigin])
    def test_optimum_is_minimum(self, function) -> None:
        problem = make_instance(
            function,
            suite="test",
            name=function.__name__,
            function_id=1,
            dimension=4,
            instance=1,
            lb=-5.12,
            ub=5.12,
            x_opt=np.zeros(4),
            seed=7,
        )
        points = np.random.default_rng(0).random((200, 4))
        assert min(problem(u) for u in points) > problem.optimum.y

    def test_rotation_preserves_sphere(self) -> None:
        problem = sphere_instance(seed=3, dimension=4)
        u = np.random.default_rng(2).random(4)
        x = from_unit(u, -5.0, 5.0)
        x_star = from_unit(problem.optimum.x, -5.0, 5.0)
        assert problem(u) == pytest.approx(sphere(x - x_star) + problem.optimum.y)

    def test_identity_when_disabled(self) -> None:
        problem = sphere_instance(seed=3, shift=False, rotate=False, offset=False)
        u = np.random.default_rng(2).random(3)
        assert problem(u) == pytest.approx(sphere(from_unit(u, -5.0, 5.0)))
        assert problem.optimum.y == 0.0

    def test_rejects_optimum_outside_domain(self) -> None:
        with pytest.raises(ValueError):
            make_instance(
                sphere,
                suite="test",
                name="Sphere",
                function_id=1,
                dimension=2,
                instance=1,
                lb=0.0,
                ub=1.0,
                x_opt=[2.0, 0.5],
            )


def test_callbacks_survive_garbage_collection() -> None:
    # Regression: ioh.wrap_problem keeps no references to its callbacks.
    problem = sphere_instance(seed=1)
    expected = problem([0.3, 0.6, 0.9])
    gc.collect()
    _ = [np.zeros(1000) for _ in range(1000)]
    assert problem([0.3, 0.6, 0.9]) == expected
