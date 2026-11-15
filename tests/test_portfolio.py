import functools
from pathlib import Path

import nevergrad as ng
import numpy as np
import pandas as pd
import pytest

from bench_atlas import executor, portfolio
from bench_atlas.problems import Problem, ProblemId
from bench_atlas.suites import bbob


def _sphere(x: np.ndarray) -> np.ndarray:
    return np.sum((x - 0.3) ** 2, axis=1)


def _problem(d: int = 2) -> Problem:
    return Problem(ProblemId("test", 1, 1, d), _sphere)


class _StopAfter:
    def __init__(self, n: int) -> None:
        self.n = n

    def __call__(self, parametrization, budget, num_workers):
        optimizer = ng.optimizers.registry["RandomSearch"](parametrization, budget, num_workers)
        ask = optimizer.ask

        def limited_ask():
            if optimizer.num_ask >= self.n:
                raise ng.errors.NevergradEarlyStopping
            return ask()

        optimizer.ask = limited_ask
        return optimizer


@pytest.mark.parametrize("name", list(portfolio.PORTFOLIO))
def test_portfolio_run(name: str) -> None:
    problem = _problem()
    result = portfolio.optimize(problem, 3, name, budget_per_dimension=50)
    assert result["evaluations"] == problem.evaluations == 100
    curve_evaluations, curve_f = result["curve_evaluations"], result["curve_f"]
    assert curve_evaluations[0] == 1
    assert np.all(np.diff(curve_evaluations) > 0)
    assert np.all(np.diff(curve_f) < 0)
    assert curve_evaluations[-1] <= 100
    assert result["best_f"] == curve_f[-1]


@pytest.mark.parametrize("name", list(portfolio.PORTFOLIO))
def test_runs_depend_only_on_the_seed(name: str) -> None:
    np.random.seed(1)
    first = portfolio.optimize(_problem(), 3, name, budget_per_dimension=50)
    np.random.seed(2)
    again = portfolio.optimize(_problem(), 3, name, budget_per_dimension=50)
    other = portfolio.optimize(_problem(), 4, name, budget_per_dimension=50)
    assert first == again
    assert first != other


def test_portfolio_names_are_stable() -> None:
    assert list(portfolio.PORTFOLIO) == [
        "CMAstd",
        "DiagonalCMA",
        "TwoPointsDE",
        "PSO",
        "RNelderMead",
        "LBFGSB",
        "OnePlusOne",
        "RandomSearch",
    ]


def test_optimizer_class() -> None:
    assert portfolio.optimizer_class("CMAstd") is ng.optimizers.registry["CMAstd"]
    assert portfolio.optimizer_class("NelderMead") is ng.optimizers.registry["NelderMead"]
    with pytest.raises(KeyError):
        portfolio.optimizer_class("NoSuchOptimizer")


def test_early_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(portfolio.PORTFOLIO, "Stop", _StopAfter(7))
    problem = _problem()
    result = portfolio.optimize(problem, 0, "Stop", budget_per_dimension=50)
    assert result["evaluations"] == problem.evaluations == 7
    curve = portfolio.best_so_far(result["curve_evaluations"], result["curve_f"], 100)
    assert len(curve) == 100
    assert np.all(curve[6:] == result["best_f"])


def test_best_so_far() -> None:
    curve = portfolio.best_so_far([1, 3, 4], [5.0, 2.0, 1.0], 6)
    np.testing.assert_array_equal(curve, [5.0, 5.0, 2.0, 1.0, 1.0, 1.0])
    np.testing.assert_array_equal(portfolio.best_so_far([1, 3, 4], [5.0, 2.0, 1.0], 2), [5.0, 5.0])


def test_best_so_far_edge_cases() -> None:
    np.testing.assert_array_equal(portfolio.best_so_far([1], [0.0], 4), [0.0] * 4)
    np.testing.assert_array_equal(portfolio.best_so_far([2], [1.0], 3), [np.nan, 1.0, 1.0])
    assert np.isnan(portfolio.best_so_far([], [], 3)).all()


def test_portfolio_does_not_depend_on_n_jobs(tmp_path: Path) -> None:
    problems = bbob([2], [1])[:3]
    compute = functools.partial(portfolio.optimize, budget_per_dimension=20)
    variants = [{"optimizer": name} for name in portfolio.PORTFOLIO]
    serial, parallel = (
        executor.run(
            "runs",
            problems,
            range(2),
            compute,
            {"budget_per_dimension": 20},
            tmp_path / str(n_jobs),
            n_jobs=n_jobs,
            variants=variants,
        )
        for n_jobs in (1, 4)
    )
    assert len(serial) == 3 * 2 * len(portfolio.PORTFOLIO)
    pd.testing.assert_frame_equal(serial, parallel, check_exact=True)
