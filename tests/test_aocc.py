import math

import numpy as np
import pandas as pd
import pytest

from bench_atlas import aocc


def test_aocc_terms() -> None:
    assert aocc.aocc([1e-8, 1e2, 1e-3], 1e-8, 1e2) == pytest.approx(0.5)
    assert aocc.aocc([1e-12, 1e5], 1e-8, 1e2) == pytest.approx(0.5)
    assert aocc.aocc([0.0, -1.0], 1e-8, 1.0) == 1.0


def test_aocc_without_values() -> None:
    assert math.isnan(aocc.aocc([], 1e-8, 1.0))
    assert math.isnan(aocc.aocc([1.0, np.nan], 1e-8, 1.0))


def test_optimum_hit_at_first_evaluation() -> None:
    assert aocc.normalized_aocc([1], [3.0], 3.0, 10.0, 50) == 1.0


def test_no_progress() -> None:
    assert aocc.normalized_aocc([1], [10.0], 3.0, 10.0, 50) == 0.0
    assert aocc.normalized_aocc([1, 7], [20.0, 15.0], 3.0, 10.0, 50) == 0.0


def test_constant_error() -> None:
    assert aocc.normalized_aocc([1], [1e-4], 0.0, 1.0, 20) == pytest.approx(0.5)


def test_steps_and_prefixes() -> None:
    curve = ([1, 6], [1e-2, 1e-6])
    assert aocc.normalized_aocc(*curve, 0.0, 1.0, 10) == pytest.approx(0.5)
    assert aocc.normalized_aocc(*curve, 0.0, 1.0, 5) == pytest.approx(0.25)


def test_early_stop_keeps_the_last_best_value() -> None:
    stopped = aocc.normalized_aocc([1, 3], [1e-2, 1e-4], 0.0, 1.0, 10)
    errors = [1e-2, 1e-2] + [1e-4] * 8
    assert stopped == pytest.approx(aocc.aocc(errors, 1e-8, 1.0))


def test_error_below_the_optimum_counts_as_reached() -> None:
    assert aocc.normalized_aocc([1], [-1.0], 0.0, 1.0, 4) == 1.0


def test_degenerate_normalization() -> None:
    assert math.isnan(aocc.normalized_aocc([1], [1.0], 2.0, 2.0, 4))
    assert math.isnan(aocc.normalized_aocc([1], [1.0], 2.0, 1.0, 4))
    assert math.isnan(aocc.normalized_aocc([1], [1.0], math.nan, 1.0, 4))


def _run(problem: str, optimizer: str, f_opt: float, curve: tuple[list, list]) -> dict:
    return {
        "problem": problem,
        "suite": "test",
        "function": int(problem[1]),
        "instance": 1,
        "dimension": 2,
        "replicate": 0,
        "global_seed": 0,
        "seed": 7,
        "optimizer": optimizer,
        "budget_per_dimension": 200,
        "f_opt": f_opt,
        "best_f": curve[1][-1],
        "curve_evaluations": curve[0],
        "curve_f": curve[1],
    }


def _ela(problem: str, f_meds: list[float], points_per_dimension: int = 50) -> list[dict]:
    return [
        {
            "problem": problem,
            "replicate": r,
            "points_per_dimension": points_per_dimension,
            "f_med": f,
        }
        for r, f in enumerate(f_meds)
    ]


def test_table() -> None:
    runs = pd.DataFrame(
        [
            _run("p1", "A", 0.0, ([1, 201], [1e-2, 1e-6])),
            _run("p2", "A", math.nan, ([1], [5.0])),
            _run("p2", "B", math.nan, ([1, 10], [6.0, 4.0])),
        ]
    )
    ela = pd.DataFrame(_ela("p1", [1.0, 3.0, 2.0]) + _ela("p2", [104.0]) + _ela("p2", [9.0], 10))
    table = aocc.aocc_table(runs, ela, budgets_per_dimension=(100, 200, 300))
    assert list(table.columns[-6:]) == [
        "f_opt",
        "f_opt_known",
        "f_med",
        "aocc_100d",
        "aocc_200d",
        "aocc_300d",
    ]
    assert list(table["optimizer"]) == ["A", "A", "B"]
    assert list(table["f_med"]) == [2.0, 104.0, 104.0]
    assert list(table["f_opt"]) == [0.0, 4.0, 4.0]
    assert list(table["f_opt_known"]) == [True, False, False]
    e1, e2 = math.log10(1e-2 / 2.0), math.log10(1e-6 / 2.0)
    first, last = 1 - (e1 + 8) / 8, 1 - (e2 + 8) / 8
    assert table["aocc_100d"][0] == pytest.approx(first)
    assert table["aocc_200d"][0] == pytest.approx((200 * first + 200 * last) / 400)
    assert table["aocc_300d"].isna().all()
    assert table["aocc_200d"][2] > table["aocc_200d"][1]


def test_repeated_budgets_give_one_column() -> None:
    runs = pd.DataFrame([_run("p1", "A", 0.0, ([1], [1e-2]))])
    table = aocc.aocc_table(runs, pd.DataFrame(_ela("p1", [1.0])), (10, 5, 10))
    assert list(table.columns[-2:]) == ["aocc_10d", "aocc_5d"]
    assert table.columns.is_unique


def test_table_needs_ela() -> None:
    runs = pd.DataFrame([_run("p1", "A", 0.0, ([1], [1.0]))])
    with pytest.raises(ValueError, match=r"p1.*bench-atlas ela"):
        aocc.aocc_table(runs, pd.DataFrame(_ela("p1", [2.0], 10)))
    with pytest.raises(ValueError, match="p1"):
        aocc.aocc_table(runs, pd.DataFrame())
