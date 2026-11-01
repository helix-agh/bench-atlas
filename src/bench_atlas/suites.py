"""Benchmark suites as lists of :class:`~bench_atlas.problems.Problem`."""

from __future__ import annotations

import functools
from collections.abc import Iterable

import ioh

from bench_atlas.problems import Problem, from_ioh

__all__ = ["bbob", "ma_bbob"]


def bbob(dimensions: Iterable[int], instances: Iterable[int]) -> list[Problem]:
    """All 24 BBOB functions for the given dimensions and instances (from IOH)."""
    return [
        from_ioh(ioh.get_problem(function, instance, dimension, ioh.ProblemClass.BBOB), "bbob")
        for function in range(1, 25)
        for instance in instances
        for dimension in dimensions
    ]


def ma_bbob(dimensions: Iterable[int], instances: Iterable[int]) -> list[Problem]:
    """MA-BBOB problems for the given dimensions and instances (from IOH).

    IOH derives the weights, the BBOB instances and the optimum of each
    many-affine problem from its instance number.
    """
    return [
        from_ioh(
            ioh.problem.ManyAffine(instance, dimension),
            "ma_bbob",
            factory=functools.partial(ioh.problem.ManyAffine, instance, dimension),
        )
        for instance in instances
        for dimension in dimensions
    ]
