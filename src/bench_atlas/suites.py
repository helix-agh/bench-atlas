"""Benchmark suites as lists of [`Problem`][bench_atlas.problems.Problem]."""

from __future__ import annotations

import functools
from collections.abc import Iterable
from dataclasses import dataclass

import ioh
import numpy as np
from iohclustering import BASELINE_K_DIMENTIONS, CLUSTER_BASELINE_DATASETS, get_dataset

from bench_atlas.gnbg import gnbg_i, gnbg_ii, gnbg_iii
from bench_atlas.portal import portal
from bench_atlas.problems import Problem, ProblemId, from_ioh
from bench_atlas.zigzag import zigzag

__all__ = [
    "bbob",
    "cec2022",
    "gnbg_i",
    "gnbg_ii",
    "gnbg_iii",
    "iohclustering",
    "ma_bbob",
    "portal",
    "zigzag",
]


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


def cec2022(dimensions: Iterable[int]) -> list[Problem]:
    """CEC 2022 functions F1-F12 for d in {10, 20}, as implemented in IOH.

    IOH problem ids are 1000 + F. Caveat: for F3 (Schaffer F7), F5 (Levy) and F9
    (composition 1), the official CEC 2022 code (used in the competition) differs
    from its technical report, and ioh 0.3.22 follows the report: it scales F3 by
    0.5/100 and F5 by 5.12/100 instead of 1, and does not rotate the second
    component of F9. These three are therefore not the functions of the
    competition; the other nine match the official code.

    The suite is defined only for d = 10 and 20. There is one shift and rotation
    per function and d, as in the official code: IOH ignores the instance number,
    so the instance is 1.
    """
    dimensions = list(dimensions)
    if unsupported := sorted(set(dimensions) - {10, 20}):
        raise ValueError(f"CEC 2022 has no dimensions {unsupported}; available: [10, 20]")
    return [
        from_ioh(ioh.get_problem(1000 + f, 1, dimension, ioh.ProblemClass.CEC2022), "cec2022")
        for f in range(1, 13)
        for dimension in dimensions
    ]


def iohclustering() -> list[Problem]:
    """The 40 IOHClustering problems: 10 datasets x k = 2, 3, 5, 10 clusters.

    The datasets ship with the pinned ``iohclustering`` package (2 features each). A
    point holds the k centroids on the min-max normalized data, so d = 2k; the
    objective is the k-means MSE. Ids are ``iohclustering_f{dataset}_i1_d{2k}`` with
    IOHClustering's dataset numbering 1-10. No optimum is known: ``f_opt`` is None.
    """
    problems = []
    for dataset_id, name in CLUSTER_BASELINE_DATASETS.items():
        data = get_dataset(name)
        low, high = data.min(axis=0), data.max(axis=0)
        normalized = np.ascontiguousarray((data - low) / (high - low))
        for k in BASELINE_K_DIMENTIONS[name]:
            problems.append(
                Problem(
                    ProblemId("iohclustering", dataset_id, 1, k * data.shape[1]),
                    _ClusteringMSE(normalized),
                    name=f"{name}_k{k}",
                )
            )
    return problems


@dataclass(frozen=True, eq=False)
class _ClusteringMSE:
    """Mean squared distance of ``data`` to the nearest of the centroids in ``x``."""

    data: np.ndarray

    def __call__(self, x: np.ndarray) -> np.ndarray:
        centroids = x.reshape(len(x), -1, self.data.shape[1])
        squared = ((self.data[None, :, None, :] - centroids[:, None]) ** 2).sum(axis=-1)
        return squared.min(axis=-1).mean(axis=-1)
