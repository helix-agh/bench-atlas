"""IOHClustering suite: k-means problems on the datasets of the ``iohclustering`` package."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from iohclustering import BASELINE_K_DIMENTIONS, CLUSTER_BASELINE_DATASETS, get_dataset

from bench_atlas.problems import Problem, ProblemId

__all__ = ["iohclustering"]


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
