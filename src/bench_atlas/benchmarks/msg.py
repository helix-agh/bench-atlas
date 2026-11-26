"""Max-Set of Gaussians (MSG) suite."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import qmc

from bench_atlas.problems import Problem, ProblemId

__all__ = ["msg"]


def msg(dimensions: Iterable[int], instances: Iterable[int]) -> list[Problem]:
    """Max-Set of Gaussians problems (Gallagher & Yuan 2006) on [0, 1]^d, minimized.

    ``f(x) = -max_i w_i exp(-||x - c_i||^2 / (2 sigma_i^2))`` with the settings of
    Mizuta et al. 2026: m = 50 d Gaussians centred on the first m points of a
    scrambled Sobol' sequence, fixed per d; the instance seeds ``w_i ~ U[0, 1]`` and
    ``sigma_i ~ U[r/4, 3r]`` with ``r = (1/m)^(1/d)``. The optimum is the center
    with the largest weight: ``x_opt = c_i``, ``f_opt = -w_i``.
    """
    problems = []
    for d in dimensions:
        m = 50 * d
        sobol = qmc.Sobol(d, seed=np.random.default_rng(d))
        centers = sobol.random_base2(math.ceil(math.log2(m)))[:m]  # no warning for m != 2^k
        r = (1 / m) ** (1 / d)
        for instance in instances:
            rng = np.random.default_rng([instance, d])
            w, sigma = rng.uniform(0, 1, m), rng.uniform(r / 4, 3 * r, m)
            best = int(np.argmax(w))
            problems.append(
                Problem(
                    ProblemId("msg", 1, instance, d),
                    _MaxSetOfGaussians(centers, w, sigma),
                    x_opt=centers[best].copy(),
                    f_opt=float(-w[best]),
                )
            )
    return problems


@dataclass(frozen=True, eq=False)
class _MaxSetOfGaussians:
    """``-max_i w_i exp(-||x - c_i||^2 / (2 sigma_i^2))``."""

    centers: np.ndarray  # (m, d)
    w: np.ndarray  # (m,)
    sigma: np.ndarray  # (m,)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # cdist sums the squared differences, so the distance of a center to itself is 0.
        return -self._gaussians(x).max(axis=1)

    def _gaussians(self, x: np.ndarray) -> np.ndarray:
        return self.w * np.exp(-cdist(x, self.centers, "sqeuclidean") / (2 * self.sigma**2))

    def local_optima(self) -> np.ndarray:
        """The local minima: centers where no other Gaussian is at least as high."""
        g = self._gaussians(self.centers)
        others = np.where(np.eye(len(g), dtype=bool), -np.inf, g).max(axis=1)
        return self.centers[np.diag(g) > others]
