"""Benchmark suites as lists of :class:`~bench_atlas.problems.Problem`."""

from __future__ import annotations

import functools
import io
import re
import urllib.request
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import ioh
import numpy as np
from scipy.io import loadmat

from bench_atlas.problems import Problem, ProblemId, from_function, from_ioh

__all__ = ["bbob", "gnbg_iii", "ma_bbob"]

GNBG_III_URL = (
    "https://raw.githubusercontent.com/rohitsalgotra/GNBG-III/"
    "b263a6520b68e2f46735235c2c81ed0778edb58d/GNBG_III_Python.zip"
)
GNBG_III_CACHE = Path.home() / ".cache" / "bench-atlas" / "gnbg_iii"
_GNBG_III_FILE = re.compile(r"F(\d+)_(\w+)\.mat")


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


def gnbg_iii() -> list[Problem]:
    """The 22 static GNBG-III problems F1-F22 (d = 30, a single instance).

    The problem files are downloaded once from the official GNBG-III repository
    (pinned commit) into ``GNBG_III_CACHE``. F23 (noisy) and F24 (dynamic) are
    left out: they are not deterministic functions.

    Reference: D. Yazdani, M. N. Omidvar, D. Yazdani, K. Deb and A. H. Gandomi,
    "GNBG: A Generalized and Configurable Benchmark Generator for Continuous
    Numerical Optimization", arXiv:2312.07083, 2023.
    """
    problems = []
    for path in _gnbg_iii_files():
        number, name = _GNBG_III_FILE.fullmatch(path.name).groups()
        if int(number) > 22:
            continue
        g = loadmat(path, squeeze_me=True)["GNBG"][()]
        o, d = int(g["o"]), int(g["Dimension"])
        function = _GNBG(
            position=_array(g, "Component_MinimumPosition", o, d),
            rotation=np.ascontiguousarray(np.moveaxis(_array(g, "RotationMatrix", d, d, o), 2, 0)),
            sigma=_array(g, "ComponentSigma", o),
            h=_array(g, "Component_H", o, d),
            mu=_array(g, "Mu", o, 2),
            omega=_array(g, "Omega", o, 4),
            lam=_array(g, "lambda", o),
        )
        problems.append(
            from_function(
                function,
                id=ProblemId("gnbg_iii", int(number), 1, d),
                lb=float(g["MinCoordinate"]),
                ub=float(g["MaxCoordinate"]),
                name=name,
                x_opt=np.asarray(g["OptimumPosition"], dtype=float).reshape(d),
                f_opt=float(g["OptimumValue"]),
            )
        )
    return sorted(problems, key=lambda problem: problem.id)


def _array(g: np.void, key: str, *shape: int) -> np.ndarray:
    # C-contiguous, so that results stay bitwise identical after pickling.
    return np.ascontiguousarray(np.asarray(g[key], dtype=float).reshape(shape))


def _gnbg_iii_files() -> list[Path]:
    if not GNBG_III_CACHE.exists():
        with urllib.request.urlopen(GNBG_III_URL) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
        partial = GNBG_III_CACHE.with_name(GNBG_III_CACHE.name + ".partial")
        partial.mkdir(parents=True, exist_ok=True)
        for member in archive.namelist():
            if _GNBG_III_FILE.fullmatch(Path(member).name):
                (partial / Path(member).name).write_bytes(archive.read(member))
        partial.rename(GNBG_III_CACHE)
    return sorted(GNBG_III_CACHE.glob("F*.mat"))


@dataclass(frozen=True, eq=False)
class _GNBG:
    """Vectorized GNBG objective: the minimum over components k of
    ``sigma_k + (sum_i h_ki T_k(z_i)^2) ** lam_k`` with ``z = R_k (x - m_k)``.
    """

    position: np.ndarray  # (o, d)
    rotation: np.ndarray  # (o, d, d)
    sigma: np.ndarray  # (o,)
    h: np.ndarray  # (o, d)
    mu: np.ndarray  # (o, 2)
    omega: np.ndarray  # (o, 4)
    lam: np.ndarray  # (o,)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        components = zip(
            self.position,
            self.rotation,
            self.sigma,
            self.h,
            self.mu,
            self.omega,
            self.lam,
            strict=True,
        )
        return np.min(
            [
                sigma + np.sum(h * _transform((x - m) @ r.T, mu, omega) ** 2, axis=1) ** lam
                for m, r, sigma, h, mu, omega, lam in components
            ],
            axis=0,
        )


def _transform(z: np.ndarray, mu: np.ndarray, omega: np.ndarray) -> np.ndarray:
    # GNBG's non-linear transformation T; asymmetric for positive and negative z.
    with np.errstate(divide="ignore", invalid="ignore"):
        log = np.log(np.abs(z))
        positive = np.exp(log + mu[0] * (np.sin(omega[0] * log) + np.sin(omega[1] * log)))
        negative = -np.exp(log + mu[1] * (np.sin(omega[2] * log) + np.sin(omega[3] * log)))
    return np.where(z > 0, positive, np.where(z < 0, negative, 0.0))
