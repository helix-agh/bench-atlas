"""GNBG suites: GNBG-I, GNBG-II and the static problems of GNBG-III.

The problem files are downloaded once from the official repositories (pinned
commits) into ``CACHE``; their code and data are not redistributed here. The
objective is our own vectorized implementation of GNBG, checked against the
official code in the tests.

Reference: D. Yazdani, M. N. Omidvar, D. Yazdani, K. Deb and A. H. Gandomi,
"GNBG: A Generalized and Configurable Benchmark Generator for Continuous
Numerical Optimization", arXiv:2312.07083, 2023.
"""

from __future__ import annotations

import io
import re
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from bench_atlas.problems import Problem, ProblemId, to_unit

__all__ = ["gnbg_i", "gnbg_ii", "gnbg_iii"]

CACHE = Path.home() / ".cache" / "bench-atlas"
GNBG_I_URL = (
    "https://raw.githubusercontent.com/rohitsalgotra/GNBG-I/"
    "35dad28f53add8cf726932e929b7da2d0cf184f6/GNBG_Instances.MATLAB.zip"
)
GNBG_II_URL = (
    "https://raw.githubusercontent.com/rohitsalgotra/GNBG-II/"
    "59d506ed0b317082950d48dd75f21ed2f02aed62/GNBG_Instances.Python-main.zip"
)
GNBG_III_URL = (
    "https://raw.githubusercontent.com/rohitsalgotra/GNBG-III/"
    "b263a6520b68e2f46735235c2c81ed0778edb58d/GNBG_III_Python.zip"
)


def gnbg_i() -> list[Problem]:
    """The 24 GNBG-I problems f1-f24 of the 2024 GECCO competition.

    All problems have d = 30 and a single instance. This is the corrected
    version from December 2024: in the original files (November 2023), f20
    records a wrong optimum and all five components of f24 coincide.
    """
    files = _download(GNBG_I_URL, CACHE / "gnbg_i", re.compile(r"f(\d+)\.mat"))
    return [_gnbg(files[number], "gnbg_i", number) for number in range(1, 25)]


def gnbg_ii() -> list[Problem]:
    """The 24 GNBG-II problems f1-f24 of the 2025 CEC and GECCO competitions.

    All problems have d = 30 and a single instance. Suite reference:
    A. H. Gandomi, D. Yazdani, M. N. Omidvar and K. Deb, "GNBG-Generated Test
    Suite for Box-Constrained Numerical Global Optimization", arXiv:2312.07034, 2023.
    """
    files = _download(GNBG_II_URL, CACHE / "gnbg_ii", re.compile(r"f(\d+)\.mat"))
    return [_gnbg(files[number], "gnbg_ii", number) for number in range(1, 25)]


def gnbg_iii() -> list[Problem]:
    """The 22 static GNBG-III problems F1-F22 (d = 30, a single instance).

    F23 (noisy) and F24 (dynamic) are left out: they are not deterministic
    functions.
    """
    files = _download(GNBG_III_URL, CACHE / "gnbg_iii", re.compile(r"F(\d+)_\w+\.mat"))
    return [_gnbg(files[number], "gnbg_iii", number) for number in range(1, 23)]


def _download(url: str, cache: Path, pattern: re.Pattern[str]) -> dict[int, Path]:
    """Extract the files matching ``pattern`` from the zip at ``url`` into ``cache``.

    Returns the extracted files by the number captured by ``pattern``.
    """
    if not cache.exists():
        with urllib.request.urlopen(url) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
        cache.parent.mkdir(parents=True, exist_ok=True)
        partial = Path(tempfile.mkdtemp(dir=cache.parent))  # one per caller
        for member in archive.namelist():
            if pattern.fullmatch(Path(member).name):
                (partial / Path(member).name).write_bytes(archive.read(member))
        try:
            partial.rename(cache)
        except OSError:  # another process published the cache first
            shutil.rmtree(partial)
    return {
        int(match[1]): path for path in cache.iterdir() if (match := pattern.fullmatch(path.name))
    }


def _gnbg(path: Path, suite: str, number: int) -> Problem:
    g = loadmat(path, squeeze_me=True)["GNBG"][()]
    o, d = int(g["o"]), int(g["Dimension"])
    lb, ub = float(g["MinCoordinate"]), float(g["MaxCoordinate"])
    function = _GNBG(
        position=to_unit(_array(g, "Component_MinimumPosition", o, d), lb, ub),
        width=ub - lb,
        rotation=np.ascontiguousarray(np.moveaxis(_array(g, "RotationMatrix", d, d, o), 2, 0)),
        sigma=_array(g, "ComponentSigma", o),
        h=_array(g, "Component_H", o, d),
        mu=_array(g, "Mu", o, 2),
        omega=_array(g, "Omega", o, 4),
        lam=_array(g, "lambda", o),
    )
    return Problem(
        id=ProblemId(suite, number, 1, d),
        function=function,
        name=path.stem.partition("_")[2],
        x_opt=to_unit(_array(g, "OptimumPosition", d), lb, ub),
        f_opt=float(g["OptimumValue"]),
    )


def _array(g: np.void, key: str, *shape: int) -> np.ndarray:
    # C-contiguous, so that results stay bitwise identical after pickling.
    return np.ascontiguousarray(np.asarray(g[key], dtype=float).reshape(shape))


@dataclass(frozen=True, eq=False)
class _GNBG:
    """Vectorized GNBG objective on [0, 1]^d: the minimum over components k of
    ``sigma_k + (sum_i h_ki T_k(z_i)^2) ** lam_k`` with ``z = R_k (x - m_k)``.

    ``x - m_k`` is computed as ``(u - u_k) * width`` in unit coordinates, so
    that the optimum, which is the minimum of one component, gives exactly
    ``f_opt``: with exponents down to 0.05, mapping ``u`` back to the box first
    turns rounding errors of 1e-14 into errors of up to 0.1 in f.
    """

    position: np.ndarray  # (o, d), on [0, 1]^d
    width: float  # ub - lb of the original box
    rotation: np.ndarray  # (o, d, d)
    sigma: np.ndarray  # (o,)
    h: np.ndarray  # (o, d)
    mu: np.ndarray  # (o, 2)
    omega: np.ndarray  # (o, 4)
    lam: np.ndarray  # (o,)

    def __call__(self, u: np.ndarray) -> np.ndarray:
        if not np.all(np.isfinite(u)):
            raise ValueError("GNBG points must contain only finite coordinates")
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
                sigma
                + np.sum(h * _transform((u - m) * self.width @ r.T, mu, omega) ** 2, axis=1) ** lam
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
