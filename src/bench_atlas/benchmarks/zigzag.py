"""Zigzag suite: the four zigzag functions F1-F4 of Kůdela and Matoušek.

The shifts and matrices are read from the official MATLAB files (pinned commit),
downloaded once into ``CACHE``; the repository has no license, so its files are
not redistributed here. The objective is our own vectorized implementation,
checked against the official code in the tests.

Reference: J. Kůdela and R. Matoušek, "New Benchmark Functions for
Single-Objective Optimization Based on a Zigzag Pattern", IEEE Access 10,
8262-8278, 2022, doi:10.1109/ACCESS.2022.3144067.
"""

from __future__ import annotations

import re
import shutil
import tempfile
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bench_atlas.problems import Problem, ProblemId, to_unit

__all__ = ["zigzag"]

CACHE = Path.home() / ".cache" / "bench-atlas" / "zigzag"
URL = (
    "https://raw.githubusercontent.com/JakubKudela89/Zigzag/"
    "a7ce2c260b3a46a3d67fa097f8e9544de4fbf5f6/journal%20paper/F{}.m"
)
DIMENSIONS = (2, 5, 10, 15, 20)  # the dimensions with a shift and matrix in the MATLAB code
# Table 1 of the paper: parametrization p = 1, ..., 100 runs over k fastest, then lambda, then m.
K = (1, 2, 4, 8, 16)
LAMBDA = (0.01, 0.1, 0.5, 0.9, 0.99)
M = (0.1, 0.5, 0.9, 1.0)


def zigzag(dimensions: Iterable[int], parametrizations: Iterable[int]) -> list[Problem]:
    """Zigzag functions F1-F4 for the given dimensions and parametrizations.

    Parametrizations are the IDs 1-100 of Table 1 of the paper, the grid
    k in {1, 2, 4, 8, 16} x lambda in {0.01, 0.1, 0.5, 0.9, 0.99} x
    m in {0.1, 0.5, 0.9, 1}; the problem id is ``zigzag_f{F}_i{ID}_d{d}``. Only
    d in {2, 5, 10, 15, 20} have a shift and matrix in the official code. The
    domain is [-100, 100]^d and the optimum is f = 0 at x = -s.
    """
    dimensions, parametrizations = list(dimensions), list(parametrizations)
    if unsupported := sorted(set(dimensions) - set(DIMENSIONS)):
        raise ValueError(f"Zigzag has no dimensions {unsupported}; available: {DIMENSIONS}")
    if invalid := sorted(set(parametrizations) - set(range(1, 101))):
        raise ValueError(f"Zigzag parametrizations are 1-100, got {invalid}")
    problems = []
    for function in range(1, 5):
        transforms = _transforms(_download() / f"F{function}.m")
        for p in parametrizations:
            k, lam, m = K[(p - 1) % 5], LAMBDA[(p - 1) // 5 % 5], M[(p - 1) // 25]
            for d in dimensions:
                matrix, shift = transforms[d]
                optimum = to_unit(-shift, -100.0, 100.0)
                problems.append(
                    Problem(
                        id=ProblemId("zigzag", function, p, d),
                        function=_Zigzag(function, k, m, lam, optimum, matrix),
                        name=f"F{function}_k{k}_m{m}_lambda{lam}",
                        x_opt=optimum,
                        f_opt=0.0,
                    )
                )
    return problems


def _download() -> Path:
    """Download F1.m-F4.m of the journal paper into ``CACHE`` (once)."""
    if not CACHE.exists():
        sources = []
        for function in range(1, 5):
            with urllib.request.urlopen(URL.format(function)) as response:
                sources.append(response.read())
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        partial = Path(tempfile.mkdtemp(dir=CACHE.parent))  # one per caller
        for function, source in enumerate(sources, start=1):
            (partial / f"F{function}.m").write_bytes(source)
        try:
            partial.rename(CACHE)
        except OSError:  # another process published the cache first
            shutil.rmtree(partial)
    return CACHE


def _transforms(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """The matrix ``M`` and shift ``s`` of every dimension in a MATLAB file F*.m."""
    pattern = re.compile(r"d == (\d+)\s+M\s*=\s*\[([^\]]*)\];\s*s\s*=\s*\[([^\]]*)\]")
    transforms = {}
    for d, matrix, shift in pattern.findall(path.read_text()):
        rows = [[float(v) for v in re.split(r"[,\s]+", row.strip())] for row in matrix.split(";")]
        s = np.array([float(v) for v in re.split(r"[,\s]+", shift.strip())])
        transforms[int(d)] = (np.ascontiguousarray(rows), s)
    if sorted(transforms) != list(DIMENSIONS):
        raise ValueError(f"{path}: expected dimensions {DIMENSIONS}, parsed {sorted(transforms)}")
    return transforms


@dataclass(frozen=True, eq=False)
class _Zigzag:
    """Vectorized zigzag function on [0, 1]^d: ``sum_i phi(y_i)`` with
    ``y = (x + s) M`` as in the official code (the paper writes ``M (x - s)``).

    ``x + s`` is computed as ``(u - u_opt) * 200`` in unit coordinates, so that
    ``f(x_opt) == 0`` exactly.
    """

    function: int  # 1-4
    k: float
    m: float
    lam: float
    optimum: np.ndarray  # (d,), u of x = -s
    matrix: np.ndarray  # (d, d)

    def __call__(self, u: np.ndarray) -> np.ndarray:
        y = ((u - self.optimum) * 200.0) @ self.matrix
        phi = _phi1 if self.function in (1, 2) else _phi3
        y = phi(y, self.k, self.m, self.lam)
        if self.function in (2, 4):
            y = phi(y, self.k, self.m, self.lam)
        return np.sum(y, axis=1)


# The 1-D functions and the zigzag, written in the same order of operations as the MATLAB code.
def _phi1(x: np.ndarray, k: float, m: float, lam: float) -> np.ndarray:
    poly = np.abs((x - 40) * (x - 185) * x * (x + 50) * (x + 180))
    return 10 * np.abs(np.sin(0.1 * x)) + 3 * 10.0**-9 * _zigzag(x, k, m, lam) * poly


def _phi3(x: np.ndarray, k: float, m: float, lam: float) -> np.ndarray:
    log = np.abs(np.log(np.abs(x) * 1000 + 1))
    return _zigzag(x, k, m, lam) * 3 * log + 30 - 30 * np.abs(np.cos(x / (np.pi * 10)))


def _zigzag(x: np.ndarray, k: float, m: float, lam: float) -> np.ndarray:
    x = np.abs(x)
    x = x / k - np.floor(x / k)
    ids = x <= lam
    return 1 - m + ids * m * (x / lam) + (1 - ids) * m * (1 - (x - lam) / (1 - lam))
