"""PORTAL suite: the curated benchmark S50 of the PORTAL landscape generator.

PORTAL builds problems from JSON recipes that set each component's centre,
depth, conditioning, exponents, rotation and ruggedness transforms, and how
components and blocks of variables are combined. ``Benchmark_S50`` is the
authors' curated suite of 50 recipes (d = 30) from four structural families:
single component, single component with transforms, several components, and
several blocks. The recipes are downloaded once into ``CACHE`` at a pinned
commit; the objective is PORTAL's own code, vendored unchanged in
:mod:`bench_atlas._portal` (MIT).

Reference: D. Yazdani et al., "PORTAL: Controllable Landscape Generator for
Continuous Optimization - Part I: Framework", arXiv:2512.00288, 2025.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bench_atlas._portal.fitness import fitness
from bench_atlas._portal.load_instance import load_instance_payload
from bench_atlas._portal.optimum import global_optimum_value
from bench_atlas.problems import Problem, ProblemId, to_unit

__all__ = ["portal"]

CACHE = Path.home() / ".cache" / "bench-atlas" / "portal"
COMMIT = "d4027ec6e0ac233b86c0bd1617181eb130e60c96"  # tag v2.0.0
URL = f"https://raw.githubusercontent.com/EvoMindLab/PORTAL/{COMMIT}/Benchmark_S50/{{}}.json"
# The recipes of Benchmark_S50 at COMMIT, by family; files f{family:02}_r{recipe:03}.json.
S50_RECIPES = {
    1: (7, 8, 18, 21, 22, 26, 29, 31),
    2: (5, 13, 20, 35, 38, 43, 67, 68, 70, 72, 98),
    3: (4, 20, 24, 25, 26, 27, 31, 55, 58, 81, 111),
    4: (5, 7, 10, 11, 13, 21, 23, 29, 43, 44, 50, 67, 68, 82, 84, 85, 90, 93, 96, 98),
}
S50 = [f"f{f:02}_r{r:03}" for f, recipes in S50_RECIPES.items() for r in recipes]


def portal() -> list[Problem]:
    """The 50 recipes of PORTAL's ``Benchmark_S50`` (d = 30).

    Ids are ``portal_f{family}_i{recipe}_d30``, the recipe's name is in ``name``.
    The domain is [-100, 100]^d. ``f_opt`` is PORTAL's analytic global minimum;
    ``x_opt`` is the centre of the deepest component of every block, where
    PORTAL's components take their minimum.
    """
    folder = _download()
    problems = []
    for recipe in S50:
        raw = json.loads((folder / f"{recipe}.json").read_text())
        instance = load_instance_payload(raw)
        instance["warn_out_of_bounds"] = False  # PORTAL prints a warning per evaluation
        lb, ub = float(instance["bounds_min"]), float(instance["bounds_max"])
        x_opt = _optimum(instance)
        f_opt = global_optimum_value(instance)
        family, number = (int(part[1:]) for part in recipe.split("_"))
        problems.append(
            Problem(
                id=ProblemId("portal", family, number, int(instance["d"])),
                function=_Portal(instance, x_opt, to_unit(x_opt, lb, ub), ub - lb),
                name=f"{recipe}_{raw['_suite']['name']}",
                x_opt=to_unit(x_opt, lb, ub),
                f_opt=f_opt,
            )
        )
    return problems


def _download() -> Path:
    """Download the recipes of ``S50`` into ``CACHE`` (once)."""
    if not CACHE.exists():
        sources = {}
        for recipe in S50:
            with urllib.request.urlopen(URL.format(recipe)) as response:
                sources[recipe] = response.read()
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        partial = Path(tempfile.mkdtemp(dir=CACHE.parent))  # one per caller
        for recipe, source in sources.items():
            (partial / f"{recipe}.json").write_bytes(source)
        try:
            partial.rename(CACHE)
        except OSError:  # another process published the cache first
            shutil.rmtree(partial)
    return CACHE


def _optimum(instance: dict[str, Any]) -> np.ndarray:
    """Centre of the deepest (lowest ``beta``) component of every block."""
    beta = np.asarray(instance["P"]["beta"], dtype=float).reshape(-1)
    centres = np.asarray(instance["P"]["c"], dtype=float)
    d = int(instance["d"])
    blocks = instance.get("_schema_blocks") or [{"K": instance["K"], "dim_indices": range(d)}]
    x = np.empty(d)
    first = 0
    for block in blocks:
        k = first + int(np.argmin(beta[first : first + int(block["K"])]))
        dims = list(block["dim_indices"])
        x[dims] = centres[k, dims]
        first += int(block["K"])
    return x


@dataclass(frozen=True, eq=False)
class _Portal:
    """PORTAL's scalar ``fitness`` on [0, 1]^d, one point at a time.

    PORTAL's batch evaluator differs from ``fitness`` in the last bits and is
    implemented only for recipes without transforms, so every point goes
    through ``fitness``: a point's value does not depend on its batch. Points
    are mapped as ``x_opt + (u - u_opt) * width`` (equal to ``lb + u * width``
    up to rounding), so that ``u_opt`` maps exactly to ``x_opt`` and
    ``f(x_opt) == f_opt``: the optima are sharp enough that a rounding error of
    1e-14 in x changes f by up to 2e-4.
    """

    instance: dict[str, Any]
    x_opt: np.ndarray  # (d,), original coordinates
    u_opt: np.ndarray  # (d,), x_opt on [0, 1]^d
    width: float

    def __call__(self, u: np.ndarray) -> np.ndarray:
        x = self.x_opt + (u - self.u_opt) * self.width
        return np.array([fitness(row, self.instance) for row in x])
