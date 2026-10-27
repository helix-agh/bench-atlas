"""IOH-based problem interface.

Every problem in bench-atlas is an ``ioh`` real-valued single-objective problem.
This module only adds what IOH does not provide out of the box:

- a stable :class:`ProblemId` (IOH meta data does not know which suite a problem
  belongs to),
- normalization of the search domain to [0, 1]^d,
- BBOB-style instances for problems without native instances.

See ``docs/ioh-deviations.md`` for the rationale behind each addition.
"""

from __future__ import annotations

import re
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING

import ioh
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import ArrayLike

__all__ = [
    "ProblemId",
    "from_unit",
    "make_instance",
    "normalize",
    "problem_id",
    "to_unit",
    "wrap",
]

Problem = ioh.iohcpp.problem.RealSingleObjective

_ID_PATTERN = re.compile(
    r"^(?P<suite>.+)_f(?P<function>\d+)_i(?P<instance>\d+)_d(?P<dimension>\d+)$"
)


@dataclass(frozen=True, order=True)
class ProblemId:
    """Stable identifier of a benchmark problem: (suite, function, instance, d)."""

    suite: str
    function: int
    instance: int
    dimension: int

    def __str__(self) -> str:
        return f"{self.suite}_f{self.function}_i{self.instance}_d{self.dimension}"

    @classmethod
    def parse(cls, value: str) -> ProblemId:
        """Inverse of ``str(problem_id)``."""
        match = _ID_PATTERN.match(value)
        if match is None:
            raise ValueError(f"Invalid problem id: {value!r}")
        return cls(
            suite=match["suite"],
            function=int(match["function"]),
            instance=int(match["instance"]),
            dimension=int(match["dimension"]),
        )


def problem_id(problem: Problem, suite: str) -> ProblemId:
    """Build the :class:`ProblemId` of an IOH problem from its meta data."""
    meta = problem.meta_data
    return ProblemId(
        suite=suite,
        function=meta.problem_id,
        instance=meta.instance,
        dimension=meta.n_variables,
    )


def to_unit(x: ArrayLike, lb: ArrayLike, ub: ArrayLike) -> np.ndarray:
    """Map points from the box [lb, ub] to [0, 1]^d."""
    lb, ub = np.asarray(lb, dtype=float), np.asarray(ub, dtype=float)
    return (np.asarray(x, dtype=float) - lb) / (ub - lb)


def from_unit(u: ArrayLike, lb: ArrayLike, ub: ArrayLike) -> np.ndarray:
    """Map points from [0, 1]^d to the box [lb, ub]."""
    lb, ub = np.asarray(lb, dtype=float), np.asarray(ub, dtype=float)
    return lb + np.asarray(u, dtype=float) * (ub - lb)


def _registry_name(suite: str, name: str, function_id: int, instance: int, dimension: int) -> str:
    # ``ioh.wrap_problem`` registers problems globally by name. A repeated name
    # only replaces the registry entry; already created problem objects keep
    # their own function.
    return f"{suite}/{name}/f{function_id}/i{instance}/d{dimension}"


def _keep_alive(problem: Problem, *objects: object) -> None:
    # ``ioh.wrap_problem`` does not hold references to the Python callbacks it
    # is given (ioh 0.3.22): once they are garbage collected, calling the problem
    # segfaults. Tie their lifetime to the problem object instead.
    weakref.finalize(problem, lambda *_: None, *objects)


def wrap(
    function: Callable[[np.ndarray], float],
    *,
    suite: str,
    name: str,
    function_id: int,
    dimension: int,
    lb: ArrayLike,
    ub: ArrayLike,
    instance: int = 1,
    optimum: tuple[ArrayLike, float] | None = None,
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
    objective_offset: float = 0.0,
) -> Problem:
    """Wrap a callable defined on [lb, ub] as an IOH problem on [0, 1]^d.

    The domain mapping and the optional ``transform`` / ``objective_offset`` are
    applied through IOH's ``transform_variables`` / ``transform_objectives`` hooks.

    Args:
        function: Objective on the original domain, ``f(x) -> float``.
        suite: Suite name, used only for the IOH registry name.
        name: Problem name stored in IOH meta data.
        function_id: Function number stored in IOH meta data.
        dimension: Number of variables.
        lb: Lower bound(s) of the original domain.
        ub: Upper bound(s) of the original domain.
        instance: Instance number stored in IOH meta data.
        optimum: Known global optimum ``(x, f)`` of the wrapped problem, with
            ``x`` in original coordinates and ``f`` including ``objective_offset``.
        transform: Variable transformation applied in original coordinates,
            after the mapping from [0, 1]^d and before ``function``.
        objective_offset: Constant added to every objective value.
    """
    lb = np.broadcast_to(np.asarray(lb, dtype=float), (dimension,)).copy()
    ub = np.broadcast_to(np.asarray(ub, dtype=float), (dimension,)).copy()
    if np.any(ub <= lb):
        raise ValueError("Upper bounds must be strictly greater than lower bounds")

    def transform_variables(u: np.ndarray, _instance: int) -> np.ndarray:
        x = from_unit(u, lb, ub)
        return x if transform is None else transform(x)

    transform_objectives = None
    if objective_offset != 0.0:

        def transform_objectives(y: float, _instance: int) -> float:
            return y + objective_offset

    calculate_objective = None
    if optimum is not None:
        x_opt = to_unit(optimum[0], lb, ub)
        f_opt = float(optimum[1])

        def calculate_objective(_instance: int, _dimension: int) -> tuple[np.ndarray, float]:
            return x_opt, f_opt

    problem = ioh.wrap_problem(
        function,
        name=_registry_name(suite, name, function_id, instance, dimension),
        dimension=dimension,
        instance=instance,
        lb=0.0,
        ub=1.0,
        transform_variables=transform_variables,
        transform_objectives=transform_objectives,
        calculate_objective=calculate_objective,
    )
    _keep_alive(problem, function, transform_variables, transform_objectives, calculate_objective)
    problem.set_id(function_id)
    problem.set_name(name)
    return problem


def normalize(problem: Problem, suite: str) -> Problem:
    """Return ``problem`` seen on [0, 1]^d.

    Problems already defined on [0, 1]^d are returned unchanged. Other problems
    (e.g. native BBOB on [-5, 5]^d) get a thin IOH wrapper; the wrapper keeps the
    function id, instance and name of the original problem, and counts its own
    evaluations in ``state.evaluations``.
    """
    lb, ub = problem.bounds.lb, problem.bounds.ub
    if np.all(lb == 0.0) and np.all(ub == 1.0):
        return problem

    meta = problem.meta_data
    optimum = None
    if np.all(np.isfinite(problem.optimum.x)) and np.isfinite(problem.optimum.y):
        optimum = (np.asarray(problem.optimum.x), problem.optimum.y)

    return wrap(
        problem,
        suite=suite,
        name=meta.name,
        function_id=meta.problem_id,
        dimension=meta.n_variables,
        lb=lb,
        ub=ub,
        instance=meta.instance,
        optimum=optimum,
    )


def _random_rotation(rng: np.random.Generator, dimension: int) -> np.ndarray:
    # QR of a Gaussian matrix with sign correction gives a Haar-uniform rotation.
    q, r = np.linalg.qr(rng.standard_normal((dimension, dimension)))
    return q * np.sign(np.diag(r))


def _bbob_offset(rng: np.random.Generator) -> float:
    # Same distribution as the BBOB f_opt: a rounded, clipped Cauchy variate.
    value = np.round(100.0 * rng.standard_normal() / rng.standard_normal()) / 100.0
    return float(np.clip(value, -1000.0, 1000.0))


def make_instance(
    function: Callable[[np.ndarray], float],
    *,
    suite: str,
    name: str,
    function_id: int,
    dimension: int,
    instance: int,
    lb: ArrayLike,
    ub: ArrayLike,
    x_opt: ArrayLike,
    seed: int = 0,
    shift: bool = True,
    rotate: bool = True,
    offset: bool = True,
) -> Problem:
    """Create a seeded BBOB-style instance of a problem without native instances.

    The instance is ``g(x) = f(R (x - x*) + x_opt) + f*`` on [lb, ub], seen on
    [0, 1]^d, where ``x_opt`` is the known global minimizer of ``f``:

    - ``x*`` (shift) is uniform in the central 80% of the domain, like BBOB's
      [-4, 4]^d within [-5, 5]^d, so the optimum stays inside the domain,
    - ``R`` (rotation) is a uniformly random orthogonal matrix,
    - ``f*`` (offset) follows the BBOB distribution.

    The random state is derived from ``(seed, function_id, instance, dimension)``,
    so instances are reproducible and independent of creation order.

    Rotation maps parts of the domain outside the original box, so ``f`` must
    be defined there and ``x_opt`` must remain its global minimizer.
    """
    lb = np.broadcast_to(np.asarray(lb, dtype=float), (dimension,)).copy()
    ub = np.broadcast_to(np.asarray(ub, dtype=float), (dimension,)).copy()
    x_opt = np.asarray(x_opt, dtype=float)
    if x_opt.shape != (dimension,):
        raise ValueError(f"x_opt must have shape ({dimension},), got {x_opt.shape}")
    if np.any(x_opt < lb) or np.any(x_opt > ub):
        raise ValueError("x_opt must lie inside [lb, ub]")

    seed_seq = np.random.SeedSequence(seed, spawn_key=(function_id, instance, dimension))
    rng = np.random.default_rng(seed_seq)
    margin = 0.1 * (ub - lb)
    x_star = rng.uniform(lb + margin, ub - margin) if shift else x_opt.copy()
    rotation = _random_rotation(rng, dimension) if rotate else np.eye(dimension)
    f_star = _bbob_offset(rng) if offset else 0.0

    if np.any(x_star < lb) or np.any(x_star > ub):
        raise RuntimeError("Shifted optimum left the domain")

    def transform(x: np.ndarray) -> np.ndarray:
        return rotation @ (x - x_star) + x_opt

    return wrap(
        function,
        suite=suite,
        name=name,
        function_id=function_id,
        dimension=dimension,
        lb=lb,
        ub=ub,
        instance=instance,
        optimum=(x_star, float(function(x_opt)) + f_star),
        transform=transform,
        objective_offset=f_star,
    )
