"""Benchmark problems.

A :class:`Problem` is a vectorized objective on [0, 1]^d with a stable
:class:`ProblemId`, an optional known optimum and an evaluation counter. IOH is
used as a source of benchmark functions (:func:`from_ioh`), not as the problem
interface.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import ArrayLike

__all__ = [
    "Problem",
    "ProblemId",
    "from_function",
    "from_ioh",
    "from_unit",
    "make_instance",
    "to_unit",
]

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


@dataclass(eq=False)
class Problem:
    """Minimization problem on [0, 1]^d.

    Calling the problem with a single point ``(d,)`` returns a float; calling it
    with a batch ``(n, d)`` returns an array ``(n,)``. Every evaluated point is
    counted in ``evaluations``.

    Attributes:
        id: Stable problem identifier; ``id.dimension`` is the dimension.
        function: Vectorized objective, ``(n, d)`` points on [0, 1]^d -> ``(n,)``.
        name: Human-readable name.
        optimum: Known global optimum ``(x, f)`` with ``x`` on [0, 1]^d.
        evaluations: Number of evaluated points.
    """

    id: ProblemId
    function: Callable[[np.ndarray], np.ndarray] = field(repr=False)
    name: str = ""
    optimum: tuple[np.ndarray, float] | None = None
    evaluations: int = 0

    @property
    def dimension(self) -> int:
        return self.id.dimension

    def __call__(self, x: ArrayLike) -> float | np.ndarray:
        x = np.asarray(x, dtype=float)
        single = x.ndim == 1
        points = np.atleast_2d(x)
        if points.ndim != 2 or points.shape[1] != self.dimension:
            raise ValueError(f"Expected points of dimension {self.dimension}, got {x.shape}")
        y = np.asarray(self.function(points), dtype=float)
        self.evaluations += len(points)
        return float(y[0]) if single else y


def to_unit(x: ArrayLike, lb: ArrayLike, ub: ArrayLike) -> np.ndarray:
    """Map points from the box [lb, ub] to [0, 1]^d."""
    lb, ub = np.asarray(lb, dtype=float), np.asarray(ub, dtype=float)
    return (np.asarray(x, dtype=float) - lb) / (ub - lb)


def from_unit(u: ArrayLike, lb: ArrayLike, ub: ArrayLike) -> np.ndarray:
    """Map points from [0, 1]^d to the box [lb, ub]."""
    lb, ub = np.asarray(lb, dtype=float), np.asarray(ub, dtype=float)
    return lb + np.asarray(u, dtype=float) * (ub - lb)


@dataclass(frozen=True, eq=False)
class _BoxFunction:
    """``u -> base(R (x - shift) + x_opt) + offset`` with ``x = from_unit(u)``.

    A plain picklable object (rather than a closure) so that problems can be
    sent to parallel workers.
    """

    base: Callable[[np.ndarray], np.ndarray]
    lb: np.ndarray
    ub: np.ndarray
    rotation: np.ndarray | None = None
    shift: np.ndarray | None = None
    x_opt: np.ndarray | None = None
    offset: float = 0.0

    def __call__(self, u: np.ndarray) -> np.ndarray:
        x = from_unit(u, self.lb, self.ub)
        if self.rotation is not None:
            x = (x - self.shift) @ self.rotation.T + self.x_opt
        return np.asarray(self.base(x), dtype=float) + self.offset


class _IOHFunction:
    """Vectorized view of an IOH problem that survives pickling.

    IOH problems cannot be pickled; the problem is rebuilt from ``factory`` on
    first use after unpickling.
    """

    def __init__(self, problem: Any, factory: Callable[[], Any]) -> None:
        self._problem = problem
        self._factory = factory

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self._problem is None:
            self._problem = self._factory()
        return np.asarray(self._problem(x), dtype=float)

    def __getstate__(self) -> dict[str, Any]:
        return {"_problem": None, "_factory": self._factory}


def _create_ioh(cls: type, function: int, instance: int, dimension: int) -> Any:
    # Module-level so that the default factory is picklable; pybind methods are not.
    return cls.create(function, instance, dimension)


def _box(dimension: int, lb: ArrayLike, ub: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    lb = np.broadcast_to(np.asarray(lb, dtype=float), (dimension,)).copy()
    ub = np.broadcast_to(np.asarray(ub, dtype=float), (dimension,)).copy()
    if np.any(ub <= lb):
        raise ValueError("Upper bounds must be strictly greater than lower bounds")
    return lb, ub


def from_function(
    function: Callable[[np.ndarray], np.ndarray],
    *,
    id: ProblemId,
    lb: ArrayLike,
    ub: ArrayLike,
    name: str = "",
    optimum: tuple[ArrayLike, float] | None = None,
) -> Problem:
    """Create a problem from a vectorized function defined on the box [lb, ub].

    Args:
        function: Objective on the original domain, ``(n, d) -> (n,)``.
        id: Problem identifier.
        lb: Lower bound(s) of the original domain.
        ub: Upper bound(s) of the original domain.
        name: Human-readable name.
        optimum: Known global optimum ``(x, f)`` with ``x`` in original coordinates.
    """
    lb, ub = _box(id.dimension, lb, ub)
    if optimum is not None:
        optimum = (to_unit(optimum[0], lb, ub), float(optimum[1]))
    return Problem(id=id, function=_BoxFunction(function, lb, ub), name=name, optimum=optimum)


def from_ioh(
    problem: Any,
    suite: str,
    factory: Callable[[], Any] | None = None,
) -> Problem:
    """Adapt an IOH real-valued single-objective problem.

    The id, name, bounds and optimum are read from the IOH problem. Evaluations
    are counted by the returned :class:`Problem`, not by IOH's ``state``.

    Args:
        problem: IOH problem, e.g. from ``ioh.get_problem``.
        suite: Suite name stored in the :class:`ProblemId`.
        factory: Rebuilds the IOH problem after unpickling. Defaults to
            ``type(problem).create(function, instance, dimension)``, which works
            for native IOH problems such as BBOB.
    """
    meta = problem.meta_data
    id = ProblemId(suite, meta.problem_id, meta.instance, meta.n_variables)
    if factory is None:
        factory = functools.partial(
            _create_ioh, type(problem), id.function, id.instance, id.dimension
        )

    optimum = None
    if np.all(np.isfinite(problem.optimum.x)) and np.isfinite(problem.optimum.y):
        optimum = (np.asarray(problem.optimum.x), problem.optimum.y)

    return from_function(
        _IOHFunction(problem, factory),
        id=id,
        lb=problem.bounds.lb,
        ub=problem.bounds.ub,
        name=meta.name,
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
    function: Callable[[np.ndarray], np.ndarray],
    *,
    id: ProblemId,
    lb: ArrayLike,
    ub: ArrayLike,
    x_opt: ArrayLike,
    name: str = "",
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

    The random state is derived from ``(seed, function, instance, dimension)`` of
    ``id``, so instances are reproducible and independent of creation order.

    Rotation maps parts of the domain outside the original box, so ``f`` must
    be defined there and ``x_opt`` must remain its global minimizer.
    """
    lb, ub = _box(id.dimension, lb, ub)
    x_opt = np.asarray(x_opt, dtype=float)
    if x_opt.shape != (id.dimension,):
        raise ValueError(f"x_opt must have shape ({id.dimension},), got {x_opt.shape}")
    if np.any(x_opt < lb) or np.any(x_opt > ub):
        raise ValueError("x_opt must lie inside [lb, ub]")

    seed_seq = np.random.SeedSequence(seed, spawn_key=(id.function, id.instance, id.dimension))
    rng = np.random.default_rng(seed_seq)
    margin = 0.1 * (ub - lb)
    x_star = rng.uniform(lb + margin, ub - margin) if shift else x_opt
    rotation = _random_rotation(rng, id.dimension) if rotate else np.eye(id.dimension)
    f_star = _bbob_offset(rng) if offset else 0.0

    f_opt = float(np.asarray(function(x_opt[None, :]))[0]) + f_star
    return Problem(
        id=id,
        function=_BoxFunction(function, lb, ub, rotation, x_star, x_opt, f_star),
        name=name,
        optimum=(to_unit(x_star, lb, ub), f_opt),
    )
