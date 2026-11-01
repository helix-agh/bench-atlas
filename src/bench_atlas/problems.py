"""Benchmark problems.

A :class:`Problem` is a vectorized objective on [0, 1]^d with a stable
:class:`ProblemId`, an optional known optimum and an evaluation counter. IOH is
used as a source of benchmark functions (:func:`from_ioh`), not as the problem
interface.
"""

from __future__ import annotations

import functools
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
    "to_unit",
]


@dataclass(frozen=True, order=True)
class ProblemId:
    """Stable identifier of a benchmark problem: (suite, function, instance, d)."""

    suite: str
    function: int
    instance: int
    dimension: int

    def __str__(self) -> str:
        return f"{self.suite}_f{self.function}_i{self.instance}_d{self.dimension}"


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
        if y.shape != (len(points),):
            raise ValueError(f"Expected objective values of shape ({len(points)},), got {y.shape}")
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
    """``u -> base(from_unit(u, lb, ub))``.

    A plain picklable object (rather than a closure) so that problems can be
    sent to parallel workers.
    """

    base: Callable[[np.ndarray], np.ndarray]
    lb: np.ndarray
    ub: np.ndarray

    def __call__(self, u: np.ndarray) -> np.ndarray:
        return self.base(from_unit(u, self.lb, self.ub))


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
