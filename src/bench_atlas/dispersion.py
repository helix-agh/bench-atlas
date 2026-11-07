"""Dispersion suite: atmospheric source location from dispersion-bench.

The objective is a Rust advection-diffusion model driven by ICON winds. The
``dispersion_bench`` package (0.4 or newer) exposes its problems as IOH
problems, which are adapted here like the other IOH suites. It is not a
declared dependency of bench-atlas (its repository is private); install it
separately to use this suite.
"""

from __future__ import annotations

import functools

from bench_atlas.problems import Problem, from_ioh

__all__ = ["dispersion"]


def dispersion() -> list[Problem]:
    """The 24 source-location problems of dispersion-bench.

    Problems 1-8 have d = 2, 9-16 have d = 4 and 17-24 have d = 6 (one, two
    and three sources). Each problem is its own function with a single
    instance. The domain is a box in metres; f_opt = 0 at the hidden true
    sources, whose location is not exposed, so ``x_opt`` is unknown.
    """
    try:
        import dispersion_bench
    except ImportError as error:
        raise ImportError(
            "The dispersion suite needs the dispersion-bench package, version 0.4 or newer "
            "(https://github.com/zakrzewow/dispersion-bench)"
        ) from error
    return [
        from_ioh(
            problem,
            "dispersion",
            factory=functools.partial(dispersion_bench.get_problem, problem.index),
        )
        for problem in dispersion_bench.get_problems()
    ]
