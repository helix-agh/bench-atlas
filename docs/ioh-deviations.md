# Deviations from plain IOH

bench-atlas uses [IOH](https://github.com/IOHprofiler/IOHexperimenter) problems
everywhere. This page lists every place where we add to or change plain IOH, and
why. All of it lives in `bench_atlas.problems`.

| What | Why |
| --- | --- |
| `ProblemId` = (suite, function, instance, d) | IOH meta data has the function id, instance and dimension but not the suite, and wrapped problems get an arbitrary IOH id. `problem_id(problem, suite)` combines the meta data with an explicit suite name. |
| `wrap()` calls `set_id` / `set_name` after `ioh.wrap_problem` | `ioh.wrap_problem` assigns its own problem id and uses the registry name as the problem name. We restore the function id and a readable name so that `problem_id` is stable. |
| Unique registry names (`suite/name/f/i/d`) for wrapped problems | `ioh.wrap_problem` registers problems globally by name and silently replaces earlier registrations. Existing problem objects are unaffected, but unique names keep the registry unambiguous. |
| Callbacks kept alive with `weakref.finalize` | `ioh.wrap_problem` (ioh 0.3.22) does not hold references to the Python callbacks it receives; once they are garbage collected, calling the problem segfaults. |
| Domain normalization to [0, 1]^d | Wrapped problems use IOH's `transform_variables` hook to map [0, 1]^d to the original box. Native problems (e.g. BBOB on [-5, 5]^d) cannot be re-bounded, so `normalize()` wraps them in a thin IOH problem that maps the point and calls the native one. The wrapper counts its own evaluations; the inner problem's state is not used. |
| BBOB-style instances via `make_instance()` | For problems without native instances. Shift and rotation go through `transform_variables`, the f offset through `transform_objectives`, and the optimum through `calculate_objective`. The random state comes from `numpy.random.SeedSequence(seed, spawn_key=(function, instance, d))`, so instances do not depend on creation order. |

## Instance transformation

`make_instance` builds `g(x) = f(R (x - x*) + x_opt) + f*` on the original box:

- `x*` is uniform in the central 80% of the box (BBOB: [-4, 4]^d within
  [-5, 5]^d), so the optimum always stays inside the domain; this is checked.
- `R` is a uniformly random orthogonal matrix (QR of a Gaussian matrix).
- `f*` follows the BBOB offset distribution: a Cauchy variate rounded to two
  decimals and clipped to [-1000, 1000].

`f` needs a known global minimizer `x_opt`. Rotation maps parts of the box
outside the original domain, so `f` must be defined there and `x_opt` must stay
its global minimizer.

## Known limitations

- Wrapped IOH problems cannot be pickled. Parallel workers must rebuild problems
  from their `ProblemId` rather than receive problem objects.
