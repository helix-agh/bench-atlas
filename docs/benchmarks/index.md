# Benchmarks

Eight suites, all exposed as [`Problem`](../problems.md)s on $[0, 1]^d$. Each is
checked in the test suite against values computed with its **reference
implementation** (IOH, the authors' Python, MATLAB or C code).

| Suite | Problems | Dimensions | Instances | Optimum known | Reference implementation |
|---|---|---|---|---|---|
| [BBOB](bbob.md) | 24 per (d, instance) | any | any | :lucide-check: | IOH |
| [MA-BBOB](bbob.md#ma-bbob) | 1 per (d, instance) | any | any | :lucide-check: | IOH |
| [CEC 2022](cec2022.md) | 12 per d | 10, 20 | 1 | :lucide-check: | official C code |
| [GNBG-I](gnbg.md) | 24 | 30 | 1 | :lucide-check: | authors' code |
| [GNBG-II](gnbg.md) | 24 | 30 | 1 | :lucide-check: | authors' code |
| [GNBG-III](gnbg.md) | 22 | 30 | 1 | :lucide-check: | authors' code |
| [Zigzag](zigzag.md) | 4 per (d, parametrization) | 2, 5, 10, 15, 20 | 1-100 | :lucide-check: | authors' code |
| [IOHClustering](iohclustering.md) | 40 | 4, 6, 10, 20 | 1 | :lucide-x: | `iohclustering` |

Each suite is one function in [`bench_atlas.suites`](../reference/suites.md), named like its id prefix:
`bbob()`, `ma_bbob()`, `cec2022()`, `gnbg_i()`, `gnbg_ii()`, `gnbg_iii()`, `zigzag()` and
`iohclustering()`.

## Problem ids

Every problem has an id `{suite}_f{function}_i{instance}_d{dimension}`, e.g.
`bbob_f7_i4_d5`, `gnbg_ii_f6_i1_d30` or `zigzag_f2_i37_d10`. Ids are stable
across versions and are the join key of all result tables.
