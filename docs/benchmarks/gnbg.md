# GNBG-I, II, III

The **G**eneralized **N**umerical **B**enchmark **G**enerator
builds a function as the minimum over components, each a rotated, possibly
asymmetric and multimodal basin with its own depth and shape. Three curated
suites of d = 30 from it were used in competitions organized at GECCO and CEC.

| Suite | Function | Problems | Competition |
|---|---|---|---|
| GNBG-I | `gnbg_i()` | f1-f24 | GECCO 2024 |
| GNBG-II | `gnbg_ii()` | f1-f24 | GECCO and CEC 2025 |
| GNBG-III | `gnbg_iii()` | F1-F22 | GECCO and WCCI 2026 |

```python
from bench_atlas.suites import gnbg_i, gnbg_ii, gnbg_iii

problems = gnbg_i() + gnbg_ii() + gnbg_iii()  # 70 problems, d = 30
```

## Notes on the data

- **GNBG-I** uses the corrected files of December 2024. In the original files
  (November 2023) f20 records a wrong optimum and all five components of f24
  coincide.
- **GNBG-III** F23 (noisy) and F24 (dynamic) are left out: they are not
  deterministic functions.
- The objective is reimplemented, vectorized, and checked against values
  computed with the official Python code.

!!! warning "Very sharp optima"

    With component shape parameters down to $\lambda = 0.05$, a rounding error of
    1e-14 in $x$ can change $f$ by about 0.1. bench-atlas evaluates GNBG directly
    on $[0, 1]^d$ so that $f(x_\text{opt}) = f_\text{opt}$ exactly, but expect a
    precision floor of about 0.1 on such problems.
