# Zigzag

Four functions F1-F4 built on a parametrized zigzag pattern [[Kůdela and Matoušek 2022]](../references.md#kudela2022){ .ba-cite }, with
one parametrization from a grid of 100:

- $k \in \{1, 2, 4, 8, 16\}$, the number of zigzags,
- $\lambda \in \{0.01, 0.1, 0.5, 0.9, 0.99\}$,
- $m \in \{0.1, 0.5, 0.9, 1\}$.

The parametrization number (1-100, the IDs of Table 1 of the paper) is the
instance in the problem id: `zigzag_f2_i37_d10`. Its $k$, $m$ and $\lambda$ are
kept in `problem.name`.

```python
from bench_atlas.suites import zigzag

zigzag(dimensions=[2, 10], parametrizations=range(1, 101))  # 800 problems
```

Each function has its own rotation and shift, available only for
d in {2, 5, 10, 15, 20}. The domain is $[-100, 100]^d$ and the unique optimum is
$f = 0$, reached exactly at $x_\text{opt}$. Values are checked against the
authors' MATLAB code run in Octave.
