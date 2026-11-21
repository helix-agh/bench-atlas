# Analyses

Every analysis maps a problem to one row of features. Rows are stored in one
Parquet table per analysis and joined on the problem id. Seeds are derived from a
global seed, so every row can be recomputed exactly.

## Exploratory landscape analysis

ELA features [[Mersmann et al. 2011]](references.md#mersmann2011){ .ba-cite } are computed with
[orivex](https://github.com/helix-agh/orivex) on a scrambled Sobol sample of
$250 \cdot d$ points. Replicate $r$ of every problem of dimension $d$ uses the
**same sample**, so differences between problems are not sampling noise.

```bash
uv run bench-atlas ela bbob ma_bbob gnbg_ii --replicates 10 -j -1
```

## Local optima networks

Basin-hopping runs sample the local optima and the transitions between them;
[lonkit](https://github.com/helix-agh/lonkit) builds the network
[[Ochoa et al. 2008]](references.md#ochoa2008){ .ba-cite } and compresses it to a CMLON [[Adair et al. 2019]](references.md#adair2019){ .ba-cite }. Sampling and network
construction are separate: runs are stored at full precision, so a CMLON can be
rebuilt at another precision without sampling again.

```bash
uv run bench-atlas lon bbob -d 2 -d 5 -d 10 -j -1
```

LON is expensive, so its default problem set is d ≤ 10 and instance 1.

## Evaluation cost

`timing` measures the wall-clock time of one evaluation, single-threaded, after
a warm-up. It tells cheap suites from expensive ones and is a cost axis in the
comparison.

## Optimizer portfolio

Five optimizers from Nevergrad [[Rapin and Teytaud 2018]](references.md#rapin2018){ .ba-cite }, one per method class, each with a
budget of $1000 \cdot d$ evaluations:

| Optimizer | Method |
|---|---|
| `CMAstd` | CMA-ES, full covariance, initial step size 0.25 |
| `LBFGSB` | L-BFGS-B with random restarts |
| `RPowell` | Powell's method with random restarts |
| `TwoPointsDE` | differential evolution |
| `RealSpacePSO` | particle swarm optimization |

The best-so-far curve of every run is stored losslessly as the evaluations at
which it improved.

## AOCC

Runs are scored with the normalized area over the convergence curve
[[López-Ibáñez et al. 2024]](references.md#lopezibanez2024){ .ba-cite }. With $f^*$ the known optimum (or the best value of any run)
and $f_\text{med}$ the median of $f$ on the ELA sample, the error after $t$
evaluations is

$$
e(t) = \frac{f_\text{best}(t) - f^*}{f_\text{med} - f^*},
\qquad
\text{AOCC} = \frac{1}{t_\text{max}} \sum_{t=1}^{t_\text{max}}
\left[ 1 - \operatorname{clip}\!\left(\frac{\log_{10} e(t) + 8}{8}, 0, 1\right) \right].
$$

AOCC = 0 means "no better than a typical random point", AOCC = 1 means reaching
$10^{-8}$ at once. It is computed at budgets of 100, 300 and 1000 · d from the
same curves.

```bash
uv run bench-atlas run bbob -j -1
uv run bench-atlas aocc
```
