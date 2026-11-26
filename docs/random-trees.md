# Random trees: fixes to doe2vec

`random_trees` ports the random function generator of doe2vec
(`nikivanstein/doe2vec`, commit 4d498c2, `modulesRandFunc` and `generateData`), which
ports Tian et al. 2020. The tree generator, injections, cleaning and acceptance filter
are unchanged. doe2vec builds operands with wrong array shapes, so some values depend
on the other points of the batch, and some operands are silently dropped in some
dimensions. We fix the shapes as follows.

| # | Problem in doe2vec | Why it fails | What we do |
|---|---|---|---|
| 1 | x1 is `array_x[:,0]`, shape `(n,)` | `(n,)` broadcasts against `(n, d)` along the last axis. For d = 1 that gives an `(n, n)` matrix, so the value of a point is averaged over the whole batch. For d = 256, the filter's sample size, the rows get mixed and the tree passes the filter. For any other d the tree raises and is rejected, so x1 almost never occurs. | `array_x[:,:1]`, shape `(n, 1)` |
| 2 | sum, mean, prod and max use `axis=1` and give `(n,)` | Same as #1: `sum(x) + x` mixes rows at d = 1 and d = 256 and is rejected elsewhere. | `keepdims=True`, shape `(n, 1)`. The objective reshapes the final `(n, 1)` to `(n,)`. |
| 3 | The translated vector `(x2, ..., xd, 0)` is built with `vstack` of raveled arrays | The lengths match only for d = 2, so for any other d the tree raises and is rejected. The operand set depends on the dimension. | `hstack((array_x[:,1:], zeros((n, 1))))`, correct for every d and identical at d = 2 |
| 4 | The rotated vector uses `np.dot` (BLAS) | BLAS blocking changes the summation order with the batch size: 1 ULP differences, amplified near singularities (one function in 1500 differed by more than 1e-9). | `einsum("ij,jk->ik")`, bitwise independent of the batch |
| 5 | The noise operand is `rand(n, 1)`, one fixed value per sample row | It is not a function of x: the value depends on the row index and batch size. | Unchanged: trees with noise are rejected (5% of draws), as before |

Not changed, kept as in doe2vec: the "rotation" matrix is uniform on [0, 1]^(d x d),
not orthogonal. The filter rejects any |y| < 1e-8 on its sample, which includes the
origin. The domain is [0, 1]^d. Acceptance on 2^8 points does not guarantee bounded
values elsewhere: a few percent of functions exceed |y| > 1e8 on random points.

## Effect

- Every value is bitwise independent of the batch. Tested with batch sizes 1, 7, d
  and 256 for d in {1, 2, 3, 5, 10, 20}, 300 instances each, and in `test_points_are_independent`
  (including d = 1 and d = 256).
- x1 and the translated vector now occur in every dimension, in about 50-65% of
  accepted functions. Before: x1 in about 1% (at d = 1 only through bug #1), the
  translated vector only at d = 2.
- Instances change wherever a tree that used to be rejected is now accepted: 191 / 300
  are the same function at d = 2, 137 / 300 at d = 20. Results computed with the old
  `random_trees` must be recomputed.

## Check against doe2vec

`test_matches_doe2vec` draws the same trees with the same seeds as doe2vec and compares
the expression strings after mapping our operands back to doe2vec's. Only the operands
above differ. `test_fixed_operands` checks their values on a small example.
