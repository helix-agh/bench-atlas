# IOHClustering

Forty k-means problems on real data: ten datasets
reduced to two features, each with $k \in \{2, 3, 5, 10\}$ clusters. A point holds
the $k$ centroids on the min-max normalized data, so $d = 2k$, and the objective
is the mean squared distance to the nearest centroid.

```python
from bench_atlas.suites import iohclustering

iohclustering()  # 40 problems, d = 4, 6, 10, 20
```

Ids use IOHClustering's dataset numbering: `iohclustering_f{dataset}_i1_d{2k}`.
The global optimum is not known, so `f_opt` is `None`.
