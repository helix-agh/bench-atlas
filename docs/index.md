---
hide:
  - navigation
  - toc
---

<div class="ba-hero" markdown>

![bench-atlas](assets/bench-atlas-lockup-light.svg#only-light){ .ba-hero__logo }
![bench-atlas](assets/bench-atlas-lockup-dark.svg#only-dark){ .ba-hero__logo }

# A map of continuous black-box optimization benchmarks { .ba-hero__title }

Benchmark suites are used to rank optimizers, but how much do they differ,
which problems are redundant, and which ones tell algorithms apart?
**bench-atlas** puts every suite on one footing and measures it.
{ .ba-hero__lead }

[Get started](getting-started.md){ .md-button .md-button--primary }
[Browse benchmarks](benchmarks/index.md){ .md-button }

</div>

## The problem

Each benchmark suite comes with its own domain, scaling, instance mechanism and
reference implementation. Comparing *suites* (not algorithms) therefore needs
one interface and one protocol: the same sample, the same budget, the same
optimizer portfolio, on every problem. bench-atlas provides both, and stores
one row per problem so that the results join into a single atlas.

<div class="grid cards" markdown>

-   :lucide-box:{ .lg .middle } **One problem interface**

    ---

    Every function is a vectorized objective on $[0, 1]^d$ with a stable id
    such as `bbob_f7_i4_d5`, an optional known optimum and an evaluation counter.

    [:octicons-arrow-right-24: Problems](problems.md)

-   :lucide-library:{ .lg .middle } **Eight suites, checked against the originals**

    ---

    BBOB, MA-BBOB, CEC 2022, GNBG-I/II/III, zigzag and IOHClustering, each
    tested against values from its reference implementation.

    [:octicons-arrow-right-24: Benchmarks](benchmarks/index.md)

-   :lucide-chart-scatter:{ .lg .middle } **Landscape features**

    ---

    Exploratory landscape analysis with orivex and compressed local optima
    networks with lonkit, on a shared, seeded sampling protocol.

    [:octicons-arrow-right-24: Analyses](analyses.md)

-   :lucide-swords:{ .lg .middle } **Algorithm discrimination**

    ---

    A five-member Nevergrad portfolio, run with an evaluation budget per
    problem and scored with the normalized AOCC.

    [:octicons-arrow-right-24: Portfolio and AOCC](analyses.md#optimizer-portfolio)

</div>

## In a few lines

```python
import numpy as np
from bench_atlas.suites import bbob

problems = bbob(dimensions=[5], instances=[1])  # 24 problems
f7 = problems[6]
print(f7.id, f7.f_opt)  # bbob_f7_i1_d5 92.94
f7(np.random.default_rng(0).random((100, 5)))  # (100,) objective values
```

!!! warning "Early development"

    bench-atlas is a research codebase under active development. The CLI
    subcommands `metrics` and `report` are still placeholders.
