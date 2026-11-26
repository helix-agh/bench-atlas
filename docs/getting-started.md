# Getting started

## Install

bench-atlas needs Python 3.12+ and is managed with [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/helix-agh/bench-atlas
cd bench-atlas
uv sync
```

GNBG and zigzag data are downloaded on first use (at pinned commits of the
authors' repositories) into `~/.cache/bench-atlas/`.

## Use it from Python

Every suite is a plain function that returns a list of
[`Problem`](reference/problems.md#bench_atlas.problems.Problem)s.

```python
import numpy as np
from bench_atlas.benchmarks import bbob, gnbg_ii

problems = bbob(dimensions=[2, 5], instances=[1, 2, 3]) + gnbg_ii()

p = problems[0]
x = np.random.default_rng(0).random((1000, p.dimension))  # points on [0, 1]^d
y = p(x)  # shape (1000,)
print(p.id, p.evaluations)  # bbob_f1_i1_d2 1000
```

## Use it from the command line

Each analysis writes one Parquet table to the results folder and skips rows that
are already there, so interrupted runs resume where they stopped.

```bash
uv run bench-atlas ela bbob gnbg_ii -j -1     # ELA features
uv run bench-atlas lon bbob -j -1             # local optima networks (d <= 10)
uv run bench-atlas timing bbob                # evaluation cost
uv run bench-atlas run bbob -j -1             # optimizer portfolio
uv run bench-atlas aocc                       # AOCC from the stored runs
```

By default `ela` and `run` use the **atlas problem set**: d = 2, 5, 10, 20, 40
and instances 1-5 for BBOB and MA-BBOB; GNBG suites have a fixed d = 30 and a
single instance. Narrow it down with `--dimension/-d` and `--instance/-i`. The
exceptions:

- `lon` defaults to d = 2, 5, 10 and instance 1 (LON is expensive).
- `timing` always uses instance 1 and takes only `--dimension/-d`.
- `aocc` selects no problems: it scores the runs already stored in the results folder.

!!! note "CEC 2022 and zigzag need `--dimension`"

    The CLI takes all eight suites, but CEC 2022 exists only for d = 10, 20 and
    zigzag only for d = 2, 5, 10, 15, 20, so the default dimensions fail for
    them. For zigzag, `--instance/-i` selects the parametrizations (1-100):

    ```bash
    uv run bench-atlas ela cec2022 -d 10 -d 20
    uv run bench-atlas ela zigzag -d 2 -d 5 -d 10 -i 1 -i 50 -i 100
    ```
