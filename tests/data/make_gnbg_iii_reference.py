"""Generate reference values for tests/test_suites.py with the official GNBG-III code.

The official Python code (``fitness.py`` and the loader in
``gnbg_iii_competition_harness.py``) needs numpy < 2.3, so run it in a separate
environment, from the repository root:

    uv run --no-project --python 3.12 --with "numpy<2.3" --with scipy \
        python tests/data/make_gnbg_iii_reference.py

Points per problem (original coordinates): the optimum itself, 10 uniform in the
box, 10 near the optimum and one near the minimum of every component.
"""

import io
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

URL = (
    "https://raw.githubusercontent.com/rohitsalgotra/GNBG-III/"
    "b263a6520b68e2f46735235c2c81ed0778edb58d/GNBG_III_Python.zip"
)
OUT = Path(__file__).with_name("gnbg_iii_reference.npz")

with tempfile.TemporaryDirectory() as tmp:
    with urllib.request.urlopen(URL) as response:
        zipfile.ZipFile(io.BytesIO(response.read())).extractall(tmp)
    code = Path(tmp) / "GNBG_III_Benchmarks_Python"
    sys.path.insert(0, str(code))
    from fitness import fitness
    from gnbg_iii_competition_harness import load_gnbg_problem

    rng = np.random.default_rng(2026)
    out = {}
    for f in range(1, 23):
        (path,) = code.glob(f"F{f}_*.mat")
        g = load_gnbg_problem(str(path))
        d, lb, ub = int(g["Dimension"]), float(g["MinCoordinate"]), float(g["MaxCoordinate"])
        x_opt = np.ravel(g["OptimumPosition"])
        minima = np.reshape(g["Component_MinimumPosition"], (-1, d))
        x = np.vstack(
            [
                x_opt[None, :],
                rng.uniform(lb, ub, (10, d)),
                x_opt + rng.normal(0, 1e-2 * (ub - lb), (10, d)),
                minima + rng.normal(0, 1e-3 * (ub - lb), minima.shape),
            ]
        )
        g["FEhistory"] = np.zeros(len(x) + 1)
        y, _ = fitness(x, g)
        out[f"f{f}_x"], out[f"f{f}_y"] = x, y
    np.savez_compressed(OUT, **out)
    print(f"wrote {OUT} with numpy {np.__version__}")
