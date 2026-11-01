"""Generate reference values for tests/test_gnbg.py with the official GNBG code.

The official Python code of GNBG-II (``GNBG_instances.py``) and GNBG-III
(``fitness.py`` and the loader in ``gnbg_iii_competition_harness.py``) needs
numpy < 2.3, so run it in a separate environment, from the repository root.
GNBG-I ships only MATLAB code, so its values come from the GNBG-II ``GNBG``
class (same generator and formula).

    uv run --no-project --python 3.12 --with "numpy<2.3" --with scipy \
        python tests/data/make_gnbg_reference.py

Points per problem (original coordinates):

- ``f{n}_x`` / ``f{n}_y``: the optimum itself, 10 points near the optimum and one
  near the minimum of every component;
- ``f{n}_uniform_y``: values at ``UNIFORM`` points uniform in the box, given by
  ``uniform`` below. Only the values are stored: the points are rebuilt from raw
  PCG64 output, which is stable across numpy versions (unlike ``Generator``
  methods), with the same function in the test.
"""

import ast
import io
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat

GNBG_I_URL = (
    "https://raw.githubusercontent.com/rohitsalgotra/GNBG-I/"
    "35dad28f53add8cf726932e929b7da2d0cf184f6/GNBG_Instances.MATLAB.zip"
)
GNBG_II_URL = (
    "https://raw.githubusercontent.com/rohitsalgotra/GNBG-II/"
    "59d506ed0b317082950d48dd75f21ed2f02aed62/GNBG_Instances.Python-main.zip"
)
GNBG_III_URL = (
    "https://raw.githubusercontent.com/rohitsalgotra/GNBG-III/"
    "b263a6520b68e2f46735235c2c81ed0778edb58d/GNBG_III_Python.zip"
)
DATA = Path(__file__).parent
UNIFORM = 5000


def uniform(suite: str, function: int, n: int, d: int) -> np.ndarray:
    """Points on [0, 1)^d from raw PCG64 output; must match tests/test_gnbg.py."""
    seed = np.random.SeedSequence([int.from_bytes(suite.encode(), "little"), function])
    return (np.random.PCG64(seed).random_raw((n, d)) >> 11) * 2.0**-53


def download(url: str, tmp: str) -> Path:
    with urllib.request.urlopen(url) as response:
        zipfile.ZipFile(io.BytesIO(response.read())).extractall(tmp)
    return next(path for path in Path(tmp).iterdir() if path.name != "__MACOSX")


def points(rng, lb, ub, x_opt, minima):
    d = len(x_opt)
    return np.vstack(
        [
            x_opt[None, :],
            x_opt + rng.normal(0, 1e-2 * (ub - lb), (10, d)),
            minima + rng.normal(0, 1e-3 * (ub - lb), minima.shape),
        ]
    )


def gnbg_ii_class(code: Path) -> type:
    # GNBG_instances.py runs an optimizer on import, so only its GNBG class is used.
    source = (code / "GNBG_instances.py").read_text()
    (cls,) = [node for node in ast.parse(source).body if isinstance(node, ast.ClassDef)]
    namespace = {"np": np}
    exec(ast.get_source_segment(source, cls), namespace)
    return namespace["GNBG"]


def gnbg_ii(code: Path, suite: str = "gnbg_ii", seed: int = 2025) -> dict:
    """GNBG-II-style suite: f1.mat-f24.mat in ``code``, evaluated with GNBG_CLASS."""
    rng, out = np.random.default_rng(seed), {}
    for f in range(1, 25):
        # The loading code of GNBG_instances.py.
        g = loadmat(code / f"f{f}.mat")["GNBG"]
        scalar = lambda key: np.array([item[0] for item in g[key].flatten()])[0, 0]  # noqa: E731, B023
        array = lambda key: np.array(g[key][0, 0])  # noqa: E731, B023
        gnbg = GNBG_CLASS(
            np.inf, scalar("AcceptanceThreshold"), scalar("Dimension"), scalar("o"),
            scalar("MinCoordinate"), scalar("MaxCoordinate"), array("Component_MinimumPosition"),
            np.array(g["ComponentSigma"][0, 0], dtype=np.float64), array("Component_H"),
            array("Mu"), array("Omega"), array("lambda"), array("RotationMatrix"),
            scalar("OptimumValue"), array("OptimumPosition"),
        )  # fmt: skip
        x = points(
            rng,
            gnbg.MinCoordinate,
            gnbg.MaxCoordinate,
            np.ravel(gnbg.OptimumPosition),
            gnbg.CompMinPos.reshape(-1, gnbg.Dimension),
        )
        out[f"f{f}_x"], out[f"f{f}_y"] = x, gnbg.fitness(x)
        u = uniform(suite, f, UNIFORM, gnbg.Dimension)
        out[f"f{f}_uniform_y"] = gnbg.fitness(
            gnbg.MinCoordinate + u * (gnbg.MaxCoordinate - gnbg.MinCoordinate)
        )
    return out


def gnbg_iii(code: Path) -> dict:
    sys.path.insert(0, str(code))
    from fitness import fitness
    from gnbg_iii_competition_harness import load_gnbg_problem

    rng, out = np.random.default_rng(2026), {}
    for f in range(1, 23):
        (path,) = code.glob(f"F{f}_*.mat")
        g = load_gnbg_problem(str(path))
        d = int(g["Dimension"])
        x = points(
            rng,
            float(g["MinCoordinate"]),
            float(g["MaxCoordinate"]),
            np.ravel(g["OptimumPosition"]),
            np.reshape(g["Component_MinimumPosition"], (-1, d)),
        )
        g["FEhistory"] = np.zeros(len(x) + UNIFORM + 1)
        out[f"f{f}_x"], out[f"f{f}_y"] = x, fitness(x, g)[0]
        lb, ub = float(g["MinCoordinate"]), float(g["MaxCoordinate"])
        out[f"f{f}_uniform_y"] = fitness(lb + uniform("gnbg_iii", f, UNIFORM, d) * (ub - lb), g)[0]
    return out


with tempfile.TemporaryDirectory() as tmp:
    gnbg_ii_code = download(GNBG_II_URL, tmp)
    GNBG_CLASS = gnbg_ii_class(gnbg_ii_code)
    with tempfile.TemporaryDirectory() as tmp_i, tempfile.TemporaryDirectory() as tmp_iii:
        for name, values in [
            ("gnbg_i", lambda: gnbg_ii(download(GNBG_I_URL, tmp_i), "gnbg_i", 2024)),
            ("gnbg_ii", lambda: gnbg_ii(gnbg_ii_code)),
            ("gnbg_iii", lambda: gnbg_iii(download(GNBG_III_URL, tmp_iii))),
        ]:
            np.savez_compressed(DATA / f"{name}_reference.npz", **values())
            print(f"wrote {name}_reference.npz with numpy {np.__version__}")
