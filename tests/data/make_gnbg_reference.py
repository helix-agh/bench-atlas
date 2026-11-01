"""Generate reference values for tests/test_suites.py with the official GNBG code.

The official Python code of GNBG-II (``GNBG_instances.py``) and GNBG-III
(``fitness.py`` and the loader in ``gnbg_iii_competition_harness.py``) needs
numpy < 2.3, so run it in a separate environment, from the repository root:

    uv run --no-project --python 3.12 --with "numpy<2.3" --with scipy \
        python tests/data/make_gnbg_reference.py

Points per problem (original coordinates): the optimum itself, 10 uniform in the
box, 10 near the optimum and one near the minimum of every component.
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

GNBG_II_URL = (
    "https://raw.githubusercontent.com/rohitsalgotra/GNBG-II/"
    "59d506ed0b317082950d48dd75f21ed2f02aed62/GNBG_Instances.Python-main.zip"
)
GNBG_III_URL = (
    "https://raw.githubusercontent.com/rohitsalgotra/GNBG-III/"
    "b263a6520b68e2f46735235c2c81ed0778edb58d/GNBG_III_Python.zip"
)
DATA = Path(__file__).parent


def download(url: str, tmp: str) -> Path:
    with urllib.request.urlopen(url) as response:
        zipfile.ZipFile(io.BytesIO(response.read())).extractall(tmp)
    return next(Path(tmp).iterdir())


def points(rng, lb, ub, x_opt, minima):
    d = len(x_opt)
    return np.vstack(
        [
            x_opt[None, :],
            rng.uniform(lb, ub, (10, d)),
            x_opt + rng.normal(0, 1e-2 * (ub - lb), (10, d)),
            minima + rng.normal(0, 1e-3 * (ub - lb), minima.shape),
        ]
    )


def gnbg_ii(code: Path) -> dict:
    # GNBG_instances.py runs an optimizer on import, so only its GNBG class is used.
    source = (code / "GNBG_instances.py").read_text()
    (cls,) = [node for node in ast.parse(source).body if isinstance(node, ast.ClassDef)]
    namespace = {"np": np}
    exec(ast.get_source_segment(source, cls), namespace)

    rng, out = np.random.default_rng(2025), {}
    for f in range(1, 25):
        # The loading code of GNBG_instances.py.
        g = loadmat(code / f"f{f}.mat")["GNBG"]
        scalar = lambda key: np.array([item[0] for item in g[key].flatten()])[0, 0]  # noqa: E731, B023
        array = lambda key: np.array(g[key][0, 0])  # noqa: E731, B023
        gnbg = namespace["GNBG"](
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
        g["FEhistory"] = np.zeros(len(x) + 1)
        out[f"f{f}_x"], out[f"f{f}_y"] = x, fitness(x, g)[0]
    return out


for name, url, make in [
    ("gnbg_ii", GNBG_II_URL, gnbg_ii),
    ("gnbg_iii", GNBG_III_URL, gnbg_iii),
]:
    with tempfile.TemporaryDirectory() as tmp:
        np.savez_compressed(DATA / f"{name}_reference.npz", **make(download(url, tmp)))
    print(f"wrote {name}_reference.npz with numpy {np.__version__}")
