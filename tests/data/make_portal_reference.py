"""Generate reference values for tests/test_portal.py with the official PORTAL code.

Clones ``EvoMindLab/PORTAL`` at the pinned commit into a temporary folder and
evaluates every recipe of ``Benchmark_S50`` with its ``portal_core.fitness``,
independently of the copy vendored in ``bench_atlas.benchmarks._portal``. From the
repository root:

    uv run python tests/data/make_portal_reference.py

Values per recipe (original coordinates, box [-100, 100]^d):

- ``{recipe}_x`` / ``{recipe}_y``: the optimum (centre of the deepest component
  of every block), 10 points within 1 of it, and ``UNIFORM`` points uniform in
  the box, given by ``uniform`` below (raw PCG64 output, stable across numpy
  versions);
- ``{recipe}_f_opt``: PORTAL's ``global_optimum_value``.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

COMMIT = "d4027ec6e0ac233b86c0bd1617181eb130e60c96"  # tag v2.0.0
DATA = Path(__file__).parent
NEAR = 10
UNIFORM = 20


def uniform(index: int, d: int, n: int) -> np.ndarray:
    """Points on [0, 1)^d from raw PCG64 output; must match tests/test_portal.py."""
    seed = np.random.SeedSequence([int.from_bytes(b"portal", "little"), index, d])
    return (np.random.PCG64(seed).random_raw((n, d)) >> 11) * 2.0**-53


def main() -> None:
    sys.path.insert(0, str(DATA.parent.parent / "src"))
    from bench_atlas.benchmarks.portal import S50

    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "PORTAL"
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "fetch",
                "-q",
                "--depth",
                "1",
                "https://github.com/EvoMindLab/PORTAL.git",
                COMMIT,
            ],
            check=True,
        )
        subprocess.run(["git", "-C", str(repo), "checkout", "-q", "FETCH_HEAD"], check=True)
        sys.path.insert(0, str(repo))
        from portal_core.fitness import fitness
        from portal_core.load_instance import load_instance_payload
        from portal_core.optimum import global_optimum_value

        out = {}
        for index, recipe in enumerate(S50):
            raw = json.loads((repo / "Benchmark_S50" / f"{recipe}.json").read_text())
            instance = load_instance_payload(raw)
            instance["warn_out_of_bounds"] = False
            d = int(instance["d"])
            beta = np.asarray(instance["P"]["beta"], dtype=float).reshape(-1)
            centres = np.asarray(instance["P"]["c"], dtype=float)
            blocks = instance.get("_schema_blocks") or [
                {"K": instance["K"], "dim_indices": range(d)}
            ]
            x_opt, first = np.empty(d), 0
            for block in blocks:
                k = first + int(np.argmin(beta[first : first + int(block["K"])]))
                dims = list(block["dim_indices"])
                x_opt[dims] = centres[k, dims]
                first += int(block["K"])
            near = x_opt + (uniform(index, d, NEAR) * 2 - 1)
            x = np.vstack([x_opt, near, -100 + 200 * uniform(1000 + index, d, UNIFORM)])
            out[f"{recipe}_x"] = x
            out[f"{recipe}_y"] = np.array([fitness(row, instance) for row in x])
            out[f"{recipe}_f_opt"] = np.array(global_optimum_value(instance))
    np.savez_compressed(DATA / "portal_reference.npz", **out)


if __name__ == "__main__":
    main()
