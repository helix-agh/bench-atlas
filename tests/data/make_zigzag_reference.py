"""Generate reference values for tests/test_zigzag.py with the official MATLAB code.

Runs the zigzag functions F1-F4 of the journal paper (``JakubKudela89/Zigzag``,
pinned commit) in GNU Octave, from the repository root:

    uv run python tests/data/make_zigzag_reference.py

Values per function F, dimension d and all 100 parametrizations of Table 1
(original coordinates, box [-100, 100]^d):

- ``f{F}_d{d}_x`` / ``f{F}_d{d}_y``: the optimum x = -s and 10 points within
  1 of it; ``y`` has one row per parametrization;
- ``f{F}_d{d}_uniform_y``: values at ``UNIFORM`` points uniform in the box, given
  by ``uniform`` below (raw PCG64 output, stable across numpy versions).
"""

import re
import subprocess
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

URL = (
    "https://raw.githubusercontent.com/JakubKudela89/Zigzag/"
    "a7ce2c260b3a46a3d67fa097f8e9544de4fbf5f6/journal%20paper/F{}.m"
)
DATA = Path(__file__).parent
DIMENSIONS = (2, 5, 10, 15, 20)
UNIFORM = 50
# Table 1 of the paper, in the order of the IDs 1-100.
PARAMETERS = [
    (k, m, lam)
    for m in ("0.1", "0.5", "0.9", "1.0")
    for lam in ("0.01", "0.1", "0.5", "0.9", "0.99")
    for k in ("1", "2", "4", "8", "16")
]


def uniform(function: int, d: int, n: int) -> np.ndarray:
    """Points on [0, 1)^d from raw PCG64 output; must match tests/test_zigzag.py."""
    seed = np.random.SeedSequence([int.from_bytes(b"zigzag", "little"), function, d])
    return (np.random.PCG64(seed).random_raw((n, d)) >> 11) * 2.0**-53


def shifts(source: str) -> dict[int, np.ndarray]:
    blocks = re.findall(r"d == (\d+)\s+M\s*=\s*\[[^\]]*\];\s*s\s*=\s*\[([^\]]*)\]", source)
    return {int(d): np.array(s.replace(",", " ").split(), dtype=float) for d, s in blocks}


def octave(code: Path, function: int, x: np.ndarray, tmp: Path) -> np.ndarray:
    """F{function}(x, k, m, lambda) for every parametrization, shape (100, len(x))."""
    (tmp / "x.bin").write_bytes(np.ascontiguousarray(x).tobytes())
    calls = "\n".join(
        f"y(end + 1, :) = F{function}(x, {k}, {m}, {lam});" for k, m, lam in PARAMETERS
    )
    script = f"""
        addpath('{code}');
        fid = fopen('{tmp / "x.bin"}'); x = fread(fid, [{x.shape[1]}, {len(x)}], 'double')'; fclose(fid);
        y = zeros(0, {len(x)});
        {calls}
        fid = fopen('{tmp / "y.bin"}', 'w'); fwrite(fid, y', 'double'); fclose(fid);
    """
    subprocess.run(["octave", "--no-gui", "--quiet", "--eval", script], check=True)
    return np.frombuffer((tmp / "y.bin").read_bytes()).reshape(len(PARAMETERS), len(x))


with tempfile.TemporaryDirectory() as tmp:
    tmp, out = Path(tmp), {}
    for function in range(1, 5):
        with urllib.request.urlopen(URL.format(function)) as response:
            source = response.read()
        (tmp / f"F{function}.m").write_bytes(source)
        for d, s in shifts(source.decode()).items():
            if d not in DIMENSIONS:
                continue
            near = -s + (2 * uniform(function, d, 10 + UNIFORM)[UNIFORM:] - 1)
            x = np.vstack([-s, near])
            out[f"f{function}_d{d}_x"] = x
            out[f"f{function}_d{d}_y"] = octave(tmp, function, x, tmp)
            out[f"f{function}_d{d}_uniform_y"] = octave(
                tmp, function, -100 + 200 * uniform(function, d, UNIFORM), tmp
            )
    np.savez_compressed(DATA / "zigzag_reference.npz", **out)
    print(f"wrote zigzag_reference.npz with {len(out)} arrays")
