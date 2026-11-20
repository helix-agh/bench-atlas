"""Generate reference values for tests/test_cec2022.py with the official CEC 2022 C code.

Compiles ``cec22_test_func.cpp`` of the official release (``P-N-Suganthan/2022-SO-BO``,
``CEC2022.zip``, pinned commit) with a small driver and evaluates it, from the
repository root (needs a C++ compiler):

    uv run python tests/data/make_cec2022_reference.py

Values per function F = 1-12 and d = 10, 20 (original coordinates, box [-100, 100]^d):

- ``f{F}_d{d}_x`` / ``f{F}_d{d}_y``: the shift o (the optimum) and 10 points
  within 1 of it;
- ``f{F}_d{d}_uniform_y``: values at ``UNIFORM`` points uniform in the box, given
  by ``uniform`` below (raw PCG64 output, stable across numpy versions).
"""

import io
import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

URL = (
    "https://raw.githubusercontent.com/P-N-Suganthan/2022-SO-BO/"
    "de20505283b76ec6bf17a2e8fc6052e655830691/CEC2022.zip"
)
DATA = Path(__file__).parent
UNIFORM = 100
DRIVER = r"""
#include <cstdio>
#include <cstdlib>
double *OShift, *M, *y, *z, *x_bound;
int ini_flag = 0, n_flag, func_flag, *SS;
void cec22_test_func(double *, double *, int, int, int);
int main(int argc, char **argv) {
    int f = atoi(argv[1]), d = atoi(argv[2]), n = atoi(argv[3]);
    double *x = (double *)malloc(sizeof(double) * n * d), *out = (double *)malloc(sizeof(double) * n);
    if (fread(x, sizeof(double), n * d, stdin) != (size_t)(n * d)) return 1;
    cec22_test_func(x, out, d, n, f);
    fwrite(out, sizeof(double), n, stdout);
    return 0;
}
"""


def uniform(function: int, d: int, n: int) -> np.ndarray:
    """Points on [0, 1)^d from raw PCG64 output; must match tests/test_cec2022.py."""
    seed = np.random.SeedSequence([int.from_bytes(b"cec2022", "little"), function, d])
    return (np.random.PCG64(seed).random_raw((n, d)) >> 11) * 2.0**-53


def evaluate(code: Path, function: int, x: np.ndarray) -> np.ndarray:
    out = subprocess.run(
        [code / "cec22", str(function), str(x.shape[1]), str(len(x))],
        input=np.ascontiguousarray(x).tobytes(),
        cwd=code,  # the C code reads input_data/ relative to the working directory
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(out)


with tempfile.TemporaryDirectory() as tmp:
    with urllib.request.urlopen(URL) as response:
        zipfile.ZipFile(io.BytesIO(response.read())).extractall(tmp)
    code = Path(tmp) / "CEC2022" / "C-Code"
    source = (code / "cec22_test_func.cpp").read_text(errors="replace")
    source = source.replace("#include <WINDOWS.H>", "").replace("<malloc.h>", "<stdlib.h>")
    (code / "cec22_test_func.cpp").write_text(source)
    (code / "driver.cpp").write_text(DRIVER)
    subprocess.run(
        [
            "c++",
            "-O2",
            "-ffp-contract=off",
            "-w",
            "-o",
            "cec22",
            "driver.cpp",
            "cec22_test_func.cpp",
        ],
        cwd=code,
        check=True,
    )
    out = {}
    for function in range(1, 13):
        shift = np.array((code / "input_data" / f"shift_data_{function}.txt").read_text().split())
        for d in (10, 20):
            o = shift[:d].astype(float)
            x = np.vstack([o, o + 2 * uniform(function, d, UNIFORM + 10)[UNIFORM:] - 1])
            out[f"f{function}_d{d}_x"] = x
            out[f"f{function}_d{d}_y"] = evaluate(code, function, x)
            u = uniform(function, d, UNIFORM)
            out[f"f{function}_d{d}_uniform_y"] = evaluate(code, function, -100 + 200 * u)
    np.savez_compressed(DATA / "cec2022_reference.npz", **out)
    print(f"wrote cec2022_reference.npz with {len(out)} arrays")
