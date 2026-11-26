"""Generate reference expressions for tests/test_random_trees.py with doe2vec.

Downloads ``modulesRandFunc`` of ``nikivanstein/doe2vec`` (MIT) at a pinned commit and
stores the expression of the first tree for every seed and dimension, as doe2vec
draws it (``random.seed`` and ``np.random.seed``, ``generate_tree(6, 16)``). Run from
the repository root:

    uv run python tests/data/make_random_trees_reference.py

Trees with the noise operand 7 are stored as null: the noise vector has one value per
sample row (doe2vec passes the sample size, here ``SAMPLE``), and bench-atlas rejects
such trees.
"""

import json
import random
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

URL = "https://raw.githubusercontent.com/nikivanstein/doe2vec/4d498c29b32d1b5d704517c01d20776cf8e3e0e1/src/modulesRandFunc/"
FILES = [
    "__init__.py",
    "NODE.py",
    "generate_tree.py",
    "generate_tree2exp.py",
    "generate_exp2fun.py",
]
SEEDS = range(200)
DIMENSIONS = [2, 5]
SAMPLE = 256


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        package = Path(tmp) / "modulesRandFunc"
        package.mkdir()
        for name in FILES:
            with urllib.request.urlopen(URL + name) as response:
                (package / name).write_bytes(response.read())
        sys.path.insert(0, tmp)
        from modulesRandFunc import generate_exp2fun, generate_tree, generate_tree2exp

        reference = {}
        for d in DIMENSIONS:
            for seed in SEEDS:
                random.seed(seed)
                np.random.seed(seed)
                tree = generate_tree2exp.generate_tree2exp(generate_tree.generate_tree(6, 16))
                expression = generate_exp2fun.generate_exp2fun(tree, SAMPLE, d)
                noisy = 7 in generate_exp2fun.flatten(tree)
                reference[f"{seed}_{d}"] = None if noisy else expression
    path = Path(__file__).parent / "random_trees_reference.json"
    path.write_text(json.dumps(reference, indent=0) + "\n")


if __name__ == "__main__":
    main()
