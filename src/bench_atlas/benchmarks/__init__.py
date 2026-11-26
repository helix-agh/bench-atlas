"""Benchmark suites as lists of [`Problem`][bench_atlas.problems.Problem].

One module per suite family; every suite function is re-exported here, so
``from bench_atlas.benchmarks import bbob`` works for all of them.
"""

from bench_atlas.benchmarks.bbob import bbob, ma_bbob
from bench_atlas.benchmarks.cec2022 import cec2022
from bench_atlas.benchmarks.gnbg import gnbg_i, gnbg_ii, gnbg_iii
from bench_atlas.benchmarks.iohclustering import iohclustering
from bench_atlas.benchmarks.portal import portal
from bench_atlas.benchmarks.random_trees import random_trees
from bench_atlas.benchmarks.zigzag import zigzag

__all__ = [
    "bbob",
    "cec2022",
    "gnbg_i",
    "gnbg_ii",
    "gnbg_iii",
    "iohclustering",
    "ma_bbob",
    "portal",
    "random_trees",
    "zigzag",
]
