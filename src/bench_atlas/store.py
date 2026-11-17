"""Results store: one Parquet file per analysis in a results folder.

Each analysis (``ela``, ``lon``, ...) is a single table ``<results>/<analysis>.parquet``.
Saving rewrites the whole file through a temporary file and an atomic rename, so the
table on disk is always complete. A lock file guards against two runs writing the same
table at once, which would lose rows, so one analysis in one results folder is written
by one run (one node) at a time. CMLON graphs are stored as GraphML files in
``<results>/cmlon/``, written the same way. IOH logs of the portfolio runs go to
``<results>/ioh-logs/<suite>/<optimizer>/`` (see :mod:`bench_atlas.ioh_logs`).

The executor (:mod:`bench_atlas.executor`) reuses a stored row when its problem, global
seed, replicate, run and analysis settings match. Neither the analysis code nor the library versions are
part of that key: the versions are recorded in ``version_*`` columns, and :func:`load`
warns if a table mixes them. After a change to the analysis code, start from a fresh
results folder or delete ``<results>/<analysis>.parquet``.
"""

from __future__ import annotations

import math
import os
import socket
import warnings
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import igraph as ig
import pandas as pd
from lonkit import CMLON

__all__ = [
    "LockedError",
    "cmlon_path",
    "ioh_logs_path",
    "load",
    "load_all",
    "load_cmlon",
    "lock",
    "path",
    "save",
    "save_cmlon",
]

VERSION_PREFIX = "version_"
CMLON_FOLDER = "cmlon"
IOH_LOGS_FOLDER = "ioh-logs"
_CMLON_ATTRIBUTES = ("best_fitness", "eq_atol", "minimize")


class LockedError(RuntimeError):
    """Another run holds the lock of a results table."""


def load_all(results: Path) -> dict[str, pd.DataFrame]:
    """Every table in ``results``, by analysis name."""
    return {file.stem: load(results, file.stem) for file in sorted(results.glob("*.parquet"))}


def load(results: Path, analysis: str) -> pd.DataFrame:
    """The table of ``analysis``; empty if it doesn't exist yet.

    Warns if the rows were computed with different versions of the libraries recorded
    in the ``version_*`` columns.
    """
    file = path(results, analysis)
    if not file.exists():
        return pd.DataFrame()
    table = pd.read_parquet(file)
    mixed = [
        column
        for column in table.columns
        if column.startswith(VERSION_PREFIX) and table[column].nunique() > 1
    ]
    if mixed:
        warnings.warn(
            f"{file} mixes versions in {', '.join(mixed)}",
            UserWarning,
            stacklevel=2,
        )
    return table


def save(results: Path, analysis: str, table: pd.DataFrame) -> None:
    """Replace the table of ``analysis`` with ``table`` atomically."""
    file = path(results, analysis)
    file.parent.mkdir(parents=True, exist_ok=True)
    temporary = file.with_name(file.name + ".tmp")
    table.to_parquet(temporary, index=False)
    temporary.replace(file)


@contextmanager
def lock(results: Path, analysis: str) -> Generator[None, None, None]:
    """Hold the lock of the ``analysis`` table for the duration of the block.

    The lock is a file created exclusively next to the table (``flock`` is not used:
    file systems on HPC clusters may not support it). It holds the host and process id of
    its owner and is removed on exit, including on exceptions.

    Raises:
        LockedError: Another run holds the lock.
    """
    file = path(results, analysis).with_suffix(".parquet.lock")
    file.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        owner = file.read_text().strip() or "unknown"
        raise LockedError(
            f"{file} is held by {owner}. Wait for that run to finish, or delete the "
            "file if it was killed."
        ) from None
    with os.fdopen(descriptor, "w") as handle:
        handle.write(f"{socket.gethostname()}:{os.getpid()}\n")
    try:
        yield
    finally:
        file.unlink(missing_ok=True)


def path(results: Path, analysis: str) -> Path:
    """Location of the ``analysis`` table."""
    return Path(results) / f"{analysis}.parquet"


def save_cmlon(results: Path, name: str, cmlon: CMLON) -> Path:
    """Write ``cmlon`` to ``<results>/cmlon/<name>.graphml`` atomically.

    GraphML keeps the vertex ``name``, ``Fitness`` and ``Count`` and the edge
    ``Count``; ``best_fitness``, ``eq_atol`` and ``minimize`` are stored as graph
    attributes, so :func:`load_cmlon` restores everything needed to plot the network.
    """
    file = cmlon_path(results, name)
    file.parent.mkdir(parents=True, exist_ok=True)
    graph = cmlon.graph.copy()
    best = cmlon.best_fitness
    graph["best_fitness"] = math.nan if best is None else float(best)
    graph["eq_atol"] = float(cmlon.eq_atol)
    graph["minimize"] = int(cmlon.minimize)
    temporary = file.with_name(file.name + ".tmp")
    graph.write_graphml(str(temporary))
    temporary.replace(file)
    return file


def load_cmlon(results: Path, name: str) -> CMLON:
    """CMLON saved by :func:`save_cmlon` (without its source LON)."""
    graph = ig.Graph.Read_GraphML(str(cmlon_path(results, name)))
    best, eq_atol, minimize = (graph[attribute] for attribute in _CMLON_ATTRIBUTES)
    for attribute in _CMLON_ATTRIBUTES:
        del graph[attribute]
    if "id" in graph.vs.attributes():
        del graph.vs["id"]
    for sequence in (graph.vs, graph.es):
        if "Count" in sequence.attributes():
            sequence["Count"] = [int(count) for count in sequence["Count"]]
    return CMLON(
        graph=graph,
        best_fitness=None if math.isnan(best) else best,
        eq_atol=eq_atol,
        minimize=bool(minimize),
    )


def cmlon_path(results: Path, name: str) -> Path:
    """Location of the CMLON ``name``."""
    return Path(results) / CMLON_FOLDER / f"{name}.graphml"


def ioh_logs_path(results: Path, suite: str, optimizer: str) -> Path:
    """Folder of the IOH logs of ``optimizer`` on ``suite``."""
    return Path(results) / IOH_LOGS_FOLDER / suite / optimizer
