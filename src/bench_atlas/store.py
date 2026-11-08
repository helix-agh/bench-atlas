"""Results store: one Parquet file per analysis in a results folder.

Each analysis (``ela``, ``lon``, ...) is a single table ``<results>/<analysis>.parquet``.
Saving rewrites the whole file through a temporary file and an atomic rename, so the
table on disk is always complete. A lock file guards against two runs writing the same
table at once, which would lose rows.
"""

from __future__ import annotations

import os
import socket
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

__all__ = ["LockedError", "load", "load_all", "lock", "path", "save"]

VERSION_PREFIX = "version_"


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
def lock(results: Path, analysis: str) -> Iterator[None]:
    """Hold the lock of the ``analysis`` table for the duration of the block.

    The lock is a file created exclusively next to the table (``flock`` is not used:
    Lustre supports it only with a mount option). It holds the host and process id of
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
