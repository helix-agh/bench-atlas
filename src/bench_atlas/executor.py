"""Cached, parallel execution of an analysis over benchmark problems.

One task is (problem, replicate). Its seed is derived from a global seed, the problem
id and the replicate (:func:`task_seed`), never from task order or worker, so results
don't depend on ``n_jobs``. Tasks run with joblib, each with single-threaded BLAS /
OpenMP. Finished rows are added to the analysis table of the results store
(:mod:`bench_atlas.store`); tasks already in the table are skipped, so an
interrupted run resumes where it stopped.
"""

from __future__ import annotations

import hashlib
import signal
import threading
import time
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from bench_atlas import store
from bench_atlas.problems import Problem, ProblemId

__all__ = ["FLUSH_SECONDS", "VERSIONED", "run", "task_seed"]

FLUSH_SECONDS = 10.0
VERSIONED = ("bench-atlas", "orivex", "lonkit", "ioh")

Compute = Callable[[Problem, int], Mapping[str, Any]]


def run(
    analysis: str,
    problems: Iterable[Problem],
    replicates: Iterable[int],
    compute: Compute,
    params: Mapping[str, Any],
    results: Path,
    n_jobs: int = 1,
    global_seed: int = 0,
    flush_seconds: float = FLUSH_SECONDS,
    log: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Rows of ``analysis`` for every problem and replicate, computing only missing ones.

    A row is ``problem, suite, function, instance, dimension, replicate, seed``, the
    ``params``, the ``version_*`` of :data:`VERSIONED` and ``compute(problem, seed)``.
    Rows of the table whose ``(problem, seed, *params)`` match a task are reused.
    New rows are saved at most every ``flush_seconds`` and when the run ends, also
    on ``KeyboardInterrupt`` and SIGTERM.

    Args:
        analysis: Table name in the results store.
        problems: Problems to analyse.
        replicates: Replicate numbers; each gets its own seed.
        compute: Picklable ``(problem, seed) -> features``, e.g. a ``functools.partial``.
        params: Analysis settings; part of the cache key and stored in every row.
        results: Folder of the results store.
        n_jobs: joblib ``n_jobs``; ``-1`` uses all CPUs.
        global_seed: Root of all task seeds.
        flush_seconds: Minimum time between two saves of the table.
        log: Receives progress messages.

    Returns:
        The requested rows in problem, then replicate order.
    """
    tasks = [
        (problem, replicate, task_seed(global_seed, problem.id, replicate))
        for problem in problems
        for replicate in replicates
    ]
    keys = [_key(str(problem.id), seed, params) for problem, _, seed in tasks]
    with store.lock(results, analysis):
        table = store.load(results, analysis)
        done = set(_table_keys(table, params))
        todo = [task for task, key in zip(tasks, keys, strict=True) if key not in done]
        _log(log, f"{analysis}: {len(tasks) - len(todo)} cached, {len(todo)} to compute")
        if todo:
            table = _execute(
                analysis, todo, compute, params, results, table, n_jobs, flush_seconds, log
            )
    index = {key: i for i, key in enumerate(_table_keys(table, params))}
    return table.iloc[[index[key] for key in keys]].reset_index(drop=True)


def task_seed(global_seed: int, problem_id: ProblemId, replicate: int) -> int:
    """Seed of one task, derived with ``numpy.random.SeedSequence``.

    The problem id enters through the first 8 bytes of its SHA-256 digest (the built-in
    ``hash`` changes between processes).
    """
    digest = hashlib.sha256(str(problem_id).encode()).digest()
    entropy = [global_seed, int.from_bytes(digest[:8], "little"), replicate]
    return int(np.random.SeedSequence(entropy).generate_state(1)[0])


def _execute(
    analysis: str,
    todo: list[tuple[Problem, int, int]],
    compute: Compute,
    params: Mapping[str, Any],
    results: Path,
    table: pd.DataFrame,
    n_jobs: int,
    flush_seconds: float,
    log: Callable[[str], None] | None,
) -> pd.DataFrame:
    """Run ``todo``, adding finished rows to ``table`` and saving it as they come.

    Every save writes ``table`` plus all rows computed so far, so a save retried after
    an interrupted one (e.g. SIGTERM inside :func:`store.save`) writes the same table
    again instead of adding rows twice.
    """
    versions = {f"version_{name.replace('-', '_')}": version(name) for name in VERSIONED}
    base = table
    new_rows: list[dict[str, Any]] = []
    saved = 0
    last_flush = time.monotonic()

    def flush() -> None:
        nonlocal table, saved, last_flush
        if len(new_rows) > saved:
            count = len(new_rows)
            new = pd.DataFrame(new_rows)
            current = new if base.empty else pd.concat([base, new], ignore_index=True)
            store.save(results, analysis, current)
            table, saved = current, count
            target = store.path(results, analysis)
            _log(log, f"{analysis}: {count}/{len(todo)} computed, saved to {target}")
        last_flush = time.monotonic()

    with _sigterm_as_exit():
        try:
            with joblib.parallel_config(backend="loky", inner_max_num_threads=1):
                rows = joblib.Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
                    joblib.delayed(_run_task)(
                        compute, problem, seed, _meta(problem, replicate, seed, params, versions)
                    )
                    for problem, replicate, seed in todo
                )
                for row in rows:
                    new_rows.append(row)
                    if time.monotonic() - last_flush >= flush_seconds:
                        flush()
        finally:
            flush()
    return table


def _run_task(
    compute: Compute, problem: Problem, seed: int, meta: dict[str, Any]
) -> dict[str, Any]:
    with threadpool_limits(limits=1):
        return {**meta, **compute(problem, seed)}


def _meta(
    problem: Problem,
    replicate: int,
    seed: int,
    params: Mapping[str, Any],
    versions: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "problem": str(problem.id),
        "suite": problem.id.suite,
        "function": problem.id.function,
        "instance": problem.id.instance,
        "dimension": problem.id.dimension,
        "replicate": replicate,
        "seed": seed,
        **params,
        **versions,
    }


def _key(problem: str, seed: int, params: Mapping[str, Any]) -> tuple[Any, ...]:
    return (problem, seed, *params.values())


def _table_keys(table: pd.DataFrame, params: Mapping[str, Any]) -> Iterator[tuple[Any, ...]]:
    columns = ["problem", "seed", *params]
    if table.empty or any(column not in table for column in columns):
        return iter(())
    return zip(*(table[column].tolist() for column in columns), strict=True)


@contextmanager
def _sigterm_as_exit() -> Generator[None, None, None]:
    """Turn SIGTERM (e.g. a Slurm time limit) into ``SystemExit``, so rows get saved."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def terminate(signum: int, _frame: Any) -> None:
        raise SystemExit(128 + signum)

    previous = signal.signal(signal.SIGTERM, terminate)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def _log(log: Callable[[str], None] | None, message: str) -> None:
    if log is not None:
        log(message)
