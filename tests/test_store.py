from pathlib import Path

import pandas as pd
import pytest

from bench_atlas import store


def _table(n: int = 3, version: str = "1.0") -> pd.DataFrame:
    return pd.DataFrame(
        {"problem": [f"p{i}" for i in range(n)], "value": range(n), "version_lib": version}
    )


def test_round_trip(tmp_path: Path) -> None:
    table = _table()
    store.save(tmp_path, "ela", table)
    pd.testing.assert_frame_equal(store.load(tmp_path, "ela"), table)
    assert [p.name for p in tmp_path.iterdir()] == ["ela.parquet"]


def test_missing_table_is_empty(tmp_path: Path) -> None:
    assert store.load(tmp_path / "nowhere", "ela").empty


def test_save_replaces_the_table(tmp_path: Path) -> None:
    store.save(tmp_path, "ela", _table(3))
    store.save(tmp_path, "ela", _table(5))
    assert len(store.load(tmp_path, "ela")) == 5


def test_load_all(tmp_path: Path) -> None:
    store.save(tmp_path, "ela", _table(2))
    store.save(tmp_path, "lon", _table(4))
    tables = store.load_all(tmp_path)
    assert list(tables) == ["ela", "lon"]
    assert [len(t) for t in tables.values()] == [2, 4]


def test_mixed_versions_warn(tmp_path: Path) -> None:
    store.save(tmp_path, "ela", pd.concat([_table(2, "1.0"), _table(2, "2.0")]))
    with pytest.warns(UserWarning, match="version_lib"):
        store.load(tmp_path, "ela")


def test_lock_is_exclusive_and_released(tmp_path: Path) -> None:
    with store.lock(tmp_path, "ela"):
        assert (tmp_path / "ela.parquet.lock").exists()
        with (
            pytest.raises(store.LockedError, match=r"ela\.parquet\.lock"),
            store.lock(tmp_path, "ela"),
        ):
            pass
    assert not (tmp_path / "ela.parquet.lock").exists()
    with store.lock(tmp_path, "ela"):
        pass


def test_lock_is_released_on_error(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError), store.lock(tmp_path, "ela"):
        raise RuntimeError
    assert not (tmp_path / "ela.parquet.lock").exists()


def _cmlon():
    from bench_atlas.lon import cmlon_from_runs, sample_run
    from bench_atlas.suites import bbob

    problem = bbob([2], [1])[21]
    runs = [sample_run(problem, seed, n_iter_no_change=20) for seed in range(6)]
    return cmlon_from_runs([(r["optima_x"], r["optima_f"]) for r in runs], 2)


def test_cmlon_round_trip(tmp_path: Path) -> None:
    cmlon = _cmlon()
    file = store.save_cmlon(tmp_path, "f22", cmlon)
    assert file == tmp_path / "cmlon" / "f22.graphml"
    assert not list((tmp_path / "cmlon").glob("*.tmp"))
    loaded = store.load_cmlon(tmp_path, "f22")
    assert loaded.graph.is_directed()
    assert loaded.graph.get_edgelist() == cmlon.graph.get_edgelist()
    for attribute in ("name", "Fitness", "Count"):
        assert loaded.graph.vs[attribute] == cmlon.graph.vs[attribute]
    assert loaded.graph.es["Count"] == cmlon.graph.es["Count"]
    assert (loaded.best_fitness, loaded.eq_atol, loaded.minimize) == (
        cmlon.best_fitness,
        cmlon.eq_atol,
        cmlon.minimize,
    )
    assert loaded.compute_network_metrics() == {
        **cmlon.compute_network_metrics(),
        "neutral": 0.0,
    }


def test_loaded_cmlon_can_be_plotted(tmp_path: Path) -> None:
    from lonkit import LONVisualizer

    store.save_cmlon(tmp_path, "f22", _cmlon())
    LONVisualizer().plot_2d(
        store.load_cmlon(tmp_path, "f22"), output_path=str(tmp_path / "f22.png")
    )
    assert (tmp_path / "f22.png").stat().st_size > 0


def test_empty_cmlon_round_trip(tmp_path: Path) -> None:
    from bench_atlas.lon import cmlon_from_runs

    store.save_cmlon(tmp_path, "empty", cmlon_from_runs([], 2))
    loaded = store.load_cmlon(tmp_path, "empty")
    assert loaded.n_vertices == 0
    assert loaded.best_fitness is None
