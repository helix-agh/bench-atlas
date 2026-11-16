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
