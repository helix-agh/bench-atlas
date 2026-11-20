import numpy as np
import pytest

from bench_atlas.diversity import (
    coverage,
    diversity,
    novelty,
    radius,
    standardize,
    vendi_score,
)


def test_vendi_of_identical_points_is_one() -> None:
    assert vendi_score(np.ones((10, 3)), bandwidth=1.0) == pytest.approx(1.0)


def test_vendi_of_far_apart_points_is_their_number() -> None:
    points = 1e3 * np.eye(10)
    assert vendi_score(points, bandwidth=1.0) == pytest.approx(10.0)


def test_vendi_of_two_points_matches_closed_form() -> None:
    # K / 2 has eigenvalues (1 +- s) / 2 with s the kernel value of the pair.
    s = np.exp(-(1.5**2) / 2)
    eigenvalues = np.array([1 + s, 1 - s]) / 2
    expected = np.exp(-np.sum(eigenvalues * np.log(eigenvalues)))
    assert vendi_score(np.array([[0.0], [1.5]]), bandwidth=1.0) == pytest.approx(expected)


def test_standardize_is_robust_and_rejects_constant_features() -> None:
    pool = np.array([[1.0], [2.0], [3.0], [4.0], [100.0]])
    z = standardize(pool)
    assert np.median(z) == 0
    assert np.subtract(*np.percentile(z, [75, 25])) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="zero IQR"):
        standardize(np.column_stack([pool, np.ones(5)]))


def test_radius_is_median_distance_to_kth_neighbour() -> None:
    pool = np.arange(10.0)[:, None]
    # The 2nd nearest other point of an interior point is at distance 1, of an end 2.
    assert radius(pool, k=2) == 1.0
    assert radius(pool, k=5) == 3.0


def test_coverage_of_the_pool_is_one() -> None:
    pool = np.random.default_rng(0).random((50, 3))
    assert coverage(pool, pool, eps=0.0) == 1.0
    assert coverage(pool, pool[:1], eps=0.0) == 1 / 50


def test_novelty_counts_problems_farther_than_eps() -> None:
    reference = np.zeros((1, 1))
    assert novelty(np.array([[0.5], [2.0], [3.0]]), reference, eps=1.0) == pytest.approx(2 / 3)


def _pool() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1)
    features = np.vstack(
        [rng.normal(0, 1, (40, 4)), rng.normal(5, 1, (30, 4)), rng.random((10, 4))]
    )
    benchmarks = np.array(["bbob"] * 40 + ["far"] * 30 + ["small"] * 10)
    return features, benchmarks


def test_diversity_table() -> None:
    table = diversity(*_pool()).set_index("benchmark")
    assert list(table.index) == ["bbob", "far", "small"]
    assert table.loc["bbob", "novelty"] == 0.0
    assert table.loc["far", "novelty"] == 1.0
    assert table.loc[["bbob", "far"], "subsampled"].all()
    small = table.loc["small"]
    assert not small["subsampled"]
    assert small["coverage_low"] == small["coverage"] == small["coverage_high"]
    for metric in ["coverage", "vendi"]:
        assert (table[f"{metric}_low"] <= table[metric]).all()
        assert (table[metric] <= table[f"{metric}_high"]).all()
    assert (table["vendi"] <= np.minimum(table["size"], 24)).all()


def test_whole_pool_as_one_benchmark_covers_it() -> None:
    features, _ = _pool()
    table = diversity(features, np.array(["bbob"] * len(features)), subset_size=len(features))
    assert table["coverage"].item() == 1.0


def test_diversity_subsets_are_seeded_per_benchmark() -> None:
    features, benchmarks = _pool()
    first = diversity(features, benchmarks, seed=3)
    assert first.equals(diversity(features, benchmarks, seed=3))
    assert not first.equals(diversity(features, benchmarks, seed=4))
    # The subsets of a benchmark don't depend on where the other benchmarks are listed.
    order = np.concatenate([np.flatnonzero(benchmarks == b) for b in ["far", "small", "bbob"]])
    reordered = diversity(features[order], benchmarks[order], seed=3).set_index("benchmark")
    expected = first.set_index("benchmark").loc["far", "vendi"]
    assert reordered.loc["far", "vendi"] == pytest.approx(expected)


def test_diversity_requires_the_reference() -> None:
    features, benchmarks = _pool()
    with pytest.raises(ValueError, match="reference"):
        diversity(features, benchmarks, reference="cec")
