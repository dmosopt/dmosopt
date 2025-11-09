"""
Comprehensive test suite for Fonseca et al. Algorithm 3 implementation.

Tests correctness, performance, and edge cases.
"""

import numpy as np
import time


from dmosopt.hv_box_decomposition import (
    HyperVolumeBoxDecomposition,
    compute_hypervolume_box_decomposition,
    compute_hypervolume_box_decomposition_batch,
)


class TestCorrectnessAnalytical:
    """Test against analytically known results."""

    def test_empty_set(self):
        """Empty point set should give zero hypervolume."""
        points = np.array([]).reshape(0, 2)
        ref = np.array([1.0, 1.0])

        hv = compute_hypervolume_box_decomposition(points, ref)
        assert hv == 0.0

    def test_single_point_2d(self):
        """Single point in 2D: rectangle to reference."""
        points = np.array([[1.0, 1.0]])
        ref = np.array([3.0, 3.0])

        hv = compute_hypervolume_box_decomposition(points, ref)
        expected = (3.0 - 1.0) * (3.0 - 1.0)  # 4.0

        assert np.isclose(hv, expected)

    def test_two_points_2d_orthogonal(self):
        """Two orthogonal points form L-shape."""
        points = np.array([[1.0, 2.0], [2.0, 1.0]])
        ref = np.array([3.0, 3.0])

        hv = compute_hypervolume_box_decomposition(points, ref)
        expected = 4.0

        assert np.isclose(hv, expected)

    def test_three_points_2d_staircase(self):
        """Three points forming staircase pattern."""
        points = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
        ref = np.array([4.0, 4.0])

        hv = compute_hypervolume_box_decomposition(points, ref)
        expected = 6.0  # Full 3x3 square for this configuration

        assert np.isclose(hv, expected, rtol=1e-6)

    def test_single_point_3d(self):
        """Single point in 3D: box to reference."""
        points = np.array([[1.0, 1.0, 1.0]])
        ref = np.array([2.0, 2.0, 2.0])

        hv = compute_hypervolume_box_decomposition(points, ref)
        expected = 1.0 * 1.0 * 1.0  # Unit cube

        assert np.isclose(hv, expected)

    def test_two_points_3d(self):
        """Two points in 3D."""
        points = np.array([[1.0, 1.0, 2.0], [1.0, 2.0, 1.0]])
        ref = np.array([3.0, 3.0, 3.0])

        hv = compute_hypervolume_box_decomposition(points, ref)
        assert hv > 0
        assert hv <= 8.0  # Can't exceed full cube

    def test_dominated_points_filtered(self):
        """Dominated points should be filtered out."""
        points = np.array(
            [
                [1.0, 1.0],
                [2.0, 2.0],  # Dominated by (1,1)
                [3.0, 0.5],
            ]
        )
        ref = np.array([4.0, 4.0])

        calc = HyperVolumeBoxDecomposition(ref)
        filtered = calc._filter_dominated(points)

        # Should keep (1,1) and (3,0.5), remove (2,2)
        assert len(filtered) == 2
        assert not np.any(np.all(filtered == [2.0, 2.0], axis=1))


class TestCorrectnessDimensions:
    """Test across different dimensions."""

    def test_1d(self):
        """1D is just a line segment."""
        points = np.array([[2.0]])
        ref = np.array([5.0])

        hv = compute_hypervolume_box_decomposition(points, ref)
        expected = 5.0 - 2.0

        assert np.isclose(hv, expected)

    def test_2d_algorithm_selection(self):
        """Test 2D with algorithm selection."""
        points = np.array([[1.0, 1.0]])
        ref = np.array([3.0, 3.0])

        hv_auto = compute_hypervolume_box_decomposition(points, ref, algorithm="auto")
        hv_2d = compute_hypervolume_box_decomposition(points, ref, algorithm="2d")

        assert np.isclose(hv_auto, hv_2d)

    def test_3d_algorithm_selection(self):
        """Test 3D with algorithm selection."""
        points = np.array([[1.0, 1.0, 1.0]])
        ref = np.array([2.0, 2.0, 2.0])

        hv_auto = compute_hypervolume_box_decomposition(points, ref, algorithm="auto")
        hv_3d = compute_hypervolume_box_decomposition(points, ref, algorithm="3d")

        assert np.isclose(hv_auto, hv_3d)

    def test_4d_fonseca(self):
        """Test 4D using full Fonseca algorithm."""
        np.random.seed(42)
        points = np.random.rand(10, 4) * 5
        ref = np.array([6.0] * 4)

        hv = compute_hypervolume_box_decomposition(points, ref, algorithm="fonseca")

        assert hv > 0
        assert hv <= 1.0  # Can't exceed unit hypercube scaled

    def test_high_dimensional(self):
        """Test 8D case."""
        np.random.seed(42)
        points = np.random.rand(5, 8) * 0.5
        ref = np.ones(8)

        hv = compute_hypervolume_box_decomposition(points, ref)

        assert hv > 0
        assert hv <= 1.0


class TestBatchProcessing:
    """Test batch interfaces."""

    def test_batch_single_ref(self):
        """Batch processing with single reference point."""
        fronts = [
            np.array([[1.0, 1.0]]),
            np.array([[0.5, 0.5]]),
            np.array([[2.0, 2.0]]),
        ]
        ref = np.array([3.0, 3.0])

        hvs = compute_hypervolume_box_decomposition_batch(fronts, ref)

        assert len(hvs) == 3
        assert hvs[1] > hvs[0]  # Better front has larger HV
        assert hvs[0] > hvs[2]  # Worse front has smaller HV

    def test_batch_multiple_refs(self):
        """Batch processing with multiple reference points."""
        fronts = [np.array([[1.0, 1.0]]), np.array([[1.0, 1.0]])]
        refs = np.array([[2.0, 2.0], [3.0, 3.0]])

        hvs = compute_hypervolume_box_decomposition_batch(fronts, refs)

        assert len(hvs) == 2
        assert hvs[1] > hvs[0]  # Larger ref gives larger HV


class TestPerformance:
    """Performance and scalability tests."""

    def test_scaling_with_points(self):
        """Test how performance scales with number of points."""
        ref = np.array([1.0, 1.0, 1.0])

        times = []
        for n in [10, 50, 100]:
            points = np.random.rand(n, 3) * 0.9

            start = time.time()
            hv = compute_hypervolume_box_decomposition(points, ref)
            elapsed = time.time() - start

            times.append(elapsed)
            print(f"n={n}: {elapsed:.4f}s, HV={hv:.6f}")

        # Should be roughly O(n log n) for 3D
        # So time ratio should be approximately (n2/n1) * log(n2/n1)
        ratio_10_to_50 = times[1] / times[0]
        expected_ratio = (50 / 10) * np.log2(50 / 10)

        print(f"Actual ratio (10->50): {ratio_10_to_50:.2f}")
        print(f"Expected ratio: {expected_ratio:.2f}")

        # Very loose bound - just check it's not exponential
        assert ratio_10_to_50 < 50  # Should be much less than O(n^2)

    def test_scaling_with_dimensions(self):
        """Test how performance scales with dimensions."""
        n_points = 20

        times = []
        for d in [3, 4, 5, 6]:
            points = np.random.rand(n_points, d) * 0.9
            ref = np.ones(d)

            start = time.time()
            hv = compute_hypervolume_box_decomposition(points, ref)
            elapsed = time.time() - start

            times.append(elapsed)
            print(f"d={d}: {elapsed:.4f}s, HV={hv:.6f}")

        # Should be O(n^(d-2) log n)
        # From d=3 to d=6 is 3 more dimensions
        # Ratio should be roughly n^3
        ratio = times[-1] / times[0]
        print(f"Time ratio (d=3->d=6): {ratio:.2f}")

        # Very loose bound
        assert ratio < n_points**4  # Should be much better than n^d


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_points_at_reference(self):
        """Points at reference point contribute nothing."""
        points = np.array([[3.0, 3.0]])
        ref = np.array([3.0, 3.0])

        hv = compute_hypervolume_box_decomposition(points, ref)
        assert hv == 0.0

    def test_all_dominated(self):
        """All points dominated should filter to minimal set."""
        points = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        ref = np.array([4.0, 4.0])

        calc = HyperVolumeBoxDecomposition(ref)
        filtered = calc._filter_dominated(points)

        assert len(filtered) == 1
        assert np.allclose(filtered[0], [1.0, 1.0])

    def test_negative_coordinates(self):
        """Should handle negative coordinates."""
        points = np.array([[-1.0, -1.0]])
        ref = np.array([1.0, 1.0])

        hv = compute_hypervolume_box_decomposition(points, ref)
        expected = 2.0 * 2.0

        assert np.isclose(hv, expected)

    def test_very_close_points(self):
        """Test numerical stability with very close points."""
        points = np.array([[1.0, 1.0], [1.0 + 1e-10, 1.0], [1.0, 1.0 + 1e-10]])
        ref = np.array([2.0, 2.0])

        hv = compute_hypervolume_box_decomposition(points, ref)

        # Should be very close to single point case
        expected = 1.0
        assert np.isclose(hv, expected, rtol=1e-8)


class TestFunctionalProperties:
    """Test functional programming properties."""

    def test_deterministic(self):
        """Same input should give same output."""
        np.random.seed(42)
        points = np.random.rand(20, 3)
        ref = np.array([1.5, 1.5, 1.5])

        hv1 = compute_hypervolume_box_decomposition(points, ref)
        hv2 = compute_hypervolume_box_decomposition(points, ref)

        assert hv1 == hv2

    def test_monotonic_in_reference(self):
        """Larger reference should give larger or equal HV."""
        points = np.array([[1.0, 1.0]])
        ref1 = np.array([2.0, 2.0])
        ref2 = np.array([3.0, 3.0])

        hv1 = compute_hypervolume_box_decomposition(points, ref1)
        hv2 = compute_hypervolume_box_decomposition(points, ref2)

        assert hv2 >= hv1

    def test_additive_for_disjoint_regions(self):
        """HV should be additive for non-overlapping fronts."""
        # Two points that don't overlap in dominated space
        points1 = np.array([[1.0, 3.0]])
        points2 = np.array([[3.0, 1.0]])
        combined = np.vstack([points1, points2])

        ref = np.array([4.0, 4.0])

        hv1 = compute_hypervolume_box_decomposition(points1, ref)
        hv2 = compute_hypervolume_box_decomposition(points2, ref)
        hv_combined = compute_hypervolume_box_decomposition(combined, ref)

        # For orthogonal points, should be additive
        # This is a special case - in general HV is NOT additive
        assert np.isclose(hv_combined, hv1 + hv2, rtol=0.1)


class TestMemoryEfficiency:
    """Test memory usage for large datasets."""

    def test_large_front_doesnt_explode_memory(self):
        """Large fronts should not cause memory issues."""
        # This is more of a smoke test
        points = np.random.rand(1000, 3) * 0.9
        ref = np.ones(3)

        hv = compute_hypervolume_box_decomposition(points, ref)

        assert hv > 0


# ============================================================================
# Benchmarking Utilities
# ============================================================================


def benchmark_algorithm(dimensions: list, n_points: list, n_trials: int = 3) -> dict:
    """
    Comprehensive benchmark across dimensions and point counts.

    Returns:
        Dictionary with timing results
    """
    results = {}

    for d in dimensions:
        results[d] = {}
        ref = np.ones(d)

        for n in n_points:
            times = []

            for trial in range(n_trials):
                points = np.random.rand(n, d) * 0.9

                start = time.time()
                _ = compute_hypervolume_box_decomposition(points, ref)
                elapsed = time.time() - start

                times.append(elapsed)

            results[d][n] = {
                "mean": np.mean(times),
                "std": np.std(times),
                "min": np.min(times),
                "max": np.max(times),
            }

    return results


def print_benchmark_results(results: dict):
    """Pretty-print benchmark results."""
    print("\nBenchmark Results")
    print("=" * 70)

    for d in sorted(results.keys()):
        print(f"\nDimension: {d}")
        print(
            f"{'Points':>10} | {'Mean (s)':>12} | {'Std (s)':>12} | {'Min (s)':>12} | {'Max (s)':>12}"
        )
        print("-" * 70)

        for n in sorted(results[d].keys()):
            r = results[d][n]
            print(
                f"{n:>10} | {r['mean']:>12.6f} | {r['std']:>12.6f} | {r['min']:>12.6f} | {r['max']:>12.6f}"
            )


if __name__ == "__main__":
    # Run basic tests
    print("Running basic correctness tests...")

    test_corr = TestCorrectnessAnalytical()
    test_corr.test_empty_set()
    test_corr.test_single_point_2d()
    test_corr.test_three_points_2d_staircase()

    print("Analytical tests passed")

    # Run performance tests
    print("\nRunning performance tests...")
    test_perf = TestPerformance()
    test_perf.test_scaling_with_points()
    test_perf.test_scaling_with_dimensions()

    print("\nPerformance tests passed")

    # Run benchmarks
    print("\nRunning benchmarks...")
    results = benchmark_algorithm(
        dimensions=[2, 3, 4, 5], n_points=[10, 50, 100], n_trials=3
    )
    print_benchmark_results(results)

    print("\nAll tests completed!")
