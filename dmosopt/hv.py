"""Adaptive hypervolume computation with Yang et al. box decomposition

This module provides adaptive hypervolume computation strategies with
automatic algorithm selection based on problem dimensionality.

Key optimizations:
1. Hybrid algorithm selection based on dimensionality
2. Lacour et al. (2016) box decomposition for low and medium dimensions (1-10 objectives)
3. Monte Carlo approximation for high dimensions (>10 objectives)

References:

Renaud Lacour, Kathrin Klamroth, Carlos M. Fonseca, A Box
Decomposition Algorithm to Compute the Hypervolume Indicator,
Computers & Operations Research, 2016.

"""

import numpy as np
from typing import Tuple, Optional
from hv_box_decomposition import compute_hypervolume_box_decomposition
from hv_adaptive import (
    ApproximationResult,
    compute_hypervolume_fpras,
    compute_hypervolume_mcm2rv,
    compute_hypervolume_hybrid,
)


def filter_dominated(points: np.ndarray, minimize: bool = True) -> np.ndarray:
    """
    Filter out dominated points to get non-dominated set (pure function).

    Uses vectorized operations for efficiency.

    Args:
        points: Array of shape (n_points, n_objectives)
        minimize: If True, filter for minimization (default)

    Returns:
        Array of non-dominated points
    """
    n = points.shape[0]
    if n == 0:
        return points.copy()

    # Vectorized dominance check
    dominated = np.zeros(n, dtype=bool)

    if minimize:
        # For minimization: point i dominates j if i <= j in all dims and i < j in at least one
        for i in range(n):
            if not dominated[i]:
                leq = np.all(points[i] <= points, axis=1)
                lt = np.any(points[i] < points, axis=1)
                dominates_i = leq & lt
                dominates_i[i] = False  # A point doesn't dominate itself
                dominated[dominates_i] = True
    else:
        # For maximization: point i dominates j if i >= j in all dims and i > j in at least one
        for i in range(n):
            if not dominated[i]:
                geq = np.all(points[i] >= points, axis=1)
                gt = np.any(points[i] > points, axis=1)
                dominates_i = geq & gt
                dominates_i[i] = False
                dominated[dominates_i] = True

    return points[~dominated].copy()


# ============================================================================
# Adaptive Hypervolume Computation
# ============================================================================


class AdaptiveHyperVolume:
    """
    Adaptive hypervolume computation with automatic algorithm selection.

    Automatically chooses the best algorithm based on problem dimensionality:
    - Box decomposition for 1 <= d <= 10 (efficient for low and medium dimensions)
    - Monte Carlo approximation for d > 10 (only practical option for high dimensions)

    This hybrid approach provides optimal performance across all dimensions.

    """

    def __init__(
        self,
        ref_point: np.ndarray,
        dimension_threshold_exact: int = 10,
        monte_carlo_samples: int = 100000,
        use_adaptive_mc: bool = True,  # Adaptive MC parameters
        mc_epsilon: float = 0.01,
        mc_delta: float = 0.25,
    ):
        """
        Initialize the hypervolume calculator.

        Parameters
        ----------
        ref_point : np.ndarray
            Reference point for hypervolume computation (for minimization:
            worst acceptable values; for maximization: nadir point)
        dimension_threshold_exact : int
            Use exact algorithms for d < this value (default: 10)
            For d >= this value, use Monte Carlo approximation
        monte_carlo_samples : int
            Number of samples for basic Monte Carlo approximation (default: 100000)
        use_adaptive_mc: If True, use adaptive Monte Carlo algorithms (recommended)
        mc_epsilon: Approximation error bound for adaptive Monte Carlo methods
        mc_delta: Error probability for adaptive Monte Carlo methods
        """
        self.ref_point = np.asarray(ref_point, dtype=np.float64)
        self.n_objectives = len(ref_point)
        self.dimension_threshold_exact = dimension_threshold_exact
        self.monte_carlo_samples = monte_carlo_samples
        self.use_adaptive_mc = use_adaptive_mc
        self.mc_epsilon = mc_epsilon
        self.mc_delta = mc_delta

    def compute_hypervolume(
        self,
        pareto_front: np.ndarray,
        algorithm: Optional[str] = None,
        verbose: bool = False,
    ) -> float:
        """
        Compute hypervolume using automatically selected or specified algorithm.

        Parameters
        ----------
        pareto_front : np.ndarray
            Pareto front points, shape (n_points, n_objectives)
        algorithm : str, optional
            Force specific algorithm:
                       - 'box': Box decomposition (exact, d<10)
                       - 'monte_carlo': Standard Monte Carlo (approximate)
                       - 'fpras': FPRAS adaptive (approximate)
                       - 'mcm2rv': MCM2RV adaptive (approximate)
                       - 'hybrid': Hybrid adaptive (approximate, recommended)
                       - None/'auto': Automatic selection
            Default is automatic selection.
        verbose : bool
            Print algorithm selection info

        Returns
        -------
        float
            Hypervolume value
        """
        pareto_front = np.asarray(pareto_front, dtype=np.float64)

        if len(pareto_front) == 0:
            return 0.0

        # Ensure all points dominate reference point (for minimization: <= ref)
        valid_mask = np.all(self.ref_point > pareto_front, axis=1)
        if not np.any(valid_mask):
            return 0.0
        points = pareto_front[valid_mask]

        # Select algorithm
        if algorithm is None or algorithm == "auto":
            if self.n_objectives < self.dimension_threshold_exact:
                algorithm = "box"
            elif self.use_adaptive_mc:
                algorithm = "hybrid"
            else:
                algorithm = "monte_carlo"

        if verbose:
            print(
                f"Computing hypervolume using '{algorithm}' algorithm "
                f"for {self.n_objectives} objectives, {len(points)} points"
            )

        # Dispatch to appropriate algorithm
        if algorithm in ["fpras", "mcm2rv", "hybrid"]:
            return self._compute_adaptive_mc(pareto_front, algorithm, verbose)
        elif algorithm == "monte_carlo":
            return self._compute_standard_mc(
                pareto_front, n_samples=self.monte_carlo_samples
            )
        elif algorithm == "box":
            return compute_hypervolume_box_decomposition(points, self.ref_point)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def _compute_standard_mc(
        self, points: np.ndarray, n_samples: int = 100000
    ) -> float:
        """
        Monte Carlo approximation for high dimensions (d > 10).

        Parameters
        ----------
        points : np.ndarray
            Pareto front points
        n_samples : int
            Number of Monte Carlo samples

        Returns
        -------
        float
            Approximate hypervolume value
        """
        # Find bounding box for sampling
        lower_bounds = np.min(points, axis=0)
        upper_bounds = self.ref_point

        # Generate random samples in the dominated space
        samples = np.random.uniform(
            lower_bounds, upper_bounds, size=(n_samples, self.n_objectives)
        )

        # Check which samples are dominated by at least one Pareto point
        # Vectorized for efficiency
        dominated = np.zeros(n_samples, dtype=bool)

        # Process in chunks to avoid memory issues
        chunk_size = 10000
        while np.sum(dominated) == 0:
            for i in range(0, len(points), chunk_size):
                chunk = points[i : i + chunk_size]
                # Sample is dominated if all objectives are >= some Pareto point
                dominated |= np.any(
                    np.all(samples[:, None, :] >= chunk[None, :, :], axis=2), axis=1
                )

            # Re-generate random samples if dominated is zero
            samples = np.random.uniform(
                lower_bounds, upper_bounds, size=(n_samples, self.n_objectives)
            )

        # Estimate hypervolume as fraction of dominated samples
        total_volume = np.prod(upper_bounds - lower_bounds)
        dominated_fraction = np.mean(dominated)

        return float(total_volume * dominated_fraction)

    def _compute_adaptive_mc(
        self, points: np.ndarray, algorithm: str, verbose: bool = False
    ) -> float:
        """Compute using adaptive Monte Carlo methods."""
        if algorithm == "fpras":
            result = compute_hypervolume_fpras(
                points, self.ref_point, self.mc_epsilon, self.mc_delta, minimize=True
            )
        elif algorithm == "mcm2rv":
            result = compute_hypervolume_mcm2rv(
                points, self.ref_point, self.mc_epsilon, self.mc_delta, minimize=True
            )
        else:  # hybrid
            result = compute_hypervolume_hybrid(
                points,
                self.ref_point,
                self.mc_epsilon,
                self.mc_delta,
                minimize=True,
                verbose=verbose,
            )

        if verbose:
            print(f"  Actual algorithm: {result.algorithm_used}")
            print(f"  Comparisons: {result.num_comparisons}")
            print(f"  Samples: {result.num_samples}")

        return result.hypervolume

    def compute_hypervolume_with_confidence(
        self,
        pareto_front: np.ndarray,
        n_runs: int = 10,
        confidence: float = 0.95,
        verbose: bool = False,
    ) -> Tuple[float, float]:
        """
        Compute hypervolume with confidence interval (Monte Carlo only).

        Useful for high-dimensional problems where exact computation is
        infeasible and you want to quantify approximation uncertainty.

        Parameters
        ----------
        pareto_front : np.ndarray
            Pareto front points
        n_runs : int
            Number of independent Monte Carlo runs
        confidence : float
            Confidence level (default: 0.95 for 95% CI)

        Returns
        -------
        Tuple[float, float]
            (mean_hypervolume, confidence_interval_width)
        """
        if self.n_objectives < self.dimension_threshold_exact:
            # Use exact algorithm, no uncertainty
            hv = self.compute_hypervolume(pareto_front)
            return hv, 0.0

        # Run multiple Monte Carlo estimates
        estimates = np.array(
            [
                self._compute_adaptive_mc(
                    pareto_front, algorithm="hybrid", verbose=verbose
                )
                for _ in range(n_runs)
            ]
        )

        # Compute mean and confidence interval
        mean_hv = np.mean(estimates)
        std_hv = np.std(estimates, ddof=1)

        # t-distribution critical value for confidence interval
        from scipy.stats import t

        t_crit = t.ppf((1 + confidence) / 2, n_runs - 1)
        ci_half_width = t_crit * std_hv / np.sqrt(n_runs)

        return float(mean_hv), float(ci_half_width)

    def compute_hypervolume_with_statistics(
        self, pareto_front: np.ndarray, algorithm: str = "hybrid", verbose: bool = False
    ) -> ApproximationResult:
        """
        Compute hypervolume and return detailed statistics.

        Parameters
        ----------
        pareto_front : np.ndarray
            Pareto front points
        algorithm : str
            'fpras', 'mcm2rv', or 'hybrid'
        verbose : bool
            Print progress information

        Returns
        -------
        ApproximationResult
            Result with hypervolume and detailed statistics
        """
        pareto_front = filter_dominated(pareto_front, minimize=True)

        if algorithm == "fpras":
            return compute_hypervolume_fpras(
                pareto_front,
                self.ref_point,
                self.mc_epsilon,
                self.mc_delta,
                minimize=True,
            )
        elif algorithm == "mcm2rv":
            return compute_hypervolume_mcm2rv(
                pareto_front,
                self.ref_point,
                self.mc_epsilon,
                self.mc_delta,
                minimize=True,
            )
        elif algorithm == "hybrid":
            return compute_hypervolume_hybrid(
                pareto_front,
                self.ref_point,
                self.mc_epsilon,
                self.mc_delta,
                minimize=True,
                verbose=verbose,
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")


# ============================================================================
# Utility Functions
# ============================================================================


def performance_comparison():
    """
    Compare performance of different implementations.
    """
    import time

    print("=" * 70)
    print("Hypervolume Performance Comparison")
    print("=" * 70)

    # Test different dimensionalities
    dimensions_to_test = [3, 5, 8, 10, 15, 20]
    dimensions_to_test = [10, 15, 20]
    n_points = 250

    for d in dimensions_to_test:
        print(f"\nDimensions: {d}, Points: {n_points}")

        # Generate random Pareto front
        np.random.seed(42)
        ref_point = np.full(d, 10.0)

        # Generate points that are somewhat Pareto-optimal
        points = []
        for _ in range(n_points):
            point = np.random.uniform(1, 9, d)
            # Make them tend toward Pareto front
            point = point * (1 - 0.5 * np.random.random(d))
            points.append(point)
        pareto_front = np.array(points)

        # Test AdaptiveHyperVolume with automatic selection
        hv_opt = AdaptiveHyperVolume(ref_point)
        start = time.time()
        result_opt = hv_opt.compute_hypervolume(pareto_front, verbose=d >= 10)
        time_opt = time.time() - start

        # Determine which algorithm was used
        if d < 10:
            algo_used = "Box Decomposition"
        else:
            algo_used = "Monte Carlo"

        print(f"  {algo_used:35s}: {time_opt:.6f}s, HV={result_opt:.4f}")

        # Show Monte Carlo explicitly
        start = time.time()
        result_mc = hv_opt.compute_hypervolume(
            pareto_front, algorithm="hybrid" if d < 10 else "monte_carlo", verbose=True
        )
        time_mc = time.time() - start
        print(f"  {'Monte Carlo (explicit)':35s}: {time_mc:.6f}s, HV={result_mc:.4f}")


def test_adaptive_algorithms():
    """Test the adaptive Monte Carlo algorithms."""
    print("=" * 80)
    print("Adaptive Monte Carlo Hypervolume Demonstration")
    print("=" * 80)

    # Test case 1: Linear front (low overlap - FPRAS should be better)
    print("\nTest 1: Linear Pareto front (20D, 100 points)")
    print("-" * 80)

    np.random.seed(42)
    d = 20
    n = 100
    ref = np.full(d, 10.0)

    # Generate linear front
    points = []
    for i in range(n):
        t = i / n
        point = np.full(d, 10.0 * (1 - t) + 1.0 * t)
        point += np.random.normal(0, 0.1, d)
        points.append(point)
    linear_front = np.array(points)

    hv_calc = AdaptiveHyperVolume(ref, mc_epsilon=0.01, mc_delta=0.25)

    # Test all three algorithms
    for algo in ["fpras", "mcm2rv", "hybrid"]:
        result = hv_calc.compute_hypervolume_with_statistics(
            linear_front, algorithm=algo
        )
        print(
            f"{algo.upper():10s}: HV={result.hypervolume:12.6f}, "
            f"Samples={result.num_samples:6d}, "
            f"Comparisons={result.num_comparisons:8d}, "
            f"Final={result.algorithm_used}"
        )

    # Test case 2: Convex front (high overlap - MCM2RV should be better)
    print("\nTest 2: Convex Pareto front (20D, 100 points)")
    print("-" * 80)

    # Generate convex front
    points = []
    for i in range(n):
        # Generate random direction
        direction = np.random.normal(0, 1, d)
        direction = direction / np.linalg.norm(direction)
        # Scale to create convex shape
        radius = np.random.uniform(1, 5)
        point = 5.0 + radius * direction
        points.append(point)
    convex_front = filter_dominated(np.array(points), minimize=True)

    for algo in ["hybrid", "fpras", "mcm2rv"]:
        result = hv_calc.compute_hypervolume_with_statistics(
            convex_front, algorithm=algo
        )
        print(
            f"{algo.upper():10s}: HV={result.hypervolume:12.6f}, "
            f"Samples={result.num_samples:6d}, "
            f"Comparisons={result.num_comparisons:8d}, "
            f"Final={result.algorithm_used}"
        )

    # Test case 3: Hybrid algorithm decision-making
    print("\nTest 3: Hybrid algorithm with verbose output (10D, 50 points)")
    print("-" * 80)

    d = 10
    n = 50
    ref = np.full(d, 10.0)
    np.random.seed(123)
    test_front = np.random.uniform(2, 8, size=(n, d))
    test_front = filter_dominated(test_front, minimize=True)

    hv_calc = AdaptiveHyperVolume(ref, mc_epsilon=0.02, mc_delta=0.25)
    result = hv_calc.compute_hypervolume_with_statistics(
        test_front, algorithm="hybrid", verbose=True
    )

    print(f"\nFinal result: HV={result.hypervolume:.6f}")
    print(f"Algorithm chosen: {result.algorithm_used}")
    print(f"Total comparisons: {result.num_comparisons}")


if __name__ == "__main__":
    # test_adaptive_algorithms()

    # Run performance comparison
    # performance_comparison()

    # Example usage
    print("\n\nHypervolume Examples:")
    print("=" * 70)

    # 2D example
    ref_point_2d = np.array([10.0, 10.0])
    pareto_2d = np.array([[2.0, 8.0], [4.0, 4.0], [8.0, 2.0]])

    hv_opt = AdaptiveHyperVolume(ref_point_2d)
    hv = hv_opt.compute_hypervolume(pareto_2d, verbose=True)
    print(f"2D Hypervolume: {hv:.4f}\n")

    # 5D example (uses box decomposition)
    ref_point_5d = np.full(5, 10.0)
    np.random.seed(42)
    pareto_5d = np.random.uniform(2, 8, size=(30, 5))

    hv_opt_5d = AdaptiveHyperVolume(ref_point_5d)
    hv_5d = hv_opt_5d.compute_hypervolume(pareto_5d, verbose=True)
    print(f"5D Hypervolume: {hv_5d:.4f}\n")

    # High-dimensional example (uses Monte Carlo)
    ref_point_30d = np.full(30, 10.0)
    np.random.seed(42)
    pareto_30d = np.random.uniform(2, 8, size=(100, 30))

    hv_opt_30d = AdaptiveHyperVolume(ref_point_30d)

    hv, ci = hv_opt_30d.compute_hypervolume_with_confidence(
        pareto_30d, n_runs=5, confidence=0.95, verbose=True
    )
    print(f"30D Hypervolume: {hv:.2e} +/- {ci:.2e}")
    print(f"Relative uncertainty: +/-{100 * ci / hv:.2f}%")
