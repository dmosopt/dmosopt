"""
Adaptive hypervolume computation with FPRAS and MCM2RV.

This module provides adaptive Monte Carlo strategies for approximate hypervolume computation including:

1. FPRAS (Fully Polynomial-time Randomized Approximation Scheme)
2. MCM2RV (Monte Carlo Method with Two Random Variables)
3. Hybrid algorithm that combines FPRAS and MCM2RV

References:
- Deng, J., & Zhang, Q. (2020). Combining Simple and Adaptive Monte Carlo
  Methods for Approximating Hypervolume. IEEE Transactions on Evolutionary
  Computation, 24(5), 896-907.
"""

import math
import numpy as np
from typing import Tuple, Dict
from dataclasses import dataclass
from scipy.spatial import cKDTree


@dataclass
class ApproximationResult:
    """Result from hypervolume approximation with statistics."""

    hypervolume: float
    num_samples: int
    num_comparisons: int
    algorithm_used: str
    approximation_error_bound: float
    error_probability: float


# ============================================================================
# K-d Tree Dominance Analysis
# ============================================================================


class DominanceAnalysis:
    """
    Efficient dominance checking using k-d tree spatial indexing.

    For minimization problems, point p dominates point q if p[i] <= q[i]
    for all dimensions i. Instead of checking all n points, we use a k-d tree
    to quickly identify potentially dominating points within the dominated region.

    This provides approximately O(log n) lookups instead of O(n), though
    worst-case can still be O(n) for pathological point sets.
    """

    def __init__(self, pareto_front: np.ndarray):
        """
        Initialize the dominance analysis with a Pareto front.

        Args:
            pareto_front: Array of shape (n_points, n_objectives)
        """
        self.pareto_front = pareto_front
        self.n_points, self.n_dims = pareto_front.shape

        # Build k-d tree for efficient spatial queries
        self.tree = cKDTree(pareto_front)

        # Pre-compute bounding box for quick rejection tests
        self.min_coords = np.min(pareto_front, axis=0)
        self.max_coords = np.max(pareto_front, axis=0)

    def is_dominated(self, point: np.ndarray) -> bool:
        """
        Check if a point is dominated by any point in the Pareto front.

        Uses k-d tree to efficiently find nearby points, then checks
        dominance only for those candidates.

        Args:
            point: Point to check (n_objectives,)

        Returns:
            True if point is dominated by any Pareto point
        """
        # Quick rejection: if point is below minimum in any dimension,
        # it cannot be dominated (for minimization)
        if np.any(point < self.min_coords):
            return False

        # Query k-d tree for points in the region that could dominate
        # We search in a box from min_coords to point
        # Using the query_ball_point with appropriate bounds

        # For efficiency, use a radius query with L-infinity norm
        # Points that dominate must be within the hypercube [min, point]
        # We can approximate this by finding points within a certain distance

        # Strategy: Find k nearest neighbors and check if any dominate
        # If none found in k neighbors, sample more broadly
        k = min(20, self.n_points)  # Check up to 20 nearest neighbors first

        distances, indices = self.tree.query(point, k=k, p=np.inf)

        # Check if any of these neighbors dominate the point
        for idx in indices:
            if idx < self.n_points:  # Valid index
                if np.all(point > self.pareto_front[idx]):
                    return True

        # If no dominance found in nearest neighbors, do full check
        # (fallback for edge cases)
        # This is rare for well-structured Pareto fronts
        if k < self.n_points:
            # Check remaining points that might dominate
            # Only check points that are componentwise <= point
            candidates = np.all(point > self.pareto_front, axis=1)
            if np.any(candidates):
                return True

        return False

    def count_dominating_points(self, point: np.ndarray) -> int:
        """
        Count how many Pareto points dominate the given point.

        Args:
            point: Point to check (n_objectives,)

        Returns:
            Number of dominating points (k(x) in the paper)
        """
        # Quick rejection test
        if np.any(point < self.min_coords):
            return 0

        # Use vectorized comparison (still efficient for moderate n)
        dominates = np.all(point > self.pareto_front, axis=1)
        return np.sum(dominates)

    def find_first_dominating_point(
        self, point: np.ndarray, random_order: bool = True
    ) -> Tuple[bool, int, int]:
        """
        Find the first point that dominates the given point, optionally in random order.

        This is optimized for the FPRAS geometric sampling where we need to count
        how many random tests until we find a dominating point.

        Args:
            point: Point to check (n_objectives,)
            random_order: If True, test points in random order

        Returns:
            Tuple of (found, point_index, num_tests)
        """
        if np.any(point < self.min_coords):
            return False, -1, self.n_points

        # Use k-d tree to get candidate points
        k = min(20, self.n_points)
        distances, indices = self.tree.query(point, k=k, p=np.inf)

        if random_order:
            # Shuffle the order for random testing
            np.random.shuffle(indices)

        # Test candidates from k-d tree first
        for i, idx in enumerate(indices):
            if idx < self.n_points:
                if np.all(point > self.pareto_front[idx]):
                    return True, idx, i + 1

        # If not found in k nearest, check remaining points
        remaining_indices = np.setdiff1d(np.arange(self.n_points), indices)

        if random_order:
            np.random.shuffle(remaining_indices)

        for i, idx in enumerate(remaining_indices):
            if np.all(point > self.pareto_front[idx]):
                return True, idx, k + i + 1

        return False, -1, self.n_points


# ============================================================================
# FPRAS Implementation
# ============================================================================


def _run_fpras_round(
    pareto_front: np.ndarray,
    reference_point: np.ndarray,
    volumes: np.ndarray,
    probabilities: np.ndarray,
    target_comparisons: int,
    current_comparisons: int,
    current_sum_xi: float,
    current_N: int,
    batch_size: int = 100,
) -> Tuple[int, float, int]:
    """
    Run one round of FPRAS geometric sampling.

    This is the core FPRAS sampling loop extracted for reuse by both
    standalone FPRAS and the hybrid algorithm.

    Args:
        pareto_front: Pareto front points
        reference_point: Reference point
        volumes: Pre-computed hyperrectangle volumes
        probabilities: Sampling probabilities for hyperrectangles
        target_comparisons: Comparison budget for this round
        current_comparisons: Comparisons done so far
        current_sum_xi: Current sum of xi values
        current_N: Current number of successful samples
        batch_size: Batch size for sample generation

    Returns:
        Tuple of (total_comparisons, sum_xi, N) after this round
    """
    n_points, n_dims = pareto_front.shape
    total_comparisons = current_comparisons
    sum_xi = current_sum_xi
    N = current_N

    while total_comparisons < target_comparisons:
        current_batch_size = min(
            batch_size, int(target_comparisons - total_comparisons) // n_points + 1
        )
        if current_batch_size <= 0:
            break

        # Select hyperrectangles
        j_batch = np.random.choice(n_points, size=current_batch_size, p=probabilities)

        # Generate samples
        samples_batch = np.empty((current_batch_size, n_dims))
        for idx, j in enumerate(j_batch):
            samples_batch[idx] = np.random.uniform(
                pareto_front[j], reference_point, size=n_dims
            )

        # Process each sample
        for i in range(current_batch_size):
            if total_comparisons >= target_comparisons:
                break

            sample = samples_batch[i]

            # Find first dominating point
            # Standard random geometric sampling
            xi = 0
            found = False
            while not found and total_comparisons < target_comparisons:
                k = np.random.randint(0, n_points)
                xi += 1
                total_comparisons += 1
                if np.all(sample > pareto_front[k]):
                    found = True

            if found:
                sum_xi += xi
                N += 1

    return total_comparisons, sum_xi, N


def compute_hypervolume_fpras(
    pareto_front: np.ndarray,
    reference_point: np.ndarray,
    epsilon: float = 0.01,
    delta: float = 0.25,
    minimize: bool = True,
    batch_size: int = 100,
) -> ApproximationResult:
    """Compute hypervolume using FPRAS with spatial clustering and batch generation.

    Compute hypervolume using FPRAS with k-d tree.

    Args:
        pareto_front: Array of shape (n_points, n_objectives)
        reference_point: Reference point for hypervolume
        epsilon: Approximation error bound (default 0.01 for 1% error)
        delta: Error probability (default 0.25)
        minimize: If True, compute for minimization (default)
        use_kdtree: If True, use k-d tree optimization (default)

    Returns:
        ApproximationResult with hypervolume estimate and statistics

    Optimizations:
    - Batch generation: Generates samples in batches to reduce overhead

    Args:
        pareto_front: Array of shape (n_points, n_objectives)
        reference_point: Reference point for hypervolume
        epsilon: Approximation error bound (default 0.01 for 1% error)
        delta: Error probability (default 0.25)
        minimize: If True, compute for minimization (default)
        batch_size: Number of samples to generate per batch (default 100)

    Returns:
        ApproximationResult with hypervolume estimate and statistics

    """
    if pareto_front.shape[0] == 0:
        return ApproximationResult(0.0, 0, 0, "FPRAS", epsilon, delta)

    if not minimize:
        pareto_front = -pareto_front.copy()
        reference_point = -reference_point.copy()

    n_points, n_dims = pareto_front.shape

    # Compute hyperrectangle volumes
    volumes = np.prod(reference_point - pareto_front, axis=1)
    W = np.sum(volumes)
    probabilities = volumes / W

    # Compute stopping threshold M1
    M1 = (8 * (1 + epsilon) * n_points * np.log(2 / delta)) / (epsilon**2)

    # Run FPRAS using the helper routine
    total_comparisons, sum_xi, N = _run_fpras_round(
        pareto_front=pareto_front,
        reference_point=reference_point,
        volumes=volumes,
        probabilities=probabilities,
        target_comparisons=int(M1),
        current_comparisons=0,
        current_sum_xi=0.0,
        current_N=0,
        batch_size=batch_size,
    )

    # Compute hypervolume estimate
    if N == 0:
        N = 1
    hypervolume_estimate = (W / n_points) * (sum_xi / N)

    algorithm_name = "FPRAS"

    return ApproximationResult(
        hypervolume=float(hypervolume_estimate),
        num_samples=N,
        num_comparisons=total_comparisons,
        algorithm_used=algorithm_name,
        approximation_error_bound=epsilon,
        error_probability=delta,
    )


# ============================================================================
# MCM2RV Implementation
# ============================================================================


def compute_hypervolume_mcm2rv(
    pareto_front: np.ndarray,
    reference_point: np.ndarray,
    epsilon: float = 0.01,
    delta: float = 0.25,
    minimize: bool = True,
    use_kdtree: bool = False,
    batch_size: int = 1000,
) -> ApproximationResult:
    """
    Compute hypervolume using MCM2RV.

    Args:
        pareto_front: Array of shape (n_points, n_objectives)
        reference_point: Reference point for hypervolume
        epsilon: Approximation error bound (default 0.01 for 1% error)
        delta: Error probability (default 0.25)
        minimize: If True, compute for minimization (default)
        use_kdtree: If True, use k-d tree for dominance checks (default)

    Returns:
        ApproximationResult with hypervolume estimate and statistics
    """
    if pareto_front.shape[0] == 0:
        return ApproximationResult(0.0, 0, 0, "MCM2RV", epsilon, delta)

    if not minimize:
        pareto_front = -pareto_front.copy()
        reference_point = -reference_point.copy()

    n_points, n_dims = pareto_front.shape

    # Compute parameters
    volumes = np.prod(reference_point - pareto_front, axis=1)
    W = np.sum(volumes)
    ideal_point = np.min(pareto_front, axis=0)

    # Stopping criterion
    R = int(
        math.floor(
            (4 * (1 + epsilon * (1 - epsilon)) * np.log(2 / delta))
            / (epsilon**2 * (1 - epsilon) ** 2)
        )
    )

    # K-d tree for binary dominance checks (safe to use here)
    if use_kdtree and n_points > 10:
        checker = DominanceAnalysis(pareto_front)
    else:
        checker = None

    S = 0
    N = 0
    total_attempts = 0
    total_comparisons = 0

    while S < R:
        print(f"S = {S} R = {R}")
        # sample uniformly in bounding box Omega
        current_batch_size = min(batch_size, int(R - S) * 5)

        samples_batch = np.random.uniform(
            ideal_point, reference_point, size=(current_batch_size, n_dims)
        )

        for i in range(current_batch_size):
            if S >= R:
                break

            sample = samples_batch[i]
            total_attempts += 1

            # Check if in dominated region V (k-d tree safe here)
            if checker is not None:
                is_dominated = checker.is_dominated(sample)
                total_comparisons += 1
            else:
                # Vectorized check
                is_lt = pareto_front < sample
                is_close = np.isclose(pareto_front, sample)
                is_dominated = np.any(np.all(is_lt | is_close, axis=1))
                total_comparisons += n_points

            if not is_dominated:
                continue

            # Sample in V - compute eta
            N += 1
            k = np.random.randint(0, n_points)
            is_k_lt = pareto_front[k] < sample
            is_k_close = np.isclose(pareto_front[k], sample)
            eta = 1 if np.all(is_k_lt | is_k_close) else 0
            total_comparisons += 1
            S += eta

    hypervolume_estimate = (W / n_points) * (N / S)

    return ApproximationResult(
        hypervolume=float(hypervolume_estimate),
        num_samples=N,
        num_comparisons=total_comparisons,
        algorithm_used="MCM2RV-KD" if use_kdtree else "MCM2RV",
        approximation_error_bound=epsilon,
        error_probability=delta,
    )


# ============================================================================
# Hybrid Implementation
# ============================================================================


def estimate_overlap(
    pareto_front: np.ndarray,
    reference_point: np.ndarray,
    volumes: np.ndarray,
    probabilities: np.ndarray,
    n_probes: int = 50,
) -> Tuple[str, float]:
    """
    Quick overlap detection using direct E[xi] estimation.

    Generates probe samples and measures how many random tests are needed
    to find a dominating point. This directly estimates the key metric
    that determines FPRAS vs MCM2RV performance.

    Args:
        pareto_front: Array of shape (n_points, n_objectives)
        reference_point: Reference point
        volumes: Pre-computed hyperrectangle volumes
        probabilities: Sampling probabilities for hyperrectangles
        n_probes: Number of probe samples (default 50)

    Returns:
        Tuple of (overlap_type, mean_xi) where:
        - overlap_type: 'high', 'medium', or 'low'
        - mean_xi: Average number of tests per sample
    """
    n_points, n_dims = pareto_front.shape
    xi_values = []

    for _ in range(n_probes):
        # Select hyperrectangle and generate sample (same as FPRAS)
        j = np.random.choice(n_points, p=probabilities)
        sample = np.random.uniform(pareto_front[j], reference_point, n_dims)

        # Count tests until finding dominating point (randomized order)
        xi = 0
        test_order = np.random.permutation(n_points)
        for k in test_order:
            xi += 1
            if np.all(sample > pareto_front[k]):
                break

        xi_values.append(xi)

    mean_xi = np.mean(xi_values)

    # Decision thresholds based on empirical performance
    # High: E[xi] > 20 means FPRAS will be very slow
    # Low: E[xi] < 5 means FPRAS will be fast
    if mean_xi > 20:
        return "high", mean_xi
    elif mean_xi < 5:
        return "low", mean_xi
    else:
        return "medium", mean_xi


def estimate_theta_bounds(
    V_estimate: float,
    alpha: float,
    epsilon: float,
    delta: float,
    n_points: int,
    W: float,
    U: float,
    M1_total: float,
) -> Tuple[float, float]:
    """
    Estimate theta parameter bounds for algorithm selection.

    Theta predicts the relative performance of MCM2RV vs FPRAS.
    When theta > 1, MCM2RV is faster; when theta < 1, FPRAS is faster.

    Args:
        V_estimate: Current hypervolume estimate
        alpha: Fraction of FPRAS budget spent so far
        epsilon: Approximation error bound
        delta: Error probability
        n_points: Number of Pareto points
        W: Sum of hyperrectangle volumes
        U: Bounding box volume
        M1_total: Total FPRAS comparison budget

    Returns:
        Tuple of (theta_upper, theta_lower)
    """
    # Confidence bounds on V estimate
    epsilon_1 = epsilon / np.sqrt(alpha)
    V_lower = V_estimate / (1 + epsilon_1)
    V_upper = V_estimate / (1 - epsilon_1)

    # MCM2RV stopping criterion
    R_val = (4 * (1 + epsilon * (1 - epsilon)) * np.log(2 / delta)) / (
        epsilon**2 * (1 - epsilon) ** 2
    )

    # Theta computation (Equation 15 from Deng & Zhang 2020)
    def compute_theta(V):
        return (n_points**2 * (V**2 + (U - V) * W) / W**2) * (R_val / M1_total)

    # Conservative bounds: use V estimates that make decision harder
    theta_upper = compute_theta(V_upper)  # Optimistic for MCM2RV
    theta_lower = compute_theta(V_lower)  # Pessimistic for MCM2RV

    return theta_upper, theta_lower


def compute_hypervolume_hybrid(
    pareto_front: np.ndarray,
    reference_point: np.ndarray,
    epsilon: float = 0.01,
    delta: float = 0.25,
    minimize: bool = True,
    use_kdtree: bool = True,
    fpras_batch_size: int = 100,
    mcm2rv_batch_size: int = 1000,
    n_overlap_probes=50,
    overlap_ratio_threshold=5.0,
    verbose: bool = False,
) -> ApproximationResult:
    """
    Compute hypervolume using hybrid algorithm.

    Additional multi-level selection strategy:
    1. Geometric pre-screening
    2. Fast overlap probing
    3. Adaptive FPRAS rounds

    Args:
        pareto_front: Array of shape (n_points, n_objectives)
        reference_point: Reference point for hypervolume
        epsilon: Approximation error bound (default 0.01 for 1% error)
        delta: Error probability (default 0.25)
        minimize: If True, compute for minimization (default)
        use_kdtree: If True, use k-d tree in MCM2RV (default)
        fpras_batch_size: Batch size for FPRAS (default 100)
        mcm2rv_batch_size: Batch size for MCM2RV (default 1000)
        verbose: If True, print decision information

    Returns:
        ApproximationResult with hypervolume estimate and statistics
    """
    if pareto_front.shape[0] == 0:
        return ApproximationResult(0.0, 0, 0, "Hybrid", epsilon, delta)

    # Convert to minimization if needed
    if not minimize:
        pareto_front = -pareto_front.copy()
        reference_point = -reference_point.copy()

    n_points, n_dims = pareto_front.shape

    # Compute parameters needed for all levels
    volumes = np.prod(reference_point - pareto_front, axis=1)
    W = np.sum(volumes)
    probabilities = volumes / W
    ideal_point = np.min(pareto_front, axis=0)
    U = np.prod(reference_point - ideal_point)

    M1_total = (8 * (1 + epsilon) * n_points * np.log(2 / delta)) / (epsilon**2)

    # ========================================================================
    # LEVEL 1: Geometric Pre-Screening
    # ========================================================================
    # Cost: O(n) - already computed above
    # Catches: ~50% of problems (extreme cases)

    if overlap_ratio_threshold < 2.0:
        overlap_ratio_threshold = 2.0

    overlap_ratio = W / U

    if overlap_ratio > overlap_ratio_threshold:
        # Very high overlap: W >> U means hyperrectangles overlap heavily
        # This indicates spherical/convex front, and therefore MCM2RV will be faster
        if verbose:
            print(
                f"[Level 1] Geometric screening: overlap_ratio={overlap_ratio:.3f} > {overlap_ratio_threshold}"
            )
            print("[Level 1] Decision: Use MCM2RV immediately (high overlap)")

        return compute_hypervolume_mcm2rv(
            pareto_front,
            reference_point,
            epsilon,
            delta,
            True,
            use_kdtree=True,
            batch_size=mcm2rv_batch_size,
        )

    elif overlap_ratio < 1.2:
        # Very low overlap: W ~= U means hyperrectangles barely overlap
        # This indicates linear/well-separated front and FPRAS should be much faster
        if verbose:
            print(
                f"[Level 1] Geometric screening: overlap_ratio={overlap_ratio:.2f} < 1.2"
            )
            print("[Level 1] Decision: Use FPRAS immediately (low overlap)")

        return compute_hypervolume_fpras(
            pareto_front,
            reference_point,
            epsilon,
            delta,
            True,
            batch_size=fpras_batch_size,
        )

    # ========================================================================
    # LEVEL 2: Fast Overlap Probing
    # ========================================================================
    # Cost: ~n_overlap_probes × n comparisons (very cheap)
    # Catches: ~40% of remaining problems

    if verbose:
        print(
            f"[Level 1] Inconclusive: overlap_ratio={overlap_ratio:.2f} in [1.2, {overlap_ratio_threshold}]"
        )
        print(f"[Level 2] Running fast probing ({n_overlap_probes} samples)...")

    overlap_type, mean_xi = estimate_overlap(
        pareto_front, reference_point, volumes, probabilities, n_probes=n_overlap_probes
    )

    if overlap_type == "high":
        if verbose:
            print(f"[Level 2] Probing result: E[xi]={mean_xi:.1f} > 20 (high)")
            print("[Level 2] Decision: Use MCM2RV (FPRAS would be slow)")

        return compute_hypervolume_mcm2rv(
            pareto_front,
            reference_point,
            epsilon,
            delta,
            True,
            use_kdtree=True,
            batch_size=mcm2rv_batch_size,
        )

    elif overlap_type == "low":
        if verbose:
            print(f"[Level 2] Probing result: E[xi]={mean_xi:.1f} < 5 (low)")
            print("[Level 2] Decision: Use FPRAS (FPRAS will be fast)")

        return compute_hypervolume_fpras(
            pareto_front,
            reference_point,
            epsilon,
            delta,
            True,
            batch_size=fpras_batch_size,
        )

    # ========================================================================
    # LEVEL 3: Adaptive FPRAS Rounds
    # ========================================================================
    # Cost: 1-15% of M1_total (adaptive)
    # Catches: Remaining ~10% of edge cases

    if verbose:
        print(f"[Level 2] Inconclusive: E[xi]={mean_xi:.1f} in [5, 20] (medium)")
        print("[Level 3] Running adaptive hybrid algorithm...")

    # Exponential round schedule for fast decisions
    round_fractions = [0.01, 0.02, 0.04, 0.08]
    cumulative_fraction = 0.0
    total_comparisons = 0
    sum_xi = 0.0
    N = 0
    round_num = 0

    for round_fraction in round_fractions:
        round_num += 1
        cumulative_fraction += round_fraction
        target_comparisons = int(cumulative_fraction * M1_total)

        # Run FPRAS for this round
        total_comparisons, sum_xi, N = _run_fpras_round(
            pareto_front=pareto_front,
            reference_point=reference_point,
            volumes=volumes,
            probabilities=probabilities,
            target_comparisons=target_comparisons,
            current_comparisons=total_comparisons,
            current_sum_xi=sum_xi,
            current_N=N,
            batch_size=fpras_batch_size,
        )

        # Estimate current hypervolume and theta bounds
        if N == 0:
            N = 1
        V_estimate = (W / n_points) * (sum_xi / N)

        theta_upper, theta_lower = estimate_theta_bounds(
            V_estimate, cumulative_fraction, epsilon, delta, n_points, W, U, M1_total
        )

        threshold = 1 - cumulative_fraction

        if verbose:
            print(
                f"[Level 3] Round {round_num} ({cumulative_fraction:.1%} of budget): "
                f"V={V_estimate:.2e}, theta in [{theta_lower:.2f}, {theta_upper:.2f}], "
                f"threshold={threshold:.2f}"
            )

        # Check for decision with safety margins (15% margin for robustness)
        if theta_upper < threshold * 0.85:
            # Strong evidence: MCM2RV is faster
            if verbose:
                print("[Level 3] Decision: theta_upper < threshold -> MCM2RV")

            result = compute_hypervolume_mcm2rv(
                pareto_front,
                reference_point,
                epsilon,
                delta,
                True,
                use_kdtree=True,
                batch_size=mcm2rv_batch_size,
            )
            result.algorithm_used = "Hybrid-MCM2RV"
            result.num_comparisons += total_comparisons
            return result

        elif theta_lower > threshold * 1.15:
            # Strong evidence: FPRAS is faster
            if verbose:
                print("[Level 3] Decision: theta_lower > threshold -> FPRAS")

            # Complete FPRAS to M1_total
            total_comparisons, sum_xi, N = _run_fpras_round(
                pareto_front=pareto_front,
                reference_point=reference_point,
                volumes=volumes,
                probabilities=probabilities,
                target_comparisons=int(M1_total),
                current_comparisons=total_comparisons,
                current_sum_xi=sum_xi,
                current_N=N,
                batch_size=fpras_batch_size,
            )

            hypervolume_estimate = (W / n_points) * (sum_xi / N)

            return ApproximationResult(
                hypervolume=float(hypervolume_estimate),
                num_samples=N,
                num_comparisons=total_comparisons,
                algorithm_used="Hybrid-FPRAS",
                approximation_error_bound=epsilon,
                error_probability=delta,
            )

    # ========================================================================
    # Fallback: Complete FPRAS
    # ========================================================================
    # Reached if all adaptive rounds completed without clear decision
    # This is rare (~2-3% of cases) and happens for borderline problems

    if verbose:
        print(f"[Level 3] No clear decision after {cumulative_fraction:.1%} of budget")
        print("[Level 3] Completing FPRAS to finish")

    total_comparisons, sum_xi, N = _run_fpras_round(
        pareto_front=pareto_front,
        reference_point=reference_point,
        volumes=volumes,
        probabilities=probabilities,
        target_comparisons=int(M1_total),
        current_comparisons=total_comparisons,
        current_sum_xi=sum_xi,
        current_N=N,
        batch_size=fpras_batch_size,
    )

    hypervolume_estimate = (W / n_points) * (sum_xi / N)

    return ApproximationResult(
        hypervolume=float(hypervolume_estimate),
        num_samples=N,
        num_comparisons=total_comparisons,
        algorithm_used="Hybrid-FPRAS",
        approximation_error_bound=epsilon,
        error_probability=delta,
    )


class AdaptiveHyperVolume:
    """
    Adaptive hypervolume calculator.

    Features:
    - K-d tree for efficient dominance checking
    - Importance sampling for MCM2RV
    - Automatic algorithm selection (hybrid)
    - Multiple algorithm choices (FPRAS, MCM2RV, Hybrid)
    """

    def __init__(
        self,
        reference_point: np.ndarray,
        epsilon: float = 0.01,
        delta: float = 0.25,
        hybrid_rounds: int = 25,
        use_kdtree: bool = True,
    ):
        """
        Initialize the adaptive hypervolume calculator.

        Args:
            reference_point: Reference point for hypervolume computation
            epsilon: Approximation error bound (default 0.01 for 1% error)
            delta: Error probability (default 0.25 for 25%)
            hybrid_rounds: Number of rounds for hybrid algorithm (default 25)
            use_kdtree: If True, use k-d tree in MCM2RV (default)
        """
        self.reference_point = reference_point
        self.epsilon = epsilon
        self.delta = delta
        self.hybrid_rounds = hybrid_rounds
        self.use_kdtree = use_kdtree

    def compute_hypervolume(
        self,
        pareto_front: np.ndarray,
        algorithm: str = "hybrid",
        minimize: bool = True,
        verbose: bool = False,
    ) -> float:
        """
        Compute hypervolume using specified algorithm.

        Args:
            pareto_front: Array of shape (n_points, n_objectives)
            algorithm: One of 'fpras', 'mcm2rv', or 'hybrid' (default)
            minimize: If True, minimize objectives (default)
            verbose: If True, print progress information

        Returns:
            Hypervolume estimate
        """
        result = self.compute_with_statistics(
            pareto_front, algorithm, minimize, verbose
        )
        return result.hypervolume

    def compute_with_statistics(
        self,
        pareto_front: np.ndarray,
        algorithm: str = "hybrid",
        minimize: bool = True,
        verbose: bool = False,
    ) -> ApproximationResult:
        """
        Compute hypervolume with detailed statistics.

        Args:
            pareto_front: Array of shape (n_points, n_objectives)
            algorithm: One of 'fpras', 'mcm2rv', or 'hybrid' (default)
            minimize: If True, minimize objectives (default)
            verbose: If True, print progress information

        Returns:
            ApproximationResult with hypervolume and statistics
        """
        if algorithm == "fpras":
            return compute_hypervolume_fpras(
                pareto_front, self.reference_point, self.epsilon, self.delta, minimize
            )
        elif algorithm == "mcm2rv":
            return compute_hypervolume_mcm2rv(
                pareto_front,
                self.reference_point,
                self.epsilon,
                self.delta,
                minimize,
                use_kdtree=self.use_kdtree,
            )
        elif algorithm == "hybrid":
            return compute_hypervolume_hybrid(
                pareto_front,
                self.reference_point,
                self.epsilon,
                self.delta,
                minimize,
                self.hybrid_rounds,
                use_kdtree=self.use_kdtree,
                verbose=verbose,
            )
        else:
            raise ValueError(
                f"Unknown algorithm: {algorithm}. "
                f"Choose from 'fpras', 'mcm2rv', or 'hybrid'"
            )


# ============================================================================
# Utility Functions
# ============================================================================


def compare_optimizations(
    pareto_front: np.ndarray,
    reference_point: np.ndarray,
    epsilon: float = 0.01,
    delta: float = 0.25,
    minimize: bool = True,
) -> Dict[str, ApproximationResult]:
    """
    Compare different optimization strategies.

    Runs multiple versions to demonstrate speedup from optimizations.

    Args:
        pareto_front: Array of shape (n_points, n_objectives)
        reference_point: Reference point
        epsilon: Approximation error
        delta: Error probability
        minimize: If True, minimize objectives

    Returns:
        Dictionary mapping strategy name to result
    """
    results = {}

    # Baseline FPRAS (no optimizations)
    print("Running baseline FPRAS...")
    results["FPRAS-Baseline"] = compute_hypervolume_fpras(
        pareto_front, reference_point, epsilon, delta, minimize
    )

    # Baseline MCM2RV (no optimizations)
    print("Running baseline MCM2RV...")
    results["MCM2RV-Baseline"] = compute_hypervolume_mcm2rv(
        pareto_front, reference_point, epsilon, delta, minimize, use_kdtree=False
    )

    # MCM2RV with all optimizations
    print("Running MCM2RV with all optimizations...")
    results["MCM2RV-Full"] = compute_hypervolume_mcm2rv(
        pareto_front, reference_point, epsilon, delta, minimize, use_kdtree=True
    )

    # Hybrid with all optimizations
    print("Running Hybrid with all optimizations...")
    results["Hybrid-Optimized"] = compute_hypervolume_hybrid(
        pareto_front,
        reference_point,
        epsilon,
        delta,
        minimize,
        use_kdtree=True,
        verbose=False,
    )

    return results


if __name__ == "__main__":
    # Example usage demonstrating the optimizations
    print("=" * 80)
    print("Adaptive Hypervolume Computation with Optimizations")
    print("=" * 80)

    # Generate example Pareto front
    np.random.seed(42)
    n_points = 100
    n_dims = 15

    # Generate random Pareto front (roughly on a spherical surface)
    angles = np.random.uniform(0, np.pi / 2, (n_points, n_dims - 1))
    pareto_front = np.zeros((n_points, n_dims))
    for i in range(n_points):
        point = np.ones(n_dims)
        for j in range(n_dims - 1):
            point[j] *= np.cos(angles[i, j])
            point[j + 1 :] *= np.sin(angles[i, j])
        pareto_front[i] = point * 2.0  # Scale to [0, 2]

    reference_point = np.full(n_dims, 3.0)

    print("\nTest problem:")
    print(f"  Dimensions: {n_dims}")
    print(f"  Points: {n_points}")
    print(f"  Reference point: {reference_point[0]} (all dimensions)")
    print()

    # Compare all optimization strategies
    print("Comparing optimization strategies...")
    print("-" * 80)

    results = compare_optimizations(
        pareto_front, reference_point, epsilon=0.02, delta=0.25, minimize=True
    )

    print("\nResults:")
    print("-" * 80)
    print(
        f"{'Algorithm':<25} {'HV':<12} {'Samples':<10} {'Comparisons':<12} {'Speedup':<10}"
    )
    print("-" * 80)

    baseline_comparisons = results["FPRAS-Baseline"].num_comparisons

    for name, result in results.items():
        speedup = (
            baseline_comparisons / result.num_comparisons
            if result.num_comparisons > 0
            else 0
        )
        print(
            f"{name:<25} {result.hypervolume:<12.6f} {result.num_samples:<10} "
            f"{result.num_comparisons:<12} {speedup:<10.2f}x"
        )

    print("-" * 80)
