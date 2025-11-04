"""
Implementation of the Box Decomposition Algorithm from Lacour et al. (2017):
"A Box Decomposition Algorithm to Compute the Hypervolume Indicator"

This algorithm partitions the dominated region into disjoint hyperrectangles
and computes hypervolume as the sum of their volumes.

Time complexity: O(n^⌊(p-1)/2⌋+1) for nonincremental version
Space complexity: O(n^⌊p/2⌋)


References:

Renaud Lacour, Kathrin Klamroth, Carlos M. Fonseca, A Box
Decomposition Algorithm to Compute the Hypervolume Indicator,
Computers & Operations Research, 2016.

"""

import numpy as np
from typing import List
from dataclasses import dataclass


@dataclass
class LocalUpperBound:
    """
    A local upper bound with its defining points.

    u: The coordinates of the upper bound
    defining_points: For each dimension j, the index of the point that defines u_j
    """

    coords: np.ndarray
    defining_points: np.ndarray  # Shape (d,), indices into point set

    def __hash__(self):
        return hash(tuple(self.coords))

    def __eq__(self, other):
        return np.allclose(self.coords, other.coords)


class HyperVolumeBoxDecomposition:
    """
    Hypervolume computation via box decomposition.

    Based on Lacour, Klamroth, and Fonseca (2017).
    """

    def __init__(self, ref_point: np.ndarray):
        """
        Initialize with reference point.

        Args:
            ref_point: Reference point dominating all points in Pareto front
        """
        self.ref_point = np.asarray(ref_point, dtype=np.float64)
        self.d = len(ref_point)

        # For tracking defining points
        self.points = None
        self.n_points = 0

        # Dummy points ẑ^j = (z^r_j, 0_{-j})
        self.dummy_indices = None

    def compute_hypervolume(self, points: np.ndarray) -> float:
        """
        Compute hypervolume of non-dominated point set.

        Args:
            points: Array of shape (n, d)

        Returns:
            Hypervolume value
        """
        if len(points) == 0:
            return 0.0

        points = np.asarray(points, dtype=np.float64)
        n, d = points.shape

        if d != self.d:
            raise ValueError(f"Points dimension {d} doesn't match ref point {self.d}")

        # Filter dominated points
        points = self._filter_dominated(points)
        n = len(points)

        if n == 0:
            return 0.0

        # Sort by last dimension (required for nonincremental algorithm)
        sort_idx = np.argsort(points[:, -1])
        self.points = points[sort_idx]
        self.n_points = n

        # Add dummy points
        self._initialize_dummy_points()

        # Compute upper bound set using nonincremental algorithm
        upper_bounds = self._compute_upper_bounds_nonincremental()

        # Compute hypervolume as sum of box volumes
        total_volume = 0.0
        for ub in upper_bounds:
            volume = self._compute_box_volume(ub)
            total_volume += volume

        return total_volume

    def _initialize_dummy_points(self):
        """Create dummy points ẑ^j = (z^r_j, 0_{-j}) for j=1,...,d."""
        # Dummy points have special negative indices
        self.dummy_indices = -(np.arange(self.d) + 1)

    def _get_point_coords(self, idx: int) -> np.ndarray:
        """Get coordinates for a point index (handles dummy points)."""
        if idx < 0:
            # Dummy point ẑ^j where j = -idx - 1
            j = -idx - 1
            coords = np.zeros(self.d)
            coords[j] = self.ref_point[j]
            return coords
        else:
            return self.points[idx]

    def _filter_dominated(self, points: np.ndarray) -> np.ndarray:
        """Remove dominated points using vectorized comparison."""
        n = len(points)
        if n <= 1:
            return points

        dominated = np.zeros(n, dtype=bool)

        for i in range(n):
            if dominated[i]:
                continue
            # Point i dominates j if all coordinates of i <= j
            dominates = np.all(points > points[i], axis=1)
            dominates[i] = False
            dominated |= dominates

        return points[~dominated]

    def _compute_upper_bounds_nonincremental(self) -> List[LocalUpperBound]:
        """
        Compute upper bound set using nonincremental Algorithm 2.

        Assumes points are sorted by last dimension.

        Returns:
            List of LocalUpperBound objects
        """
        # Initialize with reference point
        ubs = [
            LocalUpperBound(
                coords=self.ref_point.copy(), defining_points=self.dummy_indices.copy()
            )
        ]

        # Process each point
        for point_idx in range(self.n_points):
            z_bar = self.points[point_idx]

            # Find upper bounds strictly dominated by z_bar (set A)
            A = []
            A_bar = []  # Not dominated (Ā)

            for ub in ubs:
                if np.all(z_bar < ub.coords):
                    A.append(ub)
                else:
                    A_bar.append(ub)

            if not A:
                # No upper bounds dominated, continue
                ubs = A_bar
                continue

            # Generate new upper bounds
            new_ubs = []

            # Step 2: For each u in A, create (z̄_p, u_{-p})
            for ub in A:
                new_coords = ub.coords.copy()
                new_coords[-1] = z_bar[-1]

                new_def_pts = ub.defining_points.copy()
                new_def_pts[-1] = point_idx

                new_ubs.append(
                    LocalUpperBound(coords=new_coords, defining_points=new_def_pts)
                )

            # Step 3: For each u in A, for j=1,...,p-1, create (z̄_j, u_{-j})
            # if z̄_j ≥ max_{k≠j}{z^k_j(u)}
            for ub in A:
                for j in range(self.d - 1):
                    # Compute max_{k≠j}{z^k_j(u)}
                    max_val = -np.inf
                    for k in range(self.d):
                        if k != j:
                            def_point_idx = ub.defining_points[k]
                            def_point = self._get_point_coords(def_point_idx)
                            max_val = max(max_val, def_point[j])

                    # Check condition z̄_j ≥ max_{k≠j}{z^k_j(u)}
                    if max_val < z_bar[j]:
                        new_coords = ub.coords.copy()
                        new_coords[j] = z_bar[j]

                        new_def_pts = ub.defining_points.copy()
                        new_def_pts[j] = point_idx

                        new_ubs.append(
                            LocalUpperBound(
                                coords=new_coords, defining_points=new_def_pts
                            )
                        )

            # Combine: new upper bounds + unddominated old ones (Ā)
            ubs = new_ubs + A_bar

            # Remove duplicates
            ubs = self._remove_duplicate_upper_bounds(ubs)

        return ubs

    def _remove_duplicate_upper_bounds(
        self, ubs: List[LocalUpperBound]
    ) -> List[LocalUpperBound]:
        """Remove duplicate upper bounds based on coordinates."""
        if not ubs:
            return ubs

        # Use dict to track unique upper bounds
        unique = {}
        for ub in ubs:
            key = tuple(ub.coords)
            if key not in unique:
                unique[key] = ub

        return list(unique.values())

    def _compute_box_volume(self, ub: LocalUpperBound) -> float:
        """
        Compute volume of box B(u) defined by upper bound u.

        From equation (2):
        B(u) = [z^1_1(u), z^r_1] × ∏^p_{j=2}[max_{k<j}{z^k_j(u)}, u_j]

        Args:
            ub: Local upper bound

        Returns:
            Volume of the box
        """
        volume = 1.0

        # First dimension: [z^1_1(u), z^r_1]
        def_point_0 = self._get_point_coords(ub.defining_points[0])
        dim_0_length = self.ref_point[0] - def_point_0[0]

        if dim_0_length <= 0:
            return 0.0

        volume *= dim_0_length

        # Remaining dimensions: [max_{k<j}{z^k_j(u)}, u_j]
        for j in range(1, self.d):
            # Compute max_{k<j}{z^k_j(u)}
            max_val = -np.inf
            for k in range(j):
                def_point_k = self._get_point_coords(ub.defining_points[k])
                max_val = max(max_val, def_point_k[j])

            dim_j_length = ub.coords[j] - max_val

            if dim_j_length <= 0:
                return 0.0

            volume *= dim_j_length

        return volume


# ============================================================================
# Functional Interface
# ============================================================================


def compute_hypervolume_box_decomposition(
    points: np.ndarray, ref_point: np.ndarray
) -> float:
    """
    Compute hypervolume using box decomposition algorithm.

    Args:
        points: Point set of shape (n, d)
        ref_point: Reference point of shape (d,)

    Returns:
        Hypervolume value

    Example:
        >>> points = np.array([[1.0, 1.0], [2.0, 0.5]])
        >>> ref = np.array([3.0, 3.0])
        >>> hv = compute_hypervolume(points, ref)
    """
    calc = HyperVolumeBoxDecomposition(ref_point)
    return calc.compute_hypervolume(points)


def compute_hypervolume_box_decomposition_batch(
    point_sets: List[np.ndarray], ref_points: np.ndarray
) -> np.ndarray:
    """
    Compute hypervolumes for multiple point sets in batch.

    Args:
        point_sets: List of point arrays
        ref_points: Reference points of shape (k, d) or (d,) for all

    Returns:
        Array of hypervolume values
    """
    if len(point_sets) == 0:
        return np.array([])

    # Broadcast reference point if needed
    if ref_points.ndim == 1:
        ref_points = np.tile(ref_points, (len(point_sets), 1))

    results = []
    for points, ref in zip(point_sets, ref_points):
        calc = HyperVolumeBoxDecomposition(ref)
        hv = calc.compute_hypervolume(points)
        results.append(hv)

    return np.array(results)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import time

    # Example 1: Basic 2D case - staircase pattern
    print("Example 1: 2D hypervolume (staircase)")
    points_2d = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
    ref_2d = np.array([4.0, 4.0])

    hv = compute_hypervolume_box_decomposition(points_2d, ref_2d)
    print(f"Hypervolume: {hv:.6f}")
    print(f"Expected: {6.0:.6f}")
    print()

    # Example 2: Simple 2D
    print("Example 2: 2D hypervolume_box_decomposition (two points)")
    points_2d_simple = np.array([[1.0, 2.0], [2.0, 1.0]])
    ref_2d_simple = np.array([3.0, 3.0])

    hv = compute_hypervolume_box_decomposition(points_2d_simple, ref_2d_simple)
    print(f"Hypervolume: {hv:.6f}")
    print(f"Expected: {3.0:.6f}")
    print()

    # Example 3: 3D case
    print("Example 3: 3D hypervolume")
    points_3d = np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 2.0], [2.0, 1.0, 2.0]])
    ref_3d = np.array([3.0, 3.0, 3.0])

    hv = compute_hypervolume_box_decomposition(points_3d, ref_3d)
    print(f"Hypervolume: {hv:.6f}")
    print()

    # Example 4: Higher dimensional
    print("Example 4: 5D hypervolume")
    np.random.seed(42)
    points_5d = np.random.rand(20, 5) * 5
    ref_5d = np.array([6.0] * 5)

    calc = HyperVolumeBoxDecomposition(ref_5d)
    hv = calc.compute_hypervolume(points_5d)
    print(f"Hypervolume: {hv:.6f}")
    print()

    # Example 5: Batch processing
    print("Example 5: Batch processing")
    fronts = [np.random.rand(10, 3) * 5 for _ in range(5)]
    refs = np.array([[6.0, 6.0, 6.0]] * 5)

    hvs = compute_hypervolume_box_decomposition_batch(fronts, refs)
    print(f"Batch hypervolumes: {hvs}")
    print()

    # Performance test
    dimensions_to_test = [3, 5, 8]
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

        calc = HyperVolumeBoxDecomposition(ref_point)
        # Test AdaptiveHyperVolume with automatic selection
        start = time.time()
        hv = calc.compute_hypervolume(pareto_front)
        time_opt = time.time() - start

        print(f"  HV={hv:.4f}, computation time: {time_opt:.6f}s ")
