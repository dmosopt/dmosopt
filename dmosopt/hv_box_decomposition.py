"""
Hypervolume computation via box decomposition (Lacour et al. 2017) with fast
paths for 2D/3D and an optional moocore backend.

When the ``moocore`` package is installed, exact hypervolume computation is
delegated to its compiled C library (``fpli_hv``), which uses O(n log n)
algorithms for 2D/3D and optimised recursive algorithms for higher dimensions.

When ``moocore`` is not available the following Python fallbacks are used:
- 2D: O(n log n) sort + vectorised sweep (no Python loops in the hot path)
- 3D: O(n^2) plane-sweep using the 2D helper per z-slice
- d >= 4: Lacour et al. (2017) nonincremental box decomposition with the
  dominated-set check vectorised via NumPy broadcasting.

The stochastic approximation methods for d >= 10 (hv_adaptive.py) are
completely unchanged by this module.

References:

Renaud Lacour, Kathrin Klamroth, Carlos M. Fonseca, A Box
Decomposition Algorithm to Compute the Hypervolume Indicator,
Computers & Operations Research, 2016.

"""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
from scipy.stats import norm

try:
    import moocore as _moocore

    _MOOCORE_AVAILABLE = True
except ImportError:
    _MOOCORE_AVAILABLE = False


# ============================================================================
# Pure-Python / NumPy fallback helpers (used when moocore is absent)
# ============================================================================


def _compute_hypervolume_2d(points: np.ndarray, ref_point: np.ndarray) -> float:
    """O(n log n) 2D hypervolume via sort + vectorised swept-area formula.

    Points strictly dominated by ref_point are the only valid contributors.
    Dominated-by-each-other points are pruned during the sort pass.
    """
    mask = np.all(points < ref_point, axis=1)
    pts = points[mask]
    if len(pts) == 0:
        return 0.0

    # Sort by first objective ascending; for a non-dominated front the second
    # objective will then be strictly descending.
    idx = np.argsort(pts[:, 0])
    pts = pts[idx]

    # Keep only non-dominated points: scan right-to-left, keep only those
    # with a strictly smaller second coordinate than any seen so far.
    keep = np.ones(len(pts), dtype=bool)
    min_y = np.inf
    for i in range(len(pts) - 1, -1, -1):
        if pts[i, 1] < min_y:
            min_y = pts[i, 1]
        else:
            keep[i] = False
    pts = pts[keep]
    if len(pts) == 0:
        return 0.0

    # Swept area: width_i * height_i where width_i = x_{i+1} - x_i
    # (with x_{n+1} = ref_x) and height_i = ref_y - y_i.
    x_next = np.empty(len(pts))
    x_next[:-1] = pts[1:, 0]
    x_next[-1] = ref_point[0]
    return float(np.dot(x_next - pts[:, 0], ref_point[1] - pts[:, 1]))


def _compute_hypervolume_3d(points: np.ndarray, ref_point: np.ndarray) -> float:
    """O(n^2) 3D hypervolume via z-plane sweep using the 2D helper.

    Sort non-dominated points by z3 ascending. Between consecutive z3 levels
    the 2D HV of the projection onto (z1, z2) is constant, so:

        HV3D = sum_{i=0}^{n-1} (z3_{i+1} - z3_i) * HV2D(pts[0..i], ref[:2])
             + (ref_z - z3_{n-1}) * HV2D(all_pts, ref[:2])

    Each inner 2D HV call is O(n log n) NumPy; no Python loops inside them.
    """
    mask = np.all(points < ref_point, axis=1)
    pts = points[mask]
    if len(pts) == 0:
        return 0.0

    # Sort by third objective ascending.
    order = np.argsort(pts[:, 2])
    pts = pts[order]
    ref2 = ref_point[:2]
    ref_z = ref_point[2]

    total = 0.0
    n = len(pts)
    for i in range(n):
        z_lo = pts[i, 2]
        z_hi = pts[i + 1, 2] if i + 1 < n else ref_z
        dz = z_hi - z_lo
        if dz > 0:
            total += dz * _compute_hypervolume_2d(pts[: i + 1, :2], ref2)

    return total


# ============================================================================
# Box decomposition data structures (used for d >= 4 fallback and for EHVI)
# ============================================================================


@dataclass
class Box:
    lower: np.ndarray
    upper: np.ndarray
    _volume: float = None

    @property
    def volume(self) -> float:
        if self._volume is None:
            mask = ~np.isinf(self.lower) & ~np.isinf(self.upper)
            if not np.any(mask):
                self._volume = 0.0
            else:
                self._volume = np.prod(self.upper[mask] - self.lower[mask])
        return self._volume


@dataclass
class LocalUpperBound:
    """A local upper bound with its defining points.

    coords: coordinates of the upper bound
    defining_points: for each dimension j, the index of the point defining u_j
    """

    coords: np.ndarray
    defining_points: np.ndarray  # shape (d,), indices into point set

    def __hash__(self):
        return hash(tuple(self.coords))

    def __eq__(self, other):
        return np.allclose(self.coords, other.coords)


class HyperVolumeBoxDecomposition:
    """Hypervolume computation via box decomposition (Lacour, Klamroth, Fonseca 2017).

    When moocore is installed the ``compute_hypervolume`` method delegates to
    its compiled C backend.  Otherwise, dimension-specific Python/NumPy
    fallbacks are used for d=2 and d=3, and the full box decomposition
    algorithm (with a vectorised dominance check) is used for d >= 4.

    The EHVI / candidate-selection interface (``select_candidates``,
    ``_decompose_dominated_space``, ``_compute_batch_ehvi``) always uses the
    Python box decomposition internals regardless of moocore availability.
    """

    def __init__(self, ref_point: np.ndarray):
        self.ref_point = np.asarray(ref_point, dtype=np.float64)
        self.d = len(ref_point)

        self.points = None
        self.n_points = 0
        self.dummy_indices = None

    def compute_hypervolume(self, points: np.ndarray) -> float:
        """Compute hypervolume of a (possibly dominated) point set."""
        points = np.asarray(points, dtype=np.float64)
        if len(points) == 0:
            return 0.0

        d = points.shape[1] if points.ndim == 2 else self.d

        if d != self.d:
            raise ValueError(f"Points dimension {d} doesn't match ref point {self.d}")

        # Fast path: compiled C backend via moocore
        if _MOOCORE_AVAILABLE:
            return float(_moocore.hypervolume(points, ref=self.ref_point))

        # Python fallbacks
        if self.d == 2:
            return _compute_hypervolume_2d(points, self.ref_point)
        if self.d == 3:
            return _compute_hypervolume_3d(points, self.ref_point)

        # General d >= 4: Lacour et al. box decomposition.
        points = self._filter_dominated(points)
        n = len(points)
        if n == 0:
            return 0.0

        sort_idx = np.argsort(points[:, -1])
        self.points = points[sort_idx]
        self.n_points = n
        self._initialize_dummy_points()

        upper_bounds = self._compute_upper_bounds_nonincremental()

        total_volume = 0.0
        for ub in upper_bounds:
            total_volume += self._compute_box_volume(ub)
        return total_volume

    def _initialize_dummy_points(self):
        """Create dummy points z^j = (z^r_j, 0_{-j}) for j=1,...,d."""
        self.dummy_indices = -(np.arange(self.d) + 1)

    def _get_point_coords(self, idx: int) -> np.ndarray:
        if idx < 0:
            j = -idx - 1
            coords = np.zeros(self.d)
            coords[j] = self.ref_point[j]
            return coords
        return self.points[idx]

    def _filter_dominated(self, points: np.ndarray) -> np.ndarray:
        """Remove dominated points using vectorised comparison."""
        n = len(points)
        if n <= 1:
            return points

        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if dominated[i]:
                continue
            dominates = np.all(points > points[i], axis=1)
            dominates[i] = False
            dominated |= dominates

        return points[~dominated]

    def _compute_upper_bounds_nonincremental(self) -> List[LocalUpperBound]:
        """Nonincremental Algorithm 2 from Lacour et al. (2017).

        The dominated-set check (building set A) is vectorised: all upper-bound
        coordinates are stacked into a 2-D array so that a single np.all call
        replaces the inner Python loop.
        """
        ubs = [
            LocalUpperBound(
                coords=self.ref_point.copy(),
                defining_points=self.dummy_indices.copy(),
            )
        ]

        for point_idx in range(self.n_points):
            z_bar = self.points[point_idx]

            if not ubs:
                continue

            # Vectorised dominance check: find which UBs are strictly
            # dominated by z_bar in all coordinates.
            ub_coords = np.array([ub.coords for ub in ubs])  # (|UBS|, d)
            dominated_mask = np.all(z_bar[None, :] < ub_coords, axis=1)

            A = [ubs[i] for i in np.where(dominated_mask)[0]]
            A_bar = [ubs[i] for i in np.where(~dominated_mask)[0]]

            if not A:
                ubs = A_bar
                continue

            new_ubs = []

            # Step 2: for each u in A create (z_bar_p, u_{-p}).
            for ub in A:
                new_coords = ub.coords.copy()
                new_coords[-1] = z_bar[-1]
                new_def_pts = ub.defining_points.copy()
                new_def_pts[-1] = point_idx
                new_ubs.append(
                    LocalUpperBound(coords=new_coords, defining_points=new_def_pts)
                )

            # Step 3: for each u in A, for j=1,...,p-1 create (z_bar_j, u_{-j})
            # when z_bar_j >= max_{k != j} z^k_j(u).
            for ub in A:
                for j in range(self.d - 1):
                    max_val = -np.inf
                    for k in range(self.d):
                        if k != j:
                            max_val = max(
                                max_val,
                                self._get_point_coords(ub.defining_points[k])[j],
                            )
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

            ubs = new_ubs + A_bar
            ubs = self._remove_duplicate_upper_bounds(ubs)

        return ubs

    def _remove_duplicate_upper_bounds(
        self, ubs: List[LocalUpperBound]
    ) -> List[LocalUpperBound]:
        if not ubs:
            return ubs
        unique = {}
        for ub in ubs:
            key = tuple(ub.coords)
            if key not in unique:
                unique[key] = ub
        return list(unique.values())

    def _compute_box_volume(self, ub: LocalUpperBound) -> float:
        """Volume of box B(u) via equation (2) of Lacour et al."""
        def_point_0 = self._get_point_coords(ub.defining_points[0])
        dim_0_length = self.ref_point[0] - def_point_0[0]
        if dim_0_length <= 0:
            return 0.0

        volume = dim_0_length
        for j in range(1, self.d):
            max_val = -np.inf
            for k in range(j):
                max_val = max(
                    max_val,
                    self._get_point_coords(ub.defining_points[k])[j],
                )
            dim_j_length = ub.coords[j] - max_val
            if dim_j_length <= 0:
                return 0.0
            volume *= dim_j_length

        return volume

    # =========================================================================
    # EHVI / candidate selection interface (always uses Python implementation)
    # =========================================================================

    def select_candidates(
        self,
        pareto_front: np.ndarray,
        candidate_means: np.ndarray,
        candidate_variances: np.ndarray,
        n_select: int = 1,
        batch_size: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Select best candidate points based on Expected Hypervolume Improvement."""
        n_candidates = len(candidate_means)

        if len(pareto_front) == 0:
            boxes = []
        else:
            boxes = self._decompose_dominated_space(pareto_front)

        ehvi_values = np.zeros(n_candidates)

        for batch_start in range(0, n_candidates, batch_size):
            batch_end = min(batch_start + batch_size, n_candidates)
            batch_means = candidate_means[batch_start:batch_end]
            batch_variances = candidate_variances[batch_start:batch_end]

            if len(boxes) == 0:
                for i, (means, variances) in enumerate(
                    zip(batch_means, batch_variances)
                ):
                    ehvi_values[batch_start + i] = self._compute_empty_ehvi(
                        means, variances
                    )
            else:
                batch_ehvi = self._compute_batch_ehvi(
                    boxes, batch_means, batch_variances
                )
                ehvi_values[batch_start:batch_end] = batch_ehvi

        selected_indices = np.copy(np.argsort(-ehvi_values)[:n_select])
        return selected_indices, ehvi_values[selected_indices]

    def _compute_batch_ehvi(
        self, boxes: List[Box], batch_means: np.ndarray, batch_variances: np.ndarray
    ) -> np.ndarray:
        batch_size = len(batch_means)
        ehvi_values = np.zeros(batch_size)

        n_boxes = len(boxes)
        lowers = np.array([box.lower for box in boxes])
        uppers = np.array([box.upper for box in boxes])

        for i in range(batch_size):
            means = batch_means[i]
            variances = batch_variances[i]
            std = np.sqrt(variances)

            means = means[None, :]
            std = std[None, :]

            lower_probs = np.zeros_like(lowers, dtype=float)
            upper_probs = np.ones_like(uppers, dtype=float)

            finite_mask_lower = ~np.isinf(lowers)
            finite_mask_upper = ~np.isinf(uppers)

            if np.any(finite_mask_lower):
                lower_probs[finite_mask_lower] = norm.cdf(
                    (
                        lowers[finite_mask_lower]
                        - means.repeat(n_boxes, 0)[finite_mask_lower]
                    )
                    / std.repeat(n_boxes, 0)[finite_mask_lower]
                )

            if np.any(finite_mask_upper):
                upper_probs[finite_mask_upper] = norm.cdf(
                    (
                        uppers[finite_mask_upper]
                        - means.repeat(n_boxes, 0)[finite_mask_upper]
                    )
                    / std.repeat(n_boxes, 0)[finite_mask_upper]
                )

            partial_exp = std * (
                norm.pdf((lowers - means) / std) - norm.pdf((uppers - means) / std)
            ) + means * (upper_probs - lower_probs)

            contributions = np.prod(partial_exp, axis=1)
            ehvi_values[i] = np.sum(contributions)

        return ehvi_values

    def _decompose_dominated_space(self, pareto_front: np.ndarray) -> List[Box]:
        n_points = len(pareto_front)
        sorted_indices = np.argsort(pareto_front[:, 0])
        sorted_front = pareto_front[sorted_indices]

        lower_bounds = np.full((n_points + 1, self.d), -np.inf)
        upper_bounds = np.full((n_points + 1, self.d), np.inf)

        lower_bounds[1:] = sorted_front
        upper_bounds[:-1] = sorted_front
        upper_bounds[-1] = self.ref_point

        valid_boxes = np.all(upper_bounds > lower_bounds, axis=1)
        boxes = [
            Box(lower_bounds[i], upper_bounds[i])
            for i in range(n_points + 1)
            if valid_boxes[i]
        ]

        return boxes


# ============================================================================
# Functional interface
# ============================================================================


def compute_hypervolume_box_decomposition(
    points: np.ndarray,
    ref_point: np.ndarray,
    algorithm: str | None = None,
) -> float:
    """Compute hypervolume using the fastest available backend.

    Uses moocore's compiled C library when installed (the ``algorithm``
    parameter is then accepted but ignored — moocore dispatches internally).
    Without moocore, ``algorithm`` selects the Python fallback:

    - ``None`` / ``"auto"``: choose based on dimension (2D -> ``"2d"``,
      3D -> ``"3d"``, d >= 4->``"box"``)
    - ``"2d"``: vectorised O(n log n) sweep (d must be 2)
    - ``"3d"``: vectorised O(n^2) plane-sweep (d must be 3)
    - ``"box"`` / ``"fonseca"``: Lacour et al. (2017) box decomposition

    Args:
        points: Point set of shape (n, d)
        ref_point: Reference point of shape (d,)
        algorithm: Backend selector (see above).

    Returns:
        Hypervolume value
    """
    points = np.asarray(points, dtype=np.float64)
    ref_point = np.asarray(ref_point, dtype=np.float64)

    if len(points) == 0:
        return 0.0

    if _MOOCORE_AVAILABLE:
        return float(_moocore.hypervolume(points, ref=ref_point))

    # --- Python fallback with explicit algorithm routing ---
    d = points.shape[1] if points.ndim == 2 else len(ref_point)

    if algorithm is None or algorithm == "auto":
        if d == 2:
            algorithm = "2d"
        elif d == 3:
            algorithm = "3d"
        else:
            algorithm = "box"

    if algorithm == "2d":
        return _compute_hypervolume_2d(points, ref_point)
    if algorithm == "3d":
        return _compute_hypervolume_3d(points, ref_point)

    # "box" or "fonseca" (or any unrecognised string -> box decomposition)
    calc = HyperVolumeBoxDecomposition(ref_point)
    return calc.compute_hypervolume(points)


def compute_hypervolume_box_decomposition_batch(
    point_sets: List[np.ndarray], ref_points: np.ndarray
) -> np.ndarray:
    """Compute hypervolumes for multiple point sets in batch."""
    if len(point_sets) == 0:
        return np.array([])

    if ref_points.ndim == 1:
        ref_points = np.tile(ref_points, (len(point_sets), 1))

    results = []
    for points, ref in zip(point_sets, ref_points):
        results.append(compute_hypervolume_box_decomposition(points, ref))

    return np.array(results)


# ============================================================================
# Example usage
# ============================================================================

if __name__ == "__main__":
    import time

    print(f"moocore available: {_MOOCORE_AVAILABLE}")

    print("\nExample 1: 2D hypervolume (staircase)")
    points_2d = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
    ref_2d = np.array([4.0, 4.0])
    hv = compute_hypervolume_box_decomposition(points_2d, ref_2d)
    print(f"Hypervolume: {hv:.6f}  (expected 6.0)")

    print("\nExample 2: 2D two points")
    points_2d_simple = np.array([[1.0, 2.0], [2.0, 1.0]])
    ref_2d_simple = np.array([3.0, 3.0])
    hv = compute_hypervolume_box_decomposition(points_2d_simple, ref_2d_simple)
    print(f"Hypervolume: {hv:.6f}  (expected 3.0)")

    print("\nExample 3: 3D hypervolume")
    points_3d = np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 2.0], [2.0, 1.0, 2.0]])
    ref_3d = np.array([3.0, 3.0, 3.0])
    hv = compute_hypervolume_box_decomposition(points_3d, ref_3d)
    print(f"Hypervolume: {hv:.6f}")

    print("\nExample 4: 5D hypervolume")
    np.random.seed(42)
    points_5d = np.random.rand(20, 5) * 5
    ref_5d = np.array([6.0] * 5)
    calc = HyperVolumeBoxDecomposition(ref_5d)
    hv = calc.compute_hypervolume(points_5d)
    print(f"Hypervolume: {hv:.6f}")

    print("\nPerformance test")
    for d in [3, 5, 8]:
        np.random.seed(42)
        ref_point = np.full(d, 10.0)
        points = np.random.uniform(1, 9, (250, d))
        calc = HyperVolumeBoxDecomposition(ref_point)
        start = time.time()
        hv = calc.compute_hypervolume(points)
        elapsed = time.time() - start
        print(f"  d={d}, n=250: HV={hv:.4f}, time={elapsed:.6f}s")
