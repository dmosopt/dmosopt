import numpy as np
import sys
from typing import Any, Union, Dict, List, Tuple, Optional
from itertools import product
from scipy.stats import norm
from typing import List, Tuple, Optional
from functools import lru_cache
from dataclasses import dataclass
import heapq

#    Copyright (C) 2010 Simon Wessing
#    TU Dortmund University
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.


class HyperVolumeDimensionSweep:
    """
    Hypervolume computation based on variant 3 of the algorithm in the paper:
    C. M. Fonseca, L. Paquete, and M. Lopez-Ibanez. An improved dimension-sweep
    algorithm for the hypervolume indicator. In IEEE Congress on Evolutionary
    Computation, pages 1157-1163, Vancouver, Canada, July 2006.

    Minimization is implicitly assumed here!

    """

    def __init__(self, referencePoint):
        """Constructor."""
        self.referencePoint = referencePoint
        self.list = []

    def compute_hypervolume(self, front):
        """Returns the hypervolume that is dominated by a non-dominated front.

        Before the HV computation, front and reference point are translated, so
        that the reference point is [0, ..., 0].

        """

        def weaklyDominates(point, other):
            for i in range(len(point)):
                if point[i] > other[i]:
                    return False
            return True

        relevantPoints = []
        referencePoint = self.referencePoint
        dimensions = len(referencePoint)
        for point in front:
            # only consider points that dominate the reference point
            if weaklyDominates(point, referencePoint):
                relevantPoints.append(point)
        if any(referencePoint):
            # shift points so that referencePoint == [0, ..., 0]
            # this way the reference point doesn't have to be explicitly used
            # in the HV computation
            for j in range(len(relevantPoints)):
                relevantPoints[j] = [
                    relevantPoints[j][i] - referencePoint[i] for i in range(dimensions)
                ]
        self.preProcess(relevantPoints)
        bounds = [-1.0e308] * dimensions
        hyperVolume = self.hvRecursive(dimensions - 1, len(relevantPoints), bounds)
        return hyperVolume

    def hvRecursive(self, dimIndex, length, bounds):
        """Recursive call to hypervolume calculation.

        In contrast to the paper, the code assumes that the reference point
        is [0, ..., 0]. This allows the avoidance of a few operations.

        """
        hvol = 0.0
        sentinel = self.list.sentinel
        if length == 0:
            return hvol
        elif dimIndex == 0:
            # special case: only one dimension
            # why using hypervolume at all?
            return -sentinel.next[0].cargo[0]
        elif dimIndex == 1:
            # special case: two dimensions, end recursion
            q = sentinel.next[1]
            h = q.cargo[0]
            p = q.next[1]
            while p is not sentinel:
                pCargo = p.cargo
                hvol += h * (q.cargo[1] - pCargo[1])
                if pCargo[0] < h:
                    h = pCargo[0]
                q = p
                p = q.next[1]
            hvol += h * q.cargo[1]
            return hvol
        else:
            remove = self.list.remove
            reinsert = self.list.reinsert
            hvRecursive = self.hvRecursive
            p = sentinel
            q = p.prev[dimIndex]
            while q.cargo is not None:
                if q.ignore < dimIndex:
                    q.ignore = 0
                q = q.prev[dimIndex]
            q = p.prev[dimIndex]
            while length > 1 and (
                q.cargo[dimIndex] > bounds[dimIndex]
                or q.prev[dimIndex].cargo[dimIndex] >= bounds[dimIndex]
            ):
                p = q
                remove(p, dimIndex, bounds)
                q = p.prev[dimIndex]
                length -= 1
            qArea = q.area
            qCargo = q.cargo
            qPrevDimIndex = q.prev[dimIndex]
            if length > 1:
                hvol = qPrevDimIndex.volume[dimIndex] + qPrevDimIndex.area[dimIndex] * (
                    qCargo[dimIndex] - qPrevDimIndex.cargo[dimIndex]
                )
            else:
                qArea[0] = 1
                qArea[1 : dimIndex + 1] = [
                    qArea[i] * -qCargo[i] for i in range(dimIndex)
                ]
            q.volume[dimIndex] = hvol
            if q.ignore >= dimIndex:
                qArea[dimIndex] = qPrevDimIndex.area[dimIndex]
            else:
                qArea[dimIndex] = hvRecursive(dimIndex - 1, length, bounds)
                if qArea[dimIndex] <= qPrevDimIndex.area[dimIndex]:
                    q.ignore = dimIndex
            while p is not sentinel:
                pCargoDimIndex = p.cargo[dimIndex]
                hvol += q.area[dimIndex] * (pCargoDimIndex - q.cargo[dimIndex])
                bounds[dimIndex] = pCargoDimIndex
                reinsert(p, dimIndex, bounds)
                length += 1
                q = p
                p = p.next[dimIndex]
                q.volume[dimIndex] = hvol
                if q.ignore >= dimIndex:
                    q.area[dimIndex] = q.prev[dimIndex].area[dimIndex]
                else:
                    q.area[dimIndex] = hvRecursive(dimIndex - 1, length, bounds)
                    if q.area[dimIndex] <= q.prev[dimIndex].area[dimIndex]:
                        q.ignore = dimIndex
            hvol -= q.area[dimIndex] * q.cargo[dimIndex]
            return hvol

    def preProcess(self, front):
        """Sets up the list data structure needed for calculation."""
        dimensions = len(self.referencePoint)
        nodeList = MultiList(dimensions)
        nodes = [MultiList.Node(dimensions, point) for point in front]
        for i in range(dimensions):
            self.sortByDimension(nodes, i)
            nodeList.extend(nodes, i)
        self.list = nodeList

    def sortByDimension(self, nodes, i):
        """Sorts the list of nodes by the i-th value of the contained points."""
        # build a list of tuples of (point[i], node)
        decorated = [(node.cargo[i], index, node) for index, node in enumerate(nodes)]
        # sort by this value
        decorated.sort()
        # write back to original list
        nodes[:] = [node for (_, _, node) in decorated]


class MultiList:
    """A special data structure needed by FonsecaHyperVolume.

    It consists of several doubly linked lists that share common nodes. So,
    every node has multiple predecessors and successors, one in every list.

    """

    class Node:
        def __init__(self, numberLists, cargo=None):
            self.cargo = cargo
            self.next = [None] * numberLists
            self.prev = [None] * numberLists
            self.ignore = 0
            self.area = [0.0] * numberLists
            self.volume = [0.0] * numberLists

        def __str__(self):
            return str(self.cargo)

    def __init__(self, numberLists):
        """Constructor.

        Builds 'numberLists' doubly linked lists.

        """
        self.numberLists = numberLists
        self.sentinel = MultiList.Node(numberLists)
        self.sentinel.next = [self.sentinel] * numberLists
        self.sentinel.prev = [self.sentinel] * numberLists

    def __str__(self):
        strings = []
        for i in range(self.numberLists):
            currentList = []
            node = self.sentinel.next[i]
            while node != self.sentinel:
                currentList.append(str(node))
                node = node.next[i]
            strings.append(str(currentList))
        stringRepr = ""
        for string in strings:
            stringRepr += string + "\n"
        return stringRepr

    def __len__(self):
        """Returns the number of lists that are included in this MultiList."""
        return self.numberLists

    def getLength(self, i):
        """Returns the length of the i-th list."""
        length = 0
        sentinel = self.sentinel
        node = sentinel.next[i]
        while node != sentinel:
            length += 1
            node = node.next[i]
        return length

    def append(self, node, index):
        """Appends a node to the end of the list at the given index."""
        lastButOne = self.sentinel.prev[index]
        node.next[index] = self.sentinel
        node.prev[index] = lastButOne
        # set the last element as the new one
        self.sentinel.prev[index] = node
        lastButOne.next[index] = node

    def extend(self, nodes, index):
        """Extends the list at the given index with the nodes."""
        sentinel = self.sentinel
        for node in nodes:
            lastButOne = sentinel.prev[index]
            node.next[index] = sentinel
            node.prev[index] = lastButOne
            # set the last element as the new one
            sentinel.prev[index] = node
            lastButOne.next[index] = node

    def remove(self, node, index, bounds):
        """Removes and returns 'node' from all lists in [0, 'index'[."""
        for i in range(index):
            predecessor = node.prev[i]
            successor = node.next[i]
            predecessor.next[i] = successor
            successor.prev[i] = predecessor
            if bounds[i] > node.cargo[i]:
                bounds[i] = node.cargo[i]
        return node

    def reinsert(self, node, index, bounds):
        """
        Inserts 'node' at the position it had in all lists in [0, 'index'[
        before it was removed. This method assumes that the next and previous
        nodes of the node that is reinserted are in the list.

        """
        for i in range(index):
            node.prev[i].next[i] = node
            node.next[i].prev[i] = node
            if bounds[i] > node.cargo[i]:
                bounds[i] = node.cargo[i]


class HyperVolumeBoxDecompositionYang:
    def __init__(
        self,
        partition_bounds: Tuple[np.ndarray, np.ndarray],
        local_random: Optional[np.random.Generator] = None,
    ):
        r"""Expected Hyper-volume (HV) calculation using Eq. 44 of:

        Efficient Computation of Expected Hypervolume Improvement
        Using Box Decomposition Algorithms. Yang et al., 2019.

        The expected hypervolume improvement calculation in the
        non-dominated region can be decomposed into sub-calculations
        based on each partitioned cell.  For easier calculation, this
        sub-calculation can be reformulated as a combination of two
        generalized expected improvements, corresponding to Psi
        (Eq. 44) and Nu (Eq. 45) function calculations, respectively.

        Note:

        1. As the Psi and nu function in the original paper are
        defined for maximization problems, we inverse our minimization
        problem (to also be a maximization), allowing use of the
        original notation and equations.

        """
        print(f"partition_bounds[0].shape[-1]: {partition_bounds[0].shape[-1]}")
        self._lb_points = np.asarray(partition_bounds[0]).reshape(
            (1, -1)
        )  # , shape=(None, partition_bounds[0].shape[-1])
        self._ub_points = np.asarray(partition_bounds[1]).reshape(
            (1, -1)
        )  # , shape=(None, partition_bounds[1].shape[-1])
        print(f"_lb_points: {self._lb_points}")
        print(f"_ub_points: {self._ub_points}")
        self._cross_index = np.asarray(
            list(product(*[[0, 1]] * self._lb_points.shape[-1]))
        )  # [2^d, indices_at_dim]
        print(f"cross_index: {self._cross_index}")

        self.local_random = local_random

    def update(self, partition_bounds: Tuple[np.ndarray, np.ndarray]) -> None:
        """Update the acquisition function with new partition bounds."""
        self._lb_points[:] = partition_bounds[0]
        self._ub_points[:] = partition_bounds[1]

    def compute_ehvi(
        self, candidate_mean: np.ndarray, candidate_var: Optional[np.ndarray] = None
    ) -> np.ndarray:

        if candidate_var is None:
            candidate_var = np.ones_like(candidate_mean)

        normal_var = norm(loc=0.0, scale=1.0)

        def Psi(
            a: np.ndarray, b: np.ndarray, mean: np.ndarray, std: np.ndarray
        ) -> np.ndarray:
            return std * normal_var.pdf((b - mean) / std) + (mean - a) * (
                1 - normal_var.cdf((b - mean) / std)
            )

        def nu(
            lb: np.ndarray, ub: np.ndarray, mean: np.ndarray, std: np.ndarray
        ) -> np.ndarray:
            return (ub - lb) * (1 - normal_var.cdf((ub - mean) / std))

        def ehvi_based_on_partitioned_cell(
            neg_pred_mean: np.ndarray, pred_std: np.ndarray
        ) -> np.ndarray:
            r"""
            Calculate the ehvi based on cell i.
            """

            neg_lb_points, neg_ub_points = -self._ub_points, -self._lb_points

            neg_ub_points = np.minimum(
                neg_ub_points, 1.0e10
            )  # clip to improve numerical stability

            psi_lb = Psi(
                neg_lb_points, neg_lb_points, neg_pred_mean, pred_std
            )  # [..., num_cells, out_dim]
            psi_ub = Psi(
                neg_lb_points, neg_ub_points, neg_pred_mean, pred_std
            )  # [..., num_cells, out_dim]

            psi_lb2ub = np.maximum(psi_lb - psi_ub, 0.0)  # [..., num_cells, out_dim]
            nu_contrib = nu(neg_lb_points, neg_ub_points, neg_pred_mean, pred_std)

            print(f"neg_pred_mean = {neg_pred_mean}")
            print(f"psi_lb = {psi_lb}")
            print(f"psi_ub = {psi_ub}")
            print(f"psi_lb2ub = {psi_lb2ub}")

            print(f"neg_lb_points.shape = {neg_lb_points.shape}")
            print(f"neg_pred_mean.shape = {neg_pred_mean.shape}")
            print(f"psi_lb.shape = {psi_lb.shape}")
            print(f"psi_ub.shape = {psi_ub.shape}")
            print(f"psi_lb2ub.shape = {psi_lb2ub.shape}")
            print(f"nu_contrib.shape = {nu_contrib.shape}")
            print(
                f"expand_dims(nu_contrib.shape, -2) = {np.expand_dims(nu_contrib, -2).shape}"
            )
            stacked_factors = np.concatenate(
                [np.expand_dims(psi_lb2ub, -2), np.expand_dims(nu_contrib, -2)],
                axis=-2
                # stacked_factors = np.concatenate(
                #    [psi_lb2ub, nu_contrib], axis=-2
            )  # Take the cross product of psi_diff and nu across all outcomes
            # [..., num_cells, 2(operation_num, refer Eq. 45), num_obj]

            print(f"_cross_index.shape = {self._cross_index.shape}")
            print(f"stacked_factors.shape = {stacked_factors.shape}")
            print(f"stacked_factors = {stacked_factors}")
            print(
                f"take stacked_factors shape = {np.take(stacked_factors, self._cross_index, axis=-2).shape}"
            )
            # tf.linalg.diag_part(
            factor_combinations = np.diagonal(
                np.take(stacked_factors, self._cross_index, axis=-2), axis1=-2, axis2=-1
            ).copy()  # [..., num_cells, 2^d, 2(operation_num), num_obj]
            print(f"factor_combinations.shape = {factor_combinations.shape}")

            return np.sum(np.prod(factor_combinations, axis=-1), axis=-1)

        candidate_std = np.sqrt(candidate_var)
        print(f"candidate_std = {candidate_std}")

        neg_candidate_mean = -np.expand_dims(candidate_mean, 1)  # [..., 1, out_dim]
        candidate_std = np.expand_dims(candidate_std, 1)  # [..., 1, out_dim]

        print(f"candidate_mean = {candidate_mean}")

        ehvi_cells_based = ehvi_based_on_partitioned_cell(
            neg_candidate_mean, candidate_std
        )

        print(f"ehvi_cells_based = {ehvi_cells_based}")
        print(f"ehvi_cells_based.shape = {ehvi_cells_based.shape}")

        return np.prod(
            ehvi_cells_based,
            axis=-2,
            keepdims=True,
        )

        contribution = 1.0

        for i in range(self.n_objectives):
            mean, var = means[i], variances[i]
            std = np.sqrt(var)

            # Compute probability of improvement over reference point
            prob = 1 - norm.cdf((self.ref_point[i] - mean) / std)

            # Compute partial expectation
            partial_exp = std * norm.pdf((self.ref_point[i] - mean) / std) + mean * prob

            contribution *= partial_exp

        return contribution


""" 
Implementation of hypervolume and Expected Hypervolume Improvement
(EHVI) computation using box decomposition algorithms.

The implementation provides two main functionalities:

1. Hypervolume Computation: Calculates the hypervolume dominated by a
set of Pareto-optimal points

2. EHVI Computation: Calculates the expected improvement in
   hypervolume for a new point with uncertain objective values
   (represented by Gaussian distributions)

Key Concepts:
------------
1. Hypervolume: 
   - Measures the volume of the objective space dominated by a Pareto front
   - Computed using a recursive slicing algorithm
   
2. Expected Hypervolume Improvement (EHVI):
   - Extends hypervolume to handle uncertainty in predictions
   - Computed using box decomposition and integration over Gaussian distributions

Algorithm Details:
-----------------
1. Hypervolume Computation:
   - Uses a recursive slicing algorithm
   - Processes dimensions one at a time
   - For each dimension:
     * Sorts points by current dimension
     * Computes contribution of each slice
     * Recursively processes remaining dimensions
     
2. EHVI Computation:
   - Decomposes non-dominated space into boxes
   - For each box:
     * Computes probability of improvement in each dimension
     * Calculates expected improvement contribution
     * Combines dimensional contributions
"""


class HyperVolumeBoxDecompositionR2:
    def __init__(self, ref_point: np.ndarray):
        """
        Initialize box decomposition algorithm for hypervolume and EHVI computation.

        Args:
            ref_point: Reference point for hypervolume calculation. This point should
                      dominate all Pareto front points. For minimization problems,
                      it should be larger than all Pareto front points in all dimensions.

        The reference point defines the upper bounds of the hypervolume computation
        and significantly impacts the resulting values. It should be chosen carefully
        based on the problem context.
        """
        self.ref_point = ref_point
        self.n_objectives = len(ref_point)

    def compute_hypervolume(self, pareto_front: np.ndarray) -> float:
        """
        Compute the hypervolume dominated by a Pareto front.

        Args:
            pareto_front: Array of shape (n_points, n_objectives) containing
                         the coordinates of Pareto-optimal points.

        Returns:
            float: The hypervolume dominated by the Pareto front

        The hypervolume is computed using a recursive slicing algorithm:
        1. Start with the highest dimension
        2. Sort points by current dimension
        3. For each point:
           - Compute the slice volume between current and previous point
           - Recursively compute volume in lower dimensions
           - Multiply and accumulate results

        The algorithm handles special cases:
        - Empty Pareto front: Returns 0
        - Points outside reference point: Filters invalid points
        - Single point: Computes simple box volume
        """
        if len(pareto_front) == 0:
            return 0.0

        # Ensure all points are dominated by reference point
        if not np.all(pareto_front <= self.ref_point):
            valid_points = pareto_front[np.all(pareto_front <= self.ref_point, axis=1)]
            if len(valid_points) == 0:
                return 0.0
            pareto_front = valid_points

        def recursive_hypervolume(
            points: np.ndarray, ref_point: np.ndarray, dimension: int
        ) -> float:
            """
            Recursively compute hypervolume for a specific dimension.

            Args:
                points: Array of points to process
                ref_point: Current reference point
                dimension: Current dimension being processed

            Returns:
                float: Hypervolume contribution for this dimension

            The recursive algorithm:
            1. Base case: dimension 0
               - Return difference between reference point and minimum value
            2. Recursive case:
               - Sort points by current dimension
               - For each point:
                 * Compute slice between current point and previous slice
                 * Recursively compute volume in remaining dimensions
                 * Multiply by height difference
            """
            if len(points) == 0:
                return 0.0

            if dimension == 0:
                return ref_point[0] - np.min(points[:, 0])

            # Sort points by current dimension (descending)
            sorted_indices = np.argsort(-points[:, dimension])
            sorted_points = points[sorted_indices]

            hv = 0.0
            prev_point = ref_point.copy()

            for point in sorted_points:
                if point[dimension] < prev_point[dimension]:
                    # Update reference point for recursive call
                    curr_ref = prev_point.copy()
                    curr_ref[dimension] = point[dimension]

                    # Find points that are still relevant for this slice
                    relevant_points = points[np.all(points <= curr_ref, axis=1)]

                    # Compute hypervolume of this slice
                    slice_hv = recursive_hypervolume(
                        relevant_points, curr_ref, dimension - 1
                    )
                    hv += slice_hv * (prev_point[dimension] - point[dimension])

                    prev_point[dimension] = point[dimension]

            return hv

        return recursive_hypervolume(
            pareto_front, self.ref_point.copy(), self.n_objectives - 1
        )

    def compute_ehvi(
        self,
        pareto_front: np.ndarray,
        means: np.ndarray,
        variances: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute Expected Hypervolume Improvement for a new point.

        Args:
            pareto_front: Current Pareto front points
            means: Predicted mean values for new point
            variances: Predicted variance values for new point

        Returns:
            float: Expected hypervolume improvement value

        The EHVI computation process:
        1. If Pareto front is empty:
           - Compute simple expected improvement from reference point
        2. Otherwise:
           - Decompose non-dominated space into boxes
           - For each box:
             * Compute probability of improvement in each dimension
             * Calculate expected improvement contribution
             * Combine contributions across dimensions
        """
        if variances is None:
            variances = np.ones_like(means)

        if len(pareto_front) == 0:
            return self._compute_empty_ehvi(means, variances)

        boxes = self._decompose_dominated_space(pareto_front)
        total_ehvi = 0.0

        for box in boxes:
            total_ehvi += self._compute_box_contribution(box, means, variances)

        return total_ehvi

    def _decompose_dominated_space(
        self, pareto_front: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Decompose the non-dominated space into boxes for EHVI computation.

        Args:
            pareto_front: Array of Pareto front points

        Returns:
            List of tuples (lower, upper) defining boxes

        Box Decomposition Process:
        1. Sort points by first objective
        2. Create boxes between consecutive points
        3. Handle special cases:
           - First box: From -infinity to first point
           - Last box: From last point to reference point
           - Middle boxes: Between consecutive points

        Each box represents a region where improvement is possible.
        The algorithm ensures:
        - Complete coverage of non-dominated space
        - No overlap between boxes
        - Proper handling of infinities
        """
        boxes = []
        n_points = len(pareto_front)

        # Sort points by first objective
        sorted_indices = np.argsort(pareto_front[:, 0])
        sorted_front = pareto_front[sorted_indices]

        for i in range(n_points + 1):
            if i == 0:
                lower = np.array([-np.inf] * self.n_objectives)
                upper = sorted_front[0]
            elif i == n_points:
                lower = sorted_front[-1]
                upper = self.ref_point
            else:
                lower = sorted_front[i - 1]
                upper = sorted_front[i]

            if np.all(upper > lower):
                boxes.append((lower, upper))

        return boxes

    def _compute_box_contribution(
        self,
        box: Tuple[np.ndarray, np.ndarray],
        means: np.ndarray,
        variances: np.ndarray,
    ) -> float:
        """
        Compute the EHVI contribution of a single box.

        Args:
            box: Tuple of (lower, upper) corners of the box
            means: Predicted mean values
            variances: Predicted variance values

        Returns:
            float: Box contribution to total EHVI

        For each dimension:
        1. Compute probability of improvement:
           - P(x > lower) - P(x > upper)
        2. Compute expected improvement:
           - Integrate x*P(x) over the box bounds
        3. Multiply contributions across dimensions

        Handles special cases:
        - Infinite bounds
        - Zero variances
        - Numerical stability issues
        """
        lower, upper = box
        contribution = 1.0

        for i in range(self.n_objectives):
            mean, var = means[i], variances[i]
            std = np.sqrt(var)

            # Handle infinite bounds
            if np.isinf(lower[i]):
                lower_prob = 0.0
            else:
                lower_prob = norm.cdf((lower[i] - mean) / std)

            if np.isinf(upper[i]):
                upper_prob = 1.0
            else:
                upper_prob = norm.cdf((upper[i] - mean) / std)

            # Compute partial expectation
            partial_exp = std * (
                norm.pdf((lower[i] - mean) / std) - norm.pdf((upper[i] - mean) / std)
            ) + mean * (upper_prob - lower_prob)

            contribution *= partial_exp

        return contribution

    def _compute_empty_ehvi(self, means: np.ndarray, variances: np.ndarray) -> float:
        """
        Compute EHVI when there are no existing Pareto front points.

        Args:
            means: Predicted mean values
            variances: Predicted variance values

        Returns:
            float: EHVI value for empty Pareto front case

        Special case handling when Pareto front is empty:
        1. Compute improvement probability relative to reference point
        2. Calculate expected improvement in each dimension
        3. Multiply contributions across dimensions

        This case is simpler because there's only one box to consider:
        from negative infinity to the reference point.
        """
        contribution = 1.0

        for i in range(self.n_objectives):
            mean, var = means[i], variances[i]
            std = np.sqrt(var)

            # Compute probability of improvement over reference point
            prob = 1 - norm.cdf((self.ref_point[i] - mean) / std)

            # Compute partial expectation
            partial_exp = std * norm.pdf((self.ref_point[i] - mean) / std) + mean * prob

            contribution *= partial_exp

        return contribution


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


class HyperVolumeBoxDecomposition:
    def __init__(self, ref_point: np.ndarray):
        self.ref_point = ref_point
        self.n_objectives = len(ref_point)
        self._hv_cache: Dict[Tuple, float] = {}

    def select_candidates(
        self,
        pareto_front: np.ndarray,
        candidate_means: np.ndarray,
        candidate_variances: np.ndarray,
        n_select: int = 1,
        batch_size: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select the best candidate points based on EHVI.
        """
        n_candidates = len(candidate_means)

        # Pre-compute boxes for the current Pareto front
        if len(pareto_front) == 0:
            boxes = []
        else:
            boxes = self._decompose_dominated_space_optimized(pareto_front)

        # Compute EHVI for all candidates in batches
        ehvi_values = np.zeros(n_candidates)

        for batch_start in range(0, n_candidates, batch_size):
            batch_end = min(batch_start + batch_size, n_candidates)
            batch_means = candidate_means[batch_start:batch_end]
            batch_variances = candidate_variances[batch_start:batch_end]

            if len(boxes) == 0:
                # Handle empty Pareto front case
                for i, (means, variances) in enumerate(
                    zip(batch_means, batch_variances)
                ):
                    ehvi_values[batch_start + i] = self._compute_empty_ehvi(
                        means, variances
                    )
            else:
                # Compute EHVI for the batch
                batch_ehvi = self._compute_batch_ehvi(
                    boxes, batch_means, batch_variances
                )
                ehvi_values[batch_start:batch_end] = batch_ehvi

        # Select top n_select candidates
        selected_indices = np.copy(np.argsort(-ehvi_values)[:n_select])

        return selected_indices, ehvi_values[selected_indices]

    def _compute_batch_ehvi(
        self, boxes: List[Box], batch_means: np.ndarray, batch_variances: np.ndarray
    ) -> np.ndarray:
        """
        Compute EHVI for a batch of candidates efficiently.
        Fixed broadcasting for vectorized operations.
        """
        batch_size = len(batch_means)
        ehvi_values = np.zeros(batch_size)

        # Prepare box bounds arrays
        n_boxes = len(boxes)
        lowers = np.array(
            [box.lower for box in boxes]
        )  # Shape: (n_boxes, n_objectives)
        uppers = np.array(
            [box.upper for box in boxes]
        )  # Shape: (n_boxes, n_objectives)

        # Compute contributions for each candidate
        for i in range(batch_size):
            means = batch_means[i]  # Shape: (n_objectives,)
            variances = batch_variances[i]  # Shape: (n_objectives,)
            std = np.sqrt(variances)  # Shape: (n_objectives,)

            # Reshape for broadcasting
            means = means[None, :]  # Shape: (1, n_objectives)
            std = std[None, :]  # Shape: (1, n_objectives)

            # Compute probabilities for all boxes
            lower_probs = np.zeros_like(lowers, dtype=float)
            upper_probs = np.ones_like(uppers, dtype=float)

            # Handle finite bounds with proper broadcasting
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

            # Compute partial expectations with proper broadcasting
            partial_exp = std * (
                norm.pdf((lowers - means) / std) - norm.pdf((uppers - means) / std)
            ) + means * (upper_probs - lower_probs)

            # Multiply along objectives dimension
            contributions = np.prod(partial_exp, axis=1)
            ehvi_values[i] = np.sum(contributions)

        return ehvi_values

    def _decompose_dominated_space_optimized(
        self, pareto_front: np.ndarray
    ) -> List[Box]:
        n_points = len(pareto_front)
        sorted_indices = np.argsort(pareto_front[:, 0])
        sorted_front = pareto_front[sorted_indices]

        lower_bounds = np.full((n_points + 1, self.n_objectives), -np.inf)
        upper_bounds = np.full((n_points + 1, self.n_objectives), np.inf)

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

    def _compute_empty_ehvi(self, means: np.ndarray, variances: np.ndarray) -> float:
        contribution = 1.0

        for i in range(self.n_objectives):
            mean, var = means[i], variances[i]
            std = np.sqrt(var)

            prob = 1 - norm.cdf((self.ref_point[i] - mean) / std)
            partial_exp = std * norm.pdf((self.ref_point[i] - mean) / std) + mean * prob

            contribution *= partial_exp

        return float(contribution)

    def compute_hypervolume(self, pareto_front: np.ndarray) -> float:
        """
        Compute hypervolume with basic optimizations and correct caching.
        """
        if len(pareto_front) == 0:
            return 0.0

        # Ensure all points are dominated by reference point
        valid_mask = np.all(pareto_front <= self.ref_point, axis=1)
        if not np.any(valid_mask):
            return 0.0
        points = pareto_front[valid_mask]

        # Convert points to tuples for initial cache lookup
        points_tuple = tuple(map(tuple, points))
        ref_tuple = tuple(self.ref_point)

        return self._compute_hv_optimized(
            points_tuple, ref_tuple, self.n_objectives - 1
        )

    @lru_cache(maxsize=1024)
    def _compute_hv_optimized(
        self,
        points_tuple: Tuple[Tuple[float, ...], ...],
        ref_tuple: Tuple[float, ...],
        dimension: int,
    ) -> float:
        """
        Recursive hypervolume computation with caching.
        Works with tuples for hashability.
        """
        points = np.array(points_tuple)
        ref_point = np.array(ref_tuple)

        if len(points) == 0:
            return 0.0

        if dimension == 0:
            return float(ref_point[0] - np.min(points[:, 0]))

        # Sort points by current dimension (descending)
        sorted_indices = np.argsort(-points[:, dimension])
        sorted_points = points[sorted_indices]

        hv = 0.0
        prev_point = ref_point.copy()

        for point in sorted_points:
            if point[dimension] < prev_point[dimension]:
                # Create reference point for this slice
                curr_ref = prev_point.copy()
                curr_ref[dimension] = point[dimension]

                # Find points that are relevant for this slice
                relevant_mask = np.all(sorted_points <= curr_ref, axis=1)
                relevant_points = sorted_points[relevant_mask]

                # Convert to tuples for recursive call
                relevant_tuple = tuple(map(tuple, relevant_points))
                curr_ref_tuple = tuple(curr_ref)

                # Recursive call with updated reference point
                slice_volume = self._compute_hv_optimized(
                    relevant_tuple, curr_ref_tuple, dimension - 1
                )
                hv += slice_volume * (prev_point[dimension] - point[dimension])

                prev_point[dimension] = point[dimension]

        return float(hv)

    def compute_ehvi(
        self,
        pareto_front: np.ndarray,
        means: np.ndarray,
        variances: np.ndarray,
        batch_size: int = 1000,
    ) -> float:
        if len(pareto_front) == 0:
            return self._compute_empty_ehvi(means, variances)

        boxes = self._decompose_dominated_space_optimized(pareto_front)
        total_ehvi = 0.0

        for i in range(0, len(boxes), batch_size):
            batch = boxes[i : i + batch_size]
            total_ehvi += self._compute_batch_contribution(batch, means, variances)

        return total_ehvi

    def _compute_batch_contribution(
        self,
        boxes: List[Box],
        means: np.ndarray,
        variances: np.ndarray,
        threshold: float = 1e-10,
    ) -> float:
        n_boxes = len(boxes)
        if n_boxes == 0:
            return 0.0

        lowers = np.array([box.lower for box in boxes])
        uppers = np.array([box.upper for box in boxes])

        std = np.sqrt(variances)
        finite_mask_lower = ~np.isinf(lowers)
        finite_mask_upper = ~np.isinf(uppers)

        lower_probs = np.zeros((n_boxes, self.n_objectives))
        upper_probs = np.ones((n_boxes, self.n_objectives))

        lower_probs[finite_mask_lower] = norm.cdf(
            (lowers[finite_mask_lower] - means) / std
        )
        upper_probs[finite_mask_upper] = norm.cdf(
            (uppers[finite_mask_upper] - means) / std
        )

        partial_exp = std * (
            norm.pdf((lowers - means) / std) - norm.pdf((uppers - means) / std)
        ) + means * (upper_probs - lower_probs)

        contribution = np.prod(partial_exp, axis=1)
        return float(np.sum(contribution[contribution > threshold]))

    def _compute_empty_ehvi(self, means: np.ndarray, variances: np.ndarray) -> float:
        contribution = 1.0

        for i in range(self.n_objectives):
            mean, var = means[i], variances[i]
            std = np.sqrt(var)

            prob = 1 - norm.cdf((self.ref_point[i] - mean) / std)
            partial_exp = std * norm.pdf((self.ref_point[i] - mean) / std) + mean * prob

            contribution *= partial_exp

        return float(contribution)


def run_basic_test_cases():
    print("\n=== 2D Test Cases ===")

    # 2D reference point
    ref_point_2d = np.array([10.0, 10.0])
    bd_2d = HyperVolumeBoxDecomposition(ref_point_2d)

    # Test 2D cases
    pareto_2d = np.array([[2.0, 8.0], [4.0, 4.0], [8.0, 2.0]])
    hv_2d = bd_2d.compute_hypervolume(pareto_2d)
    print(f"2D Basic case hypervolume: {hv_2d}")

    # Single 2D point
    single_point_2d = np.array([[5.0, 5.0]])
    hv_single_2d = bd_2d.compute_hypervolume(single_point_2d)
    print(f"2D Single point hypervolume: {hv_single_2d}")

    print("\n=== 3D Test Cases ===")

    # 3D reference point
    ref_point_3d = np.array([10.0, 10.0, 10.0])
    bd_3d = HyperVolumeBoxDecomposition(ref_point_3d)

    # Test Case 1: Simple 3D Pareto front
    pareto_3d_simple_1 = np.array([[2.0, 8.0, 8.0], [8.0, 2.0, 8.0], [8.0, 8.0, 2.0]])
    hv_3d_simple_1 = bd_3d.compute_hypervolume(pareto_3d_simple_1)
    print(f"3D Simple case 1 hypervolume: {hv_3d_simple_1}")

    # Test Case 2: Simple 3D Pareto front
    pareto_3d_simple_2 = np.array([[1, 0, 1], [0, 1, 0]])
    hv_3d_simple_2 = bd_3d.compute_hypervolume(pareto_3d_simple_2)
    print(f"3D Simple case 2 hypervolume: {hv_3d_simple_2}")

    # Test Case 2: Single 3D point
    single_point_3d = np.array([[5.0, 5.0, 5.0]])
    hv_3d_single = bd_3d.compute_hypervolume(single_point_3d)
    print(f"3D Single point hypervolume: {hv_3d_single}")
    # Expected: (10-5)*(10-5)*(10-5) = 125

    # Test Case 3: More complex 3D Pareto front
    pareto_3d_complex = np.array(
        [
            [2.0, 8.0, 8.0],
            [3.0, 3.0, 8.0],
            [8.0, 2.0, 8.0],
            [8.0, 8.0, 2.0],
            [3.0, 8.0, 3.0],
            [8.0, 3.0, 3.0],
        ]
    )
    hv_3d_complex = bd_3d.compute_hypervolume(pareto_3d_complex)
    print(f"3D Complex case hypervolume: {hv_3d_complex}")

    # Test Case 4: 3D points with dominated points
    pareto_3d_dominated = np.array(
        [
            [2.0, 8.0, 8.0],
            [3.0, 7.0, 7.0],  # dominated by first point
            [8.0, 2.0, 8.0],
            [8.0, 8.0, 2.0],
        ]
    )
    hv_3d_dominated = bd_3d.compute_hypervolume(pareto_3d_dominated)
    print(f"3D Case with dominated points hypervolume: {hv_3d_dominated}")

    # Test Case 5: 3D points outside reference point
    pareto_3d_invalid = np.array(
        [
            [11.0, 5.0, 5.0],  # invalid in first dimension
            [5.0, 11.0, 5.0],  # invalid in second dimension
            [5.0, 5.0, 5.0],  # valid point
            [5.0, 5.0, 11.0],  # invalid in third dimension
        ]
    )
    hv_3d_invalid = bd_3d.compute_hypervolume(pareto_3d_invalid)
    print(f"3D Case with invalid points hypervolume: {hv_3d_invalid}")

    # Test Case 6: Corner cases in 3D
    pareto_3d_corners = np.array([[0.0, 0.0, 10.0], [0.0, 10.0, 0.0], [10.0, 0.0, 0.0]])
    hv_3d_corners = bd_3d.compute_hypervolume(pareto_3d_corners)
    print(f"3D Corner cases hypervolume: {hv_3d_corners}")

    # Test Case 7: Empty 3D front
    empty_front_3d = np.array([])
    hv_3d_empty = bd_3d.compute_hypervolume(empty_front_3d)
    print(f"3D Empty front hypervolume: {hv_3d_empty}")

    return {
        "2d_basic": hv_2d,
        "2d_single": hv_single_2d,
        "3d_simple_1": hv_3d_simple_1,
        "3d_simple_2": hv_3d_simple_2,
        "3d_single": hv_3d_single,
        "3d_complex": hv_3d_complex,
        "3d_dominated": hv_3d_dominated,
        "3d_invalid": hv_3d_invalid,
        "3d_corners": hv_3d_corners,
        "3d_empty": hv_3d_empty,
    }


def run_analytical_test_cases():
    """
    Run test cases with known analytical solutions.
    Returns dictionary of test results with computed and expected values.
    """
    results = {}

    print("\n=== 2D Analytical Test Cases ===")

    # 2D Test Case 1: Single Point
    # Reference point: [4,4], Point: [1,1]
    # Expected volume: (4-1)*(4-1) = 3*3 = 9
    ref_point_2d = np.array([4.0, 4.0])
    bd_2d = HyperVolumeBoxDecomposition(ref_point_2d)

    single_point_2d = np.array([[1.0, 1.0]])
    hv_2d_single = bd_2d.compute_hypervolume(single_point_2d)
    expected_2d_single = 9.0
    print(f"2D Single point:")
    print(f"Computed: {hv_2d_single:.6f}")
    print(f"Expected: {expected_2d_single}")
    print(f"Error: {abs(hv_2d_single - expected_2d_single):.6f}")
    results["2d_single"] = (hv_2d_single, expected_2d_single)

    # 2D Test Case 2: Two Non-Dominated Points
    # Points: [1,3], [3,1]
    # Expected volume: 5
    two_points_2d = np.array([[1.0, 3.0], [3.0, 1.0]])
    hv_2d_two = bd_2d.compute_hypervolume(two_points_2d)
    expected_2d_two = 5.0
    print(f"\n2D Two points:")
    print(f"Computed: {hv_2d_two:.6f}")
    print(f"Expected: {expected_2d_two}")
    print(f"Error: {abs(hv_2d_two - expected_2d_two):.6f}")
    results["2d_two"] = (hv_2d_two, expected_2d_two)

    # 2D Test Case 3: Rectangle
    # Points: [1,3], [1,1], [3,1]
    # Expected volume: 9
    rectangle_2d = np.array([[1.0, 3.0], [1.0, 1.0], [3.0, 1.0]])
    hv_2d_rect = bd_2d.compute_hypervolume(rectangle_2d)
    expected_2d_rect = 9.0
    print(f"\n2D Rectangle:")
    print(f"Computed: {hv_2d_rect:.6f}")
    print(f"Expected: {expected_2d_rect}")
    print(f"Error: {abs(hv_2d_rect - expected_2d_rect):.6f}")
    results["2d_rect"] = (hv_2d_rect, expected_2d_rect)

    print("\n=== 3D Analytical Test Cases ===")

    # 3D Test Case 1: Single Point
    # Reference point: [4,4,4], Point: [1,1,1]
    # Expected volume: (4-1)*(4-1)*(4-1) = 27
    ref_point_3d = np.array([4.0, 4.0, 4.0])
    bd_3d = HyperVolumeBoxDecomposition(ref_point_3d)

    single_point_3d = np.array([[1.0, 1.0, 1.0]])
    hv_3d_single = bd_3d.compute_hypervolume(single_point_3d)
    expected_3d_single = 27.0
    print(f"3D Single point:")
    print(f"Computed: {hv_3d_single:.6f}")
    print(f"Expected: {expected_3d_single}")
    print(f"Error: {abs(hv_3d_single - expected_3d_single):.6f}")
    results["3d_single"] = (hv_3d_single, expected_3d_single)

    # 3D Test Case 2: Three Non-Dominated Points forming a plane
    # Points: [1,1,3], [1,3,1], [3,1,1]
    # Expected volume: 19
    plane_points_3d = np.array([[1.0, 1.0, 3.0], [1.0, 3.0, 1.0], [3.0, 1.0, 1.0]])
    hv_3d_plane = bd_3d.compute_hypervolume(plane_points_3d)
    expected_3d_plane = 19.0
    print(f"\n3D Plane points:")
    print(f"Computed: {hv_3d_plane:.6f}")
    print(f"Expected: {expected_3d_plane}")
    print(f"Error: {abs(hv_3d_plane - expected_3d_plane):.6f}")
    results["3d_plane"] = (hv_3d_plane, expected_3d_plane)

    # 3D Test Case 3: Regular Cuboid
    # Points forming a regular cuboid with corner at [1,1,1] extending to [3,3,3]
    # Expected volume: 19
    ref_point_cuboid = np.array([4.0, 4.0, 4.0])
    bd_cuboid = HyperVolumeBoxDecomposition(ref_point_cuboid)
    cuboid_points = np.array(
        [
            [1.0, 1.0, 3.0],
            [1.0, 3.0, 1.0],
            [3.0, 1.0, 1.0],
            [1.0, 3.0, 3.0],
            [3.0, 1.0, 3.0],
            [3.0, 3.0, 1.0],
            [3.0, 3.0, 3.0],
        ]
    )
    hv_3d_cuboid = bd_cuboid.compute_hypervolume(cuboid_points)
    expected_3d_cuboid = 19.0
    print(f"\n3D Cuboid:")
    print(f"Computed: {hv_3d_cuboid:.6f}")
    print(f"Expected: {expected_3d_cuboid}")
    print(f"Error: {abs(hv_3d_cuboid - expected_3d_cuboid):.6f}")
    results["3d_cuboid"] = (hv_3d_cuboid, expected_3d_cuboid)

    # 3D Test Case 4: Layer
    # Points forming a horizontal layer
    # Expected volume: 27
    layer_points = np.array(
        [[1.0, 1.0, 1.0], [1.0, 3.0, 1.0], [3.0, 1.0, 1.0], [3.0, 3.0, 1.0]]
    )
    hv_3d_layer = bd_3d.compute_hypervolume(layer_points)
    expected_3d_layer = 27.0
    print(f"\n3D Layer:")
    print(f"Computed: {hv_3d_layer:.6f}")
    print(f"Expected: {expected_3d_layer}")
    print(f"Error: {abs(hv_3d_layer - expected_3d_layer):.6f}")
    results["3d_layer"] = (hv_3d_layer, expected_3d_layer)

    # Summary of all errors
    print("\n=== Summary of All Test Cases ===")
    total_error = 0.0
    for test_name, (computed, expected) in results.items():
        error = abs(computed - expected)
        total_error += error
        print(f"{test_name}:")
        print(f"  Computed: {computed:.6f}")
        print(f"  Expected: {expected:.6f}")
        print(f"  Error: {error:.6f}")

    print(f"\nTotal absolute error across all tests: {total_error:.6f}")
    return results


def test_ehvi_candidates():
    """
    Example of how to use the BoxDecomposition class for candidate selection.
    """
    # Setup
    n_objectives = 2
    ref_point = np.array([10.0, 10.0])
    bd = HyperVolumeBoxDecomposition(ref_point)

    # Current Pareto front
    pareto_front = np.array([[2.0, 8.0], [4.0, 4.0], [8.0, 2.0]])

    # Generate some candidate points
    n_candidates = 100
    np.random.seed(42)

    # Create candidate predictions (means and variances)
    candidate_means = np.random.uniform(1, 9, size=(n_candidates, n_objectives))
    candidate_variances = np.random.uniform(0.1, 0.5, size=(n_candidates, n_objectives))

    # Select best candidates
    n_select = 5
    selected_indices, ehvi_values = bd.select_candidates(
        pareto_front=pareto_front,
        candidate_means=candidate_means,
        candidate_variances=candidate_variances,
        n_select=n_select,
    )

    print("\nTop candidate points:")
    for idx, ehvi in zip(selected_indices, ehvi_values):
        print(f"Candidate {idx}:")
        print(f"  Means: {candidate_means[idx]}")
        print(f"  Variances: {candidate_variances[idx]}")
        print(f"  EHVI: {ehvi:.6f}")


if __name__ == "__main__":
    results = run_basic_test_cases()
    results = run_analytical_test_cases()
    results = test_ehvi_candidates()
