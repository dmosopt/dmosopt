#
# Indicator routines from PyMOO:
# https://github.com/anyoptimization/pymoo
#
from abc import abstractmethod
import numpy as np
from dmosopt.normalization import PreNormalization
from dmosopt.hv import HyperVolumeBoxDecomposition as _HyperVolume
from dmosopt.dda import dda_ens


def crowding_distance_metric(Y):
    """Crowding distance metric.
    Y is the output data matrix
    [n,d] = size(Y)
    n: number of points
    d: number of dimensions
    """
    n, d = Y.shape
    lb = np.min(Y, axis=0, keepdims=True)
    ub = np.max(Y, axis=0, keepdims=True)

    if n == 1:
        D = np.array([1.0])
    else:
        ub_minus_lb = ub - lb
        ub_minus_lb[ub_minus_lb == 0.0] = 1.0

        U = (Y - lb) / ub_minus_lb

        D = np.zeros(n)
        DS = np.zeros((n, d))

        idx = U.argsort(axis=0)
        US = np.zeros((n, d))
        for i in range(d):
            US[:, i] = U[idx[:, i], i]

        DS[0, :] = 1.0
        DS[n - 1, :] = 1.0

        for i in range(1, n - 1):
            for j in range(d):
                DS[i, j] = US[i + 1, j] - US[i - 1, j]

        for i in range(n):
            for j in range(d):
                D[idx[i, j]] += DS[i, j]
        D[np.isnan(D)] = 0.0

    return D


def euclidean_distance_metric(Y):
    """Row-wise euclidean distance."""
    n, d = Y.shape
    lb = np.min(Y, axis=0)
    ub = np.max(Y, axis=0)
    ub_minus_lb = ub - lb
    ub_minus_lb[ub_minus_lb == 0.0] = 1.0
    U = (Y - lb) / ub_minus_lb
    return np.sqrt(np.sum(U**2, axis=1))


def euclidean_distance(a, b, norm=None):
    return np.sqrt((((a - b) / norm) ** 2).sum(axis=1))


def vectorized_cdist(
    A, B, func_dist=euclidean_distance, fill_diag_with_inf=False, **kwargs
) -> object:
    assert A.ndim <= 2 and B.ndim <= 2

    A, only_row = at_least_2d_array(A, extend_as="row", return_if_reshaped=True)
    B, only_column = at_least_2d_array(B, extend_as="row", return_if_reshaped=True)

    u = np.repeat(A, B.shape[0], axis=0)
    v = np.tile(B, (A.shape[0], 1))

    D = func_dist(u, v, **kwargs)
    M = np.reshape(D, (A.shape[0], B.shape[0]))

    if fill_diag_with_inf:
        np.fill_diagonal(M, np.inf)

    if only_row and only_column:
        M = M[0, 0]
    elif only_row:
        M = M[0]
    elif only_column:
        M = M[:, [0]]

    return M


def at_least_2d_array(x, extend_as="row", return_if_reshaped=False):
    if x is None:
        return x
    elif not isinstance(x, np.ndarray):
        x = np.array([x])

    has_been_reshaped = False

    if x.ndim == 1:
        if extend_as == "row":
            x = x[None, :]
        elif extend_as == "column":
            x = x[:, None]

        has_been_reshaped = True

    if return_if_reshaped:
        return x, has_been_reshaped
    else:
        return x


def derive_ideal_and_nadir_from_pf(pf, ideal=None, nadir=None):
    # try to derive ideal and nadir if not already set and pf provided
    if pf is not None:
        if ideal is None:
            ideal = np.min(pf, axis=0)
        if nadir is None:
            nadir = np.max(pf, axis=0)

    return ideal, nadir


class SlidingWindow(list):
    def __init__(self, size=None) -> None:
        super().__init__()
        self.size = size

    def append(self, entry):
        super().append(entry)

        if self.size is not None:
            while len(self) > self.size:
                self.pop(0)

    def is_full(self):
        return self.size == len(self)


class Indicator(PreNormalization):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # what should an indicator return if no solutions are provided is defined here
        self.default_if_empty = 0.0

    def do(self, F, *args, **kwargs):
        # if it is a 1d array
        if F.ndim == 1:
            F = F[None, :]

        # if no points have been provided just return the default
        if len(F) == 0:
            return self.default_if_empty

        # do the normalization - will only be done if zero_to_one is enabled
        F = self.normalization.forward(F)

        return self._do(F, *args, **kwargs)

    @abstractmethod
    def _do(self, F, *args, **kwargs):
        return


class DistanceIndicator(Indicator):
    def __init__(
        self,
        pf,
        dist_func,
        axis,
        zero_to_one=False,
        ideal=None,
        nadir=None,
        norm_by_dist=False,
        **kwargs,
    ):
        # the pareto front if necessary to calculate the indicator
        pf = at_least_2d_array(pf, extend_as="row")
        ideal, nadir = derive_ideal_and_nadir_from_pf(pf, ideal=ideal, nadir=nadir)

        super().__init__(zero_to_one=zero_to_one, ideal=ideal, nadir=nadir, **kwargs)
        self.dist_func = dist_func
        self.axis = axis
        self.norm_by_dist = norm_by_dist
        self.pf = self.normalization.forward(pf)

    def _do(self, F):
        # a factor to normalize the distances by (1.0 disables that by default)
        norm = 1.0

        # if zero_to_one is disabled this can be used to normalize the distance calculation itself
        if self.norm_by_dist:
            assert self.ideal is not None and self.nadir is not None, (
                "If norm_by_dist is enabled ideal and nadir must be set!"
            )
            norm = self.nadir - self.ideal

        D = vectorized_cdist(self.pf, F, func_dist=self.dist_func, norm=norm)
        return np.mean(np.min(D, axis=self.axis))


class IGD(DistanceIndicator):
    def __init__(self, pf, **kwargs):
        super().__init__(pf, euclidean_distance, 1, **kwargs)


class Hypervolume(Indicator):
    def __init__(
        self,
        ref_point=None,
        pf=None,
        nds=False,
        norm_ref_point=True,
        ideal=None,
        nadir=None,
        **kwargs,
    ):
        pf = at_least_2d_array(pf, extend_as="row")
        ideal, nadir = derive_ideal_and_nadir_from_pf(pf, ideal=ideal, nadir=nadir)

        super().__init__(ideal=ideal, nadir=nadir, **kwargs)

        # whether the input should be checked for domination or not
        self.nds = nds

        # the reference point that shall be used - either derived from pf or provided
        ref_point = ref_point
        if ref_point is None:
            if pf is not None:
                ref_point = pf.max(axis=0)

        # we also have to normalize the reference point to have the same scales
        if norm_ref_point:
            ref_point = self.normalization.forward(ref_point)

        self.ref_point = ref_point
        assert self.ref_point is not None, (
            "For Hypervolume a reference point needs to be provided!"
        )

    def _do(self, F):
        if self.nds:
            rank = dda_ens(F)
            non_dom = np.argwhere(rank == 0).ravel()
            F = np.copy(F[non_dom, :])

        hv = _HyperVolume(self.ref_point)
        val = hv.compute_hypervolume(F)

        return val


class HypervolumeImprovement(Indicator):
    def __init__(
        self,
        ref_point=None,
        pf=None,
        nds=False,
        norm_ref_point=True,
        ideal=None,
        nadir=None,
        **kwargs,
    ):
        pf = at_least_2d_array(pf, extend_as="row")
        ideal, nadir = derive_ideal_and_nadir_from_pf(pf, ideal=ideal, nadir=nadir)

        super().__init__(ideal=ideal, nadir=nadir, **kwargs)

        self.default_if_empty = []

        # whether the input should be checked for domination or not
        self.nds = nds

        # the reference point that shall be used - either derived from pf or provided
        ref_point = ref_point
        if ref_point is None:
            if pf is not None:
                ref_point = pf.max(axis=0)

        # we also have to normalize the reference point to have the same scales
        if norm_ref_point:
            ref_point = self.normalization.forward(ref_point)

        self.ref_point = ref_point
        assert self.ref_point is not None, (
            "For Hypervolume a reference point needs to be provided!"
        )

    def _do(self, F, means, variances, k):
        assert k > 0
        assert len(F) > 0

        if self.nds:
            rank = dda_ens(F)
            non_dom = np.argwhere(rank == 0).ravel()
            if len(non_dom) > 0:
                F = np.copy(F[non_dom, :])

        assert len(F) > 0

        hv = _HyperVolume(self.ref_point)
        selection, _ = np.asarray(
            hv.select_candidates(F, means, variances, k), dtype=int
        )

        assert len(selection) > 0
        return selection


class PopulationDiversity(Indicator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _do(self, F, Y):
        # Calculate diversity metric
        front_0 = np.argwhere(F.flat == 0)

        diversity = len(front_0) / len(F[0])

        D = crowding_distance_metric(Y)

        # Calculate crowding distance spread in first front
        if len(front_0) > 1:
            cd_values = D[front_0.flat]
            cd_spread = np.std(cd_values) / np.mean(cd_values)
        else:
            cd_spread = 0

        return diversity, cd_spread
