#
# Indicator routines from PyMOO:
# https://github.com/anyoptimization/pymoo
#
from abc import abstractmethod

import numpy as np
from dmosopt.normalization import PreNormalization
from dmosopt.hv import HyperVolume as _HyperVolume
from dmosopt.dda import dda_non_dominated_sort


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
        **kwargs
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
            assert (
                self.ideal is not None and self.nadir is not None
            ), "If norm_by_dist is enabled ideal and nadir must be set!"
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
        **kwargs
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
        assert (
            self.ref_point is not None
        ), "For Hypervolume a reference point needs to be provided!"

    def _do(self, F):
        if self.nds:
            rank = dda_non_dominated_sort(F)
            non_dom = np.argwhere(rank == 0).ravel()
            F = np.copy(F[non_dom, :])

        hv = _HyperVolume(self.ref_point)
        val = hv.compute(F)

        return val
