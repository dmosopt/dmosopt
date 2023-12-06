# Trust Region Search, multi-objective local optimization algorithm.

import gc, itertools, math
from functools import partial
import numpy as np
from numpy.random import default_rng
from dmosopt.datatypes import OptHistory
from dmosopt.dda import dda_non_dominated_sort
from dmosopt.MOEA import (
    Struct,
    MOEA,
    remove_worst,
    remove_duplicates,
)
from dmosopt.sampling import sobol
from dmosopt.indicators import Hypervolume
from typing import Any, Union, Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TrState:
    dim: int
    is_constrained: bool = False
    length: float = 0.08
    length_init: float = 0.08
    length_min: float = 0.5**7
    length_max: float = 1.6
    success_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_tolerance: int = 0.2
    Y_best: np.ndarray = np.asarray([np.inf])  # Goal is minimization
    constraint_violation = float("inf")
    restart: bool = False

    def __post_init__(self):
        self.failure_tolerance = min(1 / self.dim, self.success_tolerance / 2.0)
        self.Y_best = np.asarray([np.inf] * self.dim).reshape((1, -1))


class TRS(MOEA):
    def __init__(
        self,
        popsize: int,
        nInput: int,
        nOutput: int,
        feasibility_model: Optional[Any],
        **kwargs,
    ):
        """Trust Region Search"""
        super().__init__(
            name="TRS",
            popsize=popsize,
            nInput=nInput,
            nOutput=nOutput,
            **kwargs,
        )

        self.feasibility_model = feasibility_model
        self.x_distance_metrics = None
        if self.feasibility_model is not None:
            self.x_distance_metrics = [self.feasibility_model.rank]
        self.indicator = Hypervolume

    @property
    def default_parameters(self) -> Dict[str, Any]:
        """Returns default parameters of TRS strategy."""
        params = {
            "nchildren": 1,
        }

        return params

    def initialize_state(
        self,
        x: np.ndarray,
        y: np.ndarray,
        bounds: np.ndarray,
        local_random: Optional[np.random.Generator] = None,
        **params,
    ):
        order, rank = sortMO(x, y, self.x_distance_metrics)
        population_parm = x[order][: self.popsize]
        population_obj = y[order][: self.popsize]
        rank = rank[: self.popsize]

        tr = TrState(dim=self.nInput)

        state = Struct(
            bounds=bounds,
            population_parm=population_parm,
            population_obj=population_obj,
            rank=rank,
            tr=tr,
        )

        return state

    def generate_strategy(self, **params):
        popsize = self.popsize
        nchildren = self.opt_params.nchildren

        local_random = self.local_random
        xlb = self.state.bounds[:, 0]
        xub = self.state.bounds[:, 1]

        population_parm = self.state.population_parm
        population_obj = self.state.population_obj

        parentidxs = local_random.integers(low=0, high=popsize, size=popsize)

        # Create the trust region boundaries
        x_centers = population_parm
        weights = xub - xlb
        weights = weights / np.mean(weights)  # This will make the next line more stable
        weights = weights / np.prod(
            np.power(weights, 1.0 / len(weights))
        )  # We now have weights.prod() = 1
        tr_lb = np.clip(x_centers - weights * self.state.tr.length / 2.0, xlb, xub)
        tr_ub = np.clip(x_centers + weights * self.state.tr.length / 2.0, xlb, xub)

        # Draw a Sobolev sequence in [lb, ub]
        pert = sobol(popsize, self.nInput, local_random=local_random)
        pert = tr_lb + (tr_ub - tr_lb) * pert

        # Creates a perturbation mask; perturbing fewer dimensions at
        # a time improves performance for high-dimensional problems
        # (R. G. Regis and C. A. Shoemaker. Combining radial basis
        # function surrogates and dynamic coordinate search in
        # high-dimensional expensive black-box optimization.
        # Engineering Optimization, 45(5):529–555, 2013)

        prob_perturb = min(20.0 / self.state.tr.dim, 1.0)
        perturb_selection = local_random.random((self.state.tr.dim,)) <= prob_perturb
        ind = np.nonzero(perturb_selection)[0]
        mask = np.zeros((self.popsize, self.state.tr.dim), dtype=int)
        mask[ind, local_random.integers(0, self.state.tr.dim - 1, size=len(ind))] = 1

        # Create candidate points
        X_cand = x_centers.copy()
        X_cand[mask] = pert[mask]

        return X_cand, {}

    def update_strategy(
        self,
        x_gen: np.ndarray,
        y_gen: np.ndarray,
        state: Dict[Any, Any],
        **params,
    ):
        population_parm = self.state.population_parm
        population_obj = self.state.population_obj
        rank = self.state.rank

        popsize = self.popsize
        nInput = self.nInput
        nOutput = self.nOutput

        candidates_x = np.vstack((x_gen, population_parm))
        candidates_y = np.vstack((y_gen, population_obj))

        P = population_parm.shape[0]
        C = x_gen.shape[0]

        candidates_offspring = np.concatenate(
            (
                np.asarray([True] * C, dtype=bool),
                np.asarray([False] * P, dtype=bool),
            )
        )

        population_parm, population_obj = self.update_state(
            candidates_x, candidates_y, candidates_offspring
        )
        if self.state.tr.restart:
            self.restart_state()

        self.state.population_parm[:] = population_parm
        self.state.population_obj[:] = population_obj
        self.state.rank[:] = rank

    def get_population_strategy(self):
        pop_x = self.state.population_parm.copy()
        pop_y = self.state.population_obj.copy()

        return pop_x, pop_y

    def select_candidates(self, candidates_x, candidates_y):

        popsize = self.popsize

        candidates_inds = np.asarray(range(candidates_x.shape[0]), dtype=np.int_)

        if candidates_x.shape[0] <= popsize:
            return np.ones_like(candidates_inds, dtype=bool_), np.zeros_like(
                candidates_inds, dtype=bool
            )

        order, rank = sortMO(candidates_x, candidates_y, self.x_distance_metrics)

        chosen = np.zeros_like(candidates_inds, dtype=bool)
        not_chosen = np.zeros_like(candidates_inds, dtype=bool)
        mid_front = None

        # Fill the next population (chosen) with the fronts until there is not enough space
        # When an entire front does not fit in the space left we rely on the hypervolume
        # for this front
        # The remaining fronts are explicitly not chosen
        full = False
        rmax = int(np.max(rank))
        chosen_count = 0
        for r in range(rmax + 1):
            front_r = np.argwhere(rank == r).ravel()
            if chosen_count + len(front_r) <= popsize and not full:
                chosen[front_r] = True
                chosen_count += len(front_r)
            elif mid_front is None and chosen_count < popsize:
                mid_front = front_r.copy()
                # With this front, we selected enough individuals
                full = True
            else:
                not_chosen[front_r] = True

        # Separate the mid front to accept only k individuals
        k = popsize - chosen_count
        if k > 0:
            # reference point is chosen in the complete population
            # as the worst in each dimension +1
            ref = np.max(candidates_y, axis=0) + 1
            indicator = self.indicator(ref_point=ref)

            def contribution(front, i):
                # The contribution of point p_i in point set P
                # is the hypervolume of P without p_i
                return indicator.do(
                    np.concatenate(
                        (candidates_y[front[:i]], candidates_y[front[i + 1 :]])
                    )
                )

            contrib_values = np.fromiter(
                map(partial(contribution, mid_front), range(len(mid_front))),
                dtype=np.float32,
            )
            contrib_order = np.argsort(contrib_values)

            chosen[mid_front[contrib_order[:k]]] = True
            not_chosen[mid_front[contrib_order[k:]]] = True

        return chosen, not_chosen

    def update_state(self, X_next, Y_next, is_offspring):

        state = self.state.tr

        chosen, not_chosen = self.select_candidates(X_next, Y_next)

        state.success_counter += np.count_nonzero(is_offspring[chosen])

        if (
            state.success_counter / self.popsize >= state.success_tolerance
        ):  # Expand trust region
            state.length = min(2.0 * state.length, state.length_max)
            state.success_counter = 0
        elif (
            state.success_counter / self.popsize < state.failure_tolerance
        ):  # Shrink trust region
            state.length /= 2.0
            state.success_counter = 0
        if state.length < state.length_min:
            state.restart = True

        return X_next[chosen], Y_next[chosen]

    def restart_state(self):
        self.state.tr.failure_counter = 0
        self.state.tr.success_counter = 0
        self.state.tr.length = self.state.tr.length_init
        self.state.tr.Y_best = np.asarray([np.inf] * state.dim)
        self.state.tr.restart = False


def sortMO(
    x,
    y,
    x_distance_metrics=None,
):
    """Non-dominated sort for multi-objective optimization
    x: input parameter matrix
    y: output objectives matrix
    """

    x_distance_functions = []
    if x_distance_metrics is not None:
        for distance_metric in x_distance_metrics:
            if callable(distance_metric):
                x_distance_functions.append(distance_metric)
            else:
                raise RuntimeError(f"sortMO: unknown distance metric {distance_metric}")

    rank = dda_non_dominated_sort(y)

    x_dists = list([np.zeros_like(rank) for _ in x_distance_functions])
    rmax = int(rank.max())
    if len(x_dists) > 0:
        for front in range(rmax + 1):
            rankidx = rank == front
            for i, x_distance_function in enumerate(x_distance_functions):
                D = x_distance_function(x[rankidx, :])
                x_dists[i][rankidx] = D

    perm = np.lexsort((list([-dist for dist in x_dists]) + [rank]))

    return perm, rank
