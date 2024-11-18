# Trust Region Search, multi-objective local optimization algorithm.
import sys
import gc, itertools, math
from functools import partial
import numpy as np
from numpy.random import default_rng
from dmosopt.datatypes import OptHistory
from dmosopt.dda import dda_non_dominated_sort
from dmosopt.MOEA import (
    Struct,
    MOEA,
    orderMO,
    remove_worst,
    remove_duplicates,
)
from dmosopt.sampling import sobol
from dmosopt.indicators import HypervolumeImprovement
from typing import Any, Union, Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TrState:
    dim: int
    is_constrained: bool = False
    length: float = 0.08
    length_init: float = 0.08
    length_min: float = 0.0001
    length_max: float = 1.0
    success_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_tolerance: int = 0.6
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
        model: Optional[Any],
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

        self.model = model
        self.x_distance_metrics = None
        if self.model.feasibility is not None:
            self.x_distance_metrics = [self.model.feasibility.rank]
        self.indicator = HypervolumeImprovement

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
        order, rank, _ = orderMO(x, y, x_distance_metrics=self.x_distance_metrics)
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

        population_parm, population_obj = remove_duplicates(
            self.state.population_parm, self.state.population_obj
        )

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
        pert = sobol(x_centers.shape[0], self.nInput, local_random=local_random)
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

        if X_cand.shape[0] < popsize:
            sample = sobol(
                popsize - X_cand.shape[0], self.nInput, local_random=local_random
            )
            sample = xlb + (xub - xlb) * sample
            X_cand = np.vstack((X_cand, sample))

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

        order, rank, _ = orderMO(
            candidates_x, candidates_y, x_distance_metrics=self.x_distance_metrics
        )
        order_inv = np.argsort(order)

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
            front_r = order_inv[np.argwhere(rank == r).ravel()]
            if chosen_count + len(front_r) <= popsize and not full:
                chosen[front_r] = True
                chosen_count += len(front_r)
                assert np.count_nonzero(chosen) == chosen_count
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
            indicator = self.indicator(ref_point=ref, nds=True)

            assert len(mid_front) > 0
            if chosen_count > 0:
                selected_indices = indicator.do(
                    candidates_y[chosen],
                    candidates_y[mid_front],
                    np.ones_like(candidates_y[mid_front, :]),
                    k,
                )
            else:
                selected_indices = np.arange(k)

            assert len(selected_indices) == k
            chosen[mid_front[selected_indices]] = True

            not_chosen_mask = np.ones(len(mid_front), np.bool)
            not_chosen_mask[selected_indices] = False
            not_chosen[mid_front[not_chosen_mask]] = True

        return chosen, not_chosen

    def update_state(self, X_next, Y_next, is_offspring):

        state = self.state.tr

        if state.restart:
            self.restart_state()

        chosen, not_chosen = self.select_candidates(X_next, Y_next)

        state.success_counter += np.count_nonzero(np.logical_and(is_offspring, chosen))
        success_frac = state.success_counter / self.popsize
        print(
            f"update_state: before state.length = {state.length} "
            f" success_counter = {state.success_counter}"
            f" success_frac = {success_frac}"
        )
        if success_frac >= state.success_tolerance:  # Expand trust region
            state.length = min((1.0 + success_frac) * state.length, state.length_max)
            state.success_counter = 0
        elif success_frac <= state.failure_tolerance:  # Shrink trust region
            state.length /= 2.0
            state.success_counter = 0
        if state.length < state.length_min:
            state.restart = True

        print(f"state.length = {state.length}")
        sys.stdout.flush()

        return X_next[chosen], Y_next[chosen]

    def restart_state(self):
        if self.state.tr.length_init > 4 * self.state.tr.length_min:
            self.state.tr.length_init /= 2.0
        self.state.tr.failure_counter = 0
        self.state.tr.success_counter = 0
        self.state.tr.length = self.state.tr.length_init
        self.state.tr.Y_best = np.asarray([np.inf] * self.state.tr.dim).reshape((1, -1))
        self.state.tr.restart = False
