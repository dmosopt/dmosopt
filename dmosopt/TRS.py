# Trust Region Search, multi-objective local optimization algorithm.

import gc, itertools, math
import numpy as np
from numpy.random import default_rng
from dmosopt.datatypes import OptHistory
from dmosopt.MOEA import (
    Struct,
    MOEA,
    sortMO,
    remove_worst,
    remove_duplicates,
)
from dmosopt.sampling import sobol
from typing import Any, Union, Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TrbState:
    dim: int
    is_constrained: bool = False
    length: float = 0.8
    length_init: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 10
    Y_best: np.ndarray = [np.inf]  # Goal is minimization
    constraint_violation = float("inf")
    restart: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(max([4.0, float(self.dim)]))
        self.Y_best = np.asarray([np.inf] * self.dim)


def update_state(state, Y_next):
    if not state.is_constrained:
        better_than_current = np.min(Y_next) < np.min(self.Y_best) - 1e-3 * math.fabs(
            np.min(self.Y_best)
        )
        state.Y_best = np.max(state.Y_best, Y_next)
    else:
        raise NotImplemented

    if better_than_current:
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    if state.length < state.length_min:
        state.restart = True
    return state


def restart_state(state):
    state.failure_counter = 0
    state.success_counter = 0
    state.length = state.length_init
    state.Y_best = np.asarray([np.inf] * state.dim)
    state.restart = False


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
        x, y, rank, _ = sortMO(
            x,
            y,
            self.nInput,
            self.nOutput,
        )
        population_parm = x[: self.popsize]
        population_obj = y[: self.popsize]
        rank = rank[: self.popsize]

        trb = TrbState(dim=nInput)

        state = Struct(
            bounds=bounds,
            population_parm=population_parm,
            population_obj=population_obj,
            rank=rank,
            trb=trb,
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
        ranks = self.state.ranks

        parentidxs = local_random.integers(low=0, high=popsize, size=popsize)

        # Create the trust region boundaries
        x_centers = population_param[None, :]
        weights = xub - xlb
        weights = weights / np.mean(weights)  # This will make the next line more stable
        weights = weights / np.prod(
            np.power(weights, 1.0 / len(weights))
        )  # We now have weights.prod() = 1
        tr_lb = np.clip(x_centers - weights * self.state.trb.length / 2.0, 0.0, 1.0)
        tr_ub = np.clip(x_centers + weights * length / 2.0, 0.0, 1.0)

        # Draw a Sobolev sequence in [lb, ub]
        pert = sobol(popsize, self.nInput, local_random=local_random)
        pert = tr_lb + (tr_ub - tr_lb) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / self.dim, 1.0)
        mask = local_random.rand(popsize, self.dim) <= prob_perturb
        ind = np.where(np.sum(mask, axis=1) == 0)[0]
        mask[ind, local_random.randint(0, self.dim - 1, size=len(ind))] = 1

        # Create candidate points
        X_cand = x_center.copy()
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

        self.state.trb = update_state(self.state.trb, y_gen)
        if self.state.trb.restart:
            self.state.trb = restart_state(self.state.trb)

        population_parm = np.vstack((population_parm, x_gen))
        population_obj = np.vstack((population_obj, y_gen))
        population_parm, population_obj = remove_duplicates(
            population_parm, population_obj
        )
        population_parm, population_obj, rank = remove_worst(
            population_parm,
            population_obj,
            popsize,
            nInput,
            nOutput,
        )

        self.state.population_parm[:] = population_parm
        self.state.population_obj[:] = population_obj
        self.state.rank[:] = rank

    def get_population_strategy(self):
        pop_x = self.state.population_parm.copy()
        pop_y = self.state.population_obj.copy()

        return pop_x, pop_y
