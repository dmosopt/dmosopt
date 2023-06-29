# Trust Region Search, multi-objective local optimization algorithm.

import numpy as np
from dmosopt.dda import dda_non_dominated_sort
from dmosopt.MOEA import (
    Struct,
    MOEA,
    mutation,
    sortMO,
    crowding_distance,
    remove_worst,
    remove_duplicates,
)
from typing import Any, Union, Dict, List, Tuple, Optional


class MOTRS(MOEA):
    def __init__(
        self,
        popsize: int,
        nInput: int,
        nOutput: int,
        lengthscales: np.ndarray,
        feasibility_model: Optional[Any],
        **kwargs,
    ):
        """
        MOTRS: Multi-objective trust region search.
        """

        super().__init__(
            name="MOTRS",
            popsize=popsize,
            nInput=nInput,
            nOutput=nOutput,
            lengthscales=lengthscales,
            **kwargs,
        )

        self.feasibility_model = feasibility_model

        self.x_distance_metrics = None
        if self.feasibility_model is not None:
            x_distance_metrics = [self.feasibility_model.rank]
        
            
    @property
    def default_parameters(self) -> Dict[str, Any]:
        """Returns default parameters of MOTRS strategy."""
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

        nInput = self.nInput
        nOutput = self.nOutput
        popsize = self.popsize

        xlb = bounds[:, 0]
        xub = bounds[:, 1]

        weights = self.opt_params.lengthscales

        ## TODO: check normalization
        weights = weights / np.mean(weights)
        weights = weights / np.prod(weights**(1.0 / len(weights)))
        
        population_parm = np.zeros((popsize, nInput), dtype=np.float32)
        population_obj = np.zeros((popsize, nOutput), dtype=np.float32)

        xs, ys, ranks, _ = sortMO(
            x,
            y,
            nInput,
            nOutput,
            x_distance_metrics=self.x_distance_metrics,
        )
        population_parm = xs[:popsize]
        population_obj = ys[:popsize]

        state = Struct(
            bounds=bounds,
            population_parm=population_parm,
            population_obj=population_obj,
            ranks=ranks[:popsize],
            weights=weights,
        )

        return state

    def generate_strategy(self, **params):

        popsize = self.popsize
        mutation_rate = self.opt_params.mutation_rate
        nchildren = self.opt_params.nchildren

        local_random = self.local_random
        xlb = self.state.bounds[:, 0]
        xub = self.state.bounds[:, 1]

        population_parm = self.state.population_parm
        population_obj = self.state.population_obj
        ranks = self.state.ranks

        xs_gens = []
        count = 0
        while count < popsize:
            parentidx = local_random.integers(low=0, high=popsize, size=(popsize, 1))
            parent = population_parm[parentidx, :]
            children = mutation(
                local_random,
                parent,
                di_mutation,
                xlb,
                xub,
                mutation_rate=mutation_rate,
                nchildren=nchildren,
            )
            child = children[0]
            xs_gens.append(child)
            count += 1

        x_gen = np.vstack([np.vstack(x) for x in xs_gens]).astype(np.float32)
        return x_gen, {}

    def update_strategy(
        self,
        x_gen: np.ndarray,
        y_gen: np.ndarray,
        state: Dict[Any, Any],
        **params,
    ):

        local_random = self.local_random

        population_parm = self.state.population_parm
        population_obj = self.state.population_obj
        ranks = self.state.ranks
        velocity = self.state.velocity

        xlb = self.state.bounds[:, 0]
        xub = self.state.bounds[:, 1]

        swarm_size = self.opt_params.swarm_size
        popsize = self.popsize
        nInput = self.nInput
        nOutput = self.nOutput

        pop_slices = self.pop_slices

        for sl in pop_slices:
            D = crowding_distance(y_gen[sl])
            velocity[sl] = velocity_vector(
                local_random, population_parm[sl], velocity[sl], x_gen[sl], D, xlb, xub
            )

        for p, sl in enumerate(pop_slices):
            population_parm_p = np.vstack((population_parm[sl], x_gen[sl]))
            population_obj_p = np.vstack((population_obj[sl], y_gen[sl]))
            population_parm_p, population_obj_p = remove_duplicates(
                population_parm_p, population_obj_p
            )
            population_parm[sl], population_obj[sl], ranks[p] = remove_worst(
                population_parm_p,
                population_obj_p,
                popsize,
                nInput,
                nOutput,
                x_distance_metrics=self.x_distance_metrics,
                y_distance_metrics=self.y_distance_metrics,
            )

    def get_population_strategy(self):

        popsize = self.popsize
        nInput = self.nInput
        nOutput = self.nOutput

        pop_parm = self.state.population_parm.copy()
        pop_obj = self.state.population_obj.copy()

        pop_parm, pop_obj = remove_duplicates(pop_parm, pop_obj)
        bestx, besty, _ = remove_worst(
            pop_parm,
            pop_obj,
            popsize,
            nInput,
            nOutput,
            x_distance_metrics=self.x_distance_metrics,
            y_distance_metrics=self.y_distance_metrics,
        )

        return pop_parm, pop_obj


