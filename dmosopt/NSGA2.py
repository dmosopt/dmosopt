# Nondominated Sorting Genetic Algorithm II (NSGA-II)
# A multi-objective optimization algorithm.

import numpy as np
from dmosopt.dda import dda_non_dominated_sort
from dmosopt.MOEA import (
    Struct,
    MOEA,
    crossover_sbx,
    mutation,
    tournament_selection,
    sortMO,
    remove_worst,
    remove_duplicates,
)
from typing import Any, Union, Dict, List, Tuple, Optional


class NSGA2(MOEA):
    def __init__(
        self,
        popsize: int,
        nInput: int,
        nOutput: int,
        feasibility_model: Optional[Any],
        distance_metric: Optional[Any],
        **kwargs,
    ):
        """Nondominated Sorting Genetic Algorithm II (NSGA-II)
        A multi-objective optimization algorithm."""

        super().__init__(
            name="NSGA2",
            popsize=popsize,
            nInput=nInput,
            nOutput=nOutput,
            **kwargs,
        )

        self.feasibility_model = feasibility_model
        self.distance_metric = distance_metric

        self.y_distance_metrics = None
        if distance_metric is not None:
            self.y_distance_metrics = []
            self.y_distance_metrics.append(distance_metric)
        self.x_distance_metrics = None
        if self.feasibility_model is not None:
            self.x_distance_metrics = [self.feasibility_model.rank]

        di_crossover = self.opt_params.di_crossover
        if np.isscalar(di_crossover):
            self.opt_params.di_crossover = np.asarray([di_crossover] * nInput)

        di_mutation = self.opt_params.di_mutation
        if np.isscalar(di_mutation):
            self.opt_params.di_mutation = np.asarray([di_mutation] * nInput)
        mutation_rate = self.opt_params.mutation_rate
        if mutation_rate is None:
            self.opt_params.mutation_rate = 1.0 / float(nInput)

        self.opt_params.poolsize = int(round(popsize / 2.0))

    @property
    def default_parameters(self) -> Dict[str, Any]:
        """Returns default parameters of NSGA-II strategy."""
        params = {
            "crossover_prob": 0.9,
            "mutation_prob": 0.1,
            "mutation_rate": None,
            "nchildren": 1,
            "di_crossover": 1.0,
            "di_mutation": 20.0,
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
            x_distance_metrics=self.x_distance_metrics,
            y_distance_metrics=self.y_distance_metrics,
        )
        population_parm = x[: self.popsize]
        population_obj = y[: self.popsize]
        rank = rank[: self.popsize]

        state = Struct(
            bounds=bounds,
            population_parm=population_parm,
            population_obj=population_obj,
            rank=rank,
        )

        return state

    def generate_strategy(self, **params):
        popsize = self.popsize
        poolsize = self.opt_params.poolsize
        crossover_prob = self.opt_params.crossover_prob
        mutation_prob = self.opt_params.mutation_prob
        mutation_rate = self.opt_params.mutation_rate
        nchildren = self.opt_params.nchildren
        di_crossover = self.opt_params.di_crossover
        di_mutation = self.opt_params.di_mutation

        local_random = self.local_random
        xlb = self.state.bounds[:, 0]
        xub = self.state.bounds[:, 1]

        population_parm = self.state.population_parm
        population_obj = self.state.population_obj
        rank = self.state.rank

        pool_idxs = tournament_selection(local_random, popsize, poolsize, rank)
        pool = population_parm[pool_idxs, :]
        count = 0
        xs_gen = []
        while count < popsize - 1:
            if local_random.random() < crossover_prob:
                parentidx = local_random.choice(poolsize, 2, replace=False)
                parent1 = pool[parentidx[0], :]
                parent2 = pool[parentidx[1], :]
                children1, children2 = crossover_sbx(
                    local_random,
                    parent1,
                    parent2,
                    di_crossover,
                    xlb,
                    xub,
                    nchildren=nchildren,
                )
                child1 = children1[0]
                child2 = children2[0]
                xs_gen.extend([child1, child2])
                count += 2
            if local_random.random() < mutation_prob:
                parentidx = local_random.integers(low=0, high=poolsize)
                parent = pool[parentidx, :]
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
                xs_gen.append(child)
                count += 1
        x_gen = np.vstack(xs_gen)
        # gen_indexes = np.ones((x_gen.shape[0],), dtype=np.uint32) * self.state.gen_index

        return x_gen, {}

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
            x_distance_metrics=self.x_distance_metrics,
            y_distance_metrics=self.y_distance_metrics,
        )

        self.state.population_parm[:] = population_parm
        self.state.population_obj[:] = population_obj
        self.state.rank[:] = rank

    def get_population_strategy(self):
        pop_x = self.state.population_parm.copy()
        pop_y = self.state.population_obj.copy()

        return pop_x, pop_y
