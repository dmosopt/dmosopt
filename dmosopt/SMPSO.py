## Speed-constrained multiobjective particle swarm optimization
## SMPSO: A New PSO Metaheuristic for Multi-objective Optimization
##

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


class SMPSO(MOEA):
    def __init__(
        self,
        popsize: int,
        nInput: int,
        nOutput: int,
        feasibility_model: Optional[Any],
        distance_metric: Optional[Any],
        **kwargs,
    ):
        """
        SMPSO: Speed-constrained multiobjective particle swarm optimization.
        """

        swarm_size = kwargs.get("swarm_size", self.default_parameters["swarm_size"])

        kwargs["initial_size"] = popsize * swarm_size
        super().__init__(
            name="SMPSO",
            popsize=popsize,
            nInput=nInput,
            nOutput=nOutput,
            **kwargs,
        )

        self.pop_slices = list(
            [range(p * popsize, (p + 1) * popsize) for p in range(swarm_size)]
        )

        self.feasibility_model = feasibility_model
        self.distance_metric = distance_metric

        self.y_distance_metrics = None
        if distance_metric is not None:
            self.y_distance_metrics = []
            self.y_distance_metrics.append(distance_metric)
        self.x_distance_metrics = None
        if self.feasibility_model is not None:
            x_distance_metrics = [self.feasibility_model.rank]

        di_mutation = self.opt_params.di_mutation
        if np.isscalar(di_mutation):
            self.opt_params.di_mutation = np.asarray([di_mutation] * nInput)
        mutation_rate = self.opt_params.mutation_rate
        if mutation_rate is None:
            self.opt_params.mutation_rate = 1.0 / float(nInput)

    @property
    def default_parameters(self) -> Dict[str, Any]:
        """Returns default parameters of SMPSO strategy."""
        params = {
            "mutation_rate": None,
            "nchildren": 1,
            "swarm_size": 5,
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
        nInput = self.nInput
        nOutput = self.nOutput
        popsize = self.popsize
        swarm_size = self.opt_params.swarm_size
        pop_slices = self.pop_slices

        xlb = bounds[:, 0]
        xub = bounds[:, 1]

        xs = []
        ys = []
        for sl in pop_slices:
            x_s = x[sl].astype(np.float32)
            y_s = y[sl].astype(np.float32)
            xs.append(x_s)
            ys.append(y_s)

        population_parm = np.zeros((swarm_size * popsize, nInput), dtype=np.float32)
        population_obj = np.zeros((swarm_size * popsize, nOutput), dtype=np.float32)

        velocity = (
            local_random.uniform(size=(swarm_size * popsize, nInput)) * (xub - xlb)
            + xlb
        )

        ranks = []
        for p, sl in enumerate(pop_slices):
            xs[p], ys[p], rank_p, _ = sortMO(
                xs[p],
                ys[p],
                nInput,
                nOutput,
                x_distance_metrics=self.x_distance_metrics,
                y_distance_metrics=self.y_distance_metrics,
            )
            population_parm[sl] = xs[p][:popsize]
            population_obj[sl] = ys[p][:popsize]
            ranks.append(rank_p)

        state = Struct(
            bounds=bounds,
            population_parm=population_parm,
            population_obj=population_obj,
            ranks=ranks,
            velocity=velocity,
        )

        return state

    def generate_strategy(self, **params):
        popsize = self.popsize
        swarm_size = self.opt_params.swarm_size
        mutation_rate = self.opt_params.mutation_rate
        nchildren = self.opt_params.nchildren
        di_mutation = self.opt_params.di_mutation

        local_random = self.local_random
        xlb = self.state.bounds[:, 0]
        xub = self.state.bounds[:, 1]

        population_parm = self.state.population_parm
        population_obj = self.state.population_obj
        ranks = self.state.ranks
        velocity = self.state.velocity

        pop_slices = self.pop_slices
        xs_gens = [[] for _ in range(swarm_size)]
        count = 0

        for p, sl in enumerate(pop_slices):
            xs_updated = update_position(population_parm[sl], velocity[sl], xlb, xub)
            xs_gens[p].append(xs_updated)

        while count < popsize:
            parentidx = local_random.integers(low=0, high=popsize, size=(swarm_size, 1))
            for p, sl in enumerate(pop_slices):
                parent = population_parm[sl][parentidx[p, 0], :]
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
                xs_gens[p].append(child)
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


def update_position(parameters, velocity, xlb, xub):
    position = np.clip(parameters + velocity, xlb, xub)
    return position


def velocity_vector(local_random, position, velocity, archive, crowding, xlb, xub):
    r1 = local_random.uniform(low=0.0, high=1.0, size=1)[0]
    r2 = local_random.uniform(low=0.0, high=1.0, size=1)[0]
    w = local_random.uniform(low=0.1, high=0.5, size=1)[0]
    c1 = local_random.uniform(low=1.5, high=2.5, size=1)[0]
    c2 = local_random.uniform(low=1.5, high=2.5, size=1)[0]
    phi = 0
    if c1 + c2 > 4:
        phi = c1 + c2
    else:
        phi = 0
    chi = 2 / (2 - phi - ((phi**2) - 4 * phi) ** (1 / 2))

    output = np.zeros((position.shape[0], velocity.shape[1]))
    delta = [(xub[i] - xlb[i]) / 2 for i in range(0, len(xlb))]
    if archive.shape[0] > 2:
        ind_1, ind_2 = local_random.integers(low=0, high=archive.shape[0], size=2)
        if crowding[ind_1] < crowding[ind_2]:
            ind_1, ind_2 = ind_2, ind_1
    else:
        ind_1 = 0
        ind_2 = 0

    for i in range(0, velocity.shape[0]):
        for j in range(0, velocity.shape[1]):
            output[i, j] = (
                w * velocity[i, j]
                + c1 * r1 * (archive[ind_1, j] - position[i, j])
                + c2 * r2 * (archive[ind_2, j] - position[i, j])
            ) * chi
            output[i, j] = np.clip(output[i, j], -delta[j], delta[j])

    return output
