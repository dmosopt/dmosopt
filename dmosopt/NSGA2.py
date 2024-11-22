# Nondominated Sorting Genetic Algorithm II (NSGA-II)
# A multi-objective optimization algorithm.

import numpy as np
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
        model: Optional[Any],
        distance_metric: Optional[Any],
        optimize_mean_variance: bool = False,
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

        self.model = model
        self.distance_metric = distance_metric
        self.optimize_mean_variance = optimize_mean_variance
        self.y_distance_metrics = None
        if distance_metric is not None:
            self.y_distance_metrics = []
            self.y_distance_metrics.append(distance_metric)
        self.x_distance_metrics = None
        if self.model.feasibility is not None:
            self.x_distance_metrics = [self.model.feasibility.rank]

        di_crossover = self.opt_params.di_crossover
        if np.isscalar(di_crossover):
            self.opt_params.di_crossover = np.asarray([di_crossover] * nInput)

        di_mutation = self.opt_params.di_mutation
        if np.isscalar(di_mutation):
            self.opt_params.di_mutation = np.asarray([di_mutation] * nInput)
        mutation_rate = self.opt_params.mutation_rate
        if mutation_rate is None:
            self.opt_params.mutation_rate = 1.0 / float(nInput)

        self.opt_params.poolsize = int(round(self.opt_params.popsize / 2.0))

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
            successful_crossovers=0,
            total_crossovers=0,
            successful_mutations=0,
            total_mutations=0,
        )

        return state

    def generate_strategy(self, **params):

        popsize = self.opt_params.popsize
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
        crossover_indices = []
        mutation_indices = []
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
                self.state.total_crossovers += 1
                crossover_indices.extend([count, count + 1])
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
                self.state.total_mutations += 1
                mutation_indices.append(count)
                count += 1
        x_gen = np.vstack(xs_gen)
        # gen_indexes = np.ones((x_gen.shape[0],), dtype=np.uint32) * self.state.gen_index

        return x_gen, {
            "crossover_indices": np.asarray(crossover_indices, dtype=int),
            "mutation_indices": np.asarray(mutation_indices, dtype=int),
        }

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

        popsize = self.opt_params.popsize
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

    def update_population_size(self):
        """Adapt population size based on convergence and diversity."""
        # Calculate diversity metric
        diversity = sum(len(front) for front in fronts[1:]) / len(fronts[0])

        # Calculate crowding distance spread in first front
        if len(fronts[0]) > 1:
            cd_values = [s.crowding_distance for s in fronts[0]]
            cd_spread = np.std(cd_values) / np.mean(cd_values)
        else:
            cd_spread = 0

        # Adjust population size
        if diversity < 0.5 and cd_spread < 0.2:
            # Low diversity - increase population
            new_size = min(max_size, int(current_size * 1.2))
        elif diversity > 2.0 or cd_spread > 0.8:
            # High diversity - decrease population
            new_size = max(50, int(current_size * 0.9))
        else:
            new_size = current_size

        return new_size

    def update_operator_rates(self):
        """Update operator rates and distribution indices based on success statistics."""
        opt_params = self.opt_params
        state = self.state
        # Update crossover parameters
        if state.total_crossovers > 0:
            success_rate = state.successful_crossovers / state.total_crossovers
            if success_rate < opt_params.min_success_rate:
                # Increase exploration
                opt_params.di_crossover = np.max(1.0, opt_params.di_crossover * 0.9)
                opt_params.crossover_prob = np.min(
                    0.95, opt_params.crossover_prob * 1.1
                )
            elif success_rate > opt_params.max_success_rate:
                # Increase exploitation
                opt_params.di_crossover = np.min(100.0, opt_params.di_crossover * 1.1)
                opt_params.crossover_prob = np.max(0.5, opt_params.crossover_prob * 0.9)

        # Update mutation parameters
        if state.total_mutations > 0:
            success_rate = state.successful_mutations / state.total_mutations
            if success_rate < opt_params.min_success_rate:
                # Increase exploration
                opt_params.di_mutation = np.max(1.0, opt_params.di_mutation * 0.9)
                opt_params.mutation_prob = np.min(0.95, opt_params.mutation_prob * 1.1)
            elif success_rate > opt_params.max_success_rate:
                # Increase exploitation
                opt_params.di_mutation = np.min(100.0, opt_params.di_mutation * 1.1)
                opt_params.mutation_prob = np.max(0.1, opt_params.mutation_prob * 0.9)
                opt_params.mutation_rate = np.max(
                    0.1 / self.nInput, opt_params.mutation_rate * 0.9
                )

        # Reset counters
        state.successful_crossovers = 0
        state.total_crossovers = 0
        state.successful_mutations = 0
        state.total_mutations = 0
