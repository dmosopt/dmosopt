#
# Adaptive evolutionary algorithm based on non-euclidean geometry for
# many-objective optimization. A. Panichella, Proceedings of the
# Genetic and Evolutionary Computation Conference, 2019.
#
#
# Based on implementations in platEMO and PyMOO (by Ben Crulis):
#
#
# https://github.com/BenCrulis/pymoo/tree/AGE_MOEA/pymoo
# https://github.com/BIMK/PlatEMO/tree/master/PlatEMO/Algorithms/Multi-objective%20optimization/AGE-MOEA
#

import numpy as np
from functools import reduce
from dmosopt.dda import dda_non_dominated_sort
from dmosopt.MOEA import (
    Struct,
    MOEA,
    crossover_sbx,
    mutation,
    tournament_selection,
    remove_duplicates,
)
from typing import Any, Union, Dict, List, Tuple, Optional


class AGEMOEA(MOEA):
    def __init__(
        self,
        popsize: int,
        nInput: int,
        nOutput: int,
        feasibility_model: Optional[Any],
        **kwargs,
    ):
        """AGE-MOEA, A multi-objective algorithm based on non-euclidean geometry."""

        super().__init__(
            name="AGEMOEA",
            popsize=popsize,
            nInput=nInput,
            nOutput=nOutput,
            **kwargs,
        )

        self.logger = None

        self.feasibility_model = feasibility_model

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
        """Returns default parameters of AGE-MOEA strategy."""
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
        population_parm, population_obj, rank, crowd_dist = environmental_selection(
            local_random,
            x,
            y,
            self.popsize,
            self.nInput,
            self.nOutput,
            logger=self.logger,
        )

        population_parm = x[: self.popsize]
        population_obj = y[: self.popsize]
        rank = rank[: self.popsize]
        crowd_dist = crowd_dist[: self.popsize]

        state = Struct(
            bounds=bounds,
            population_parm=population_parm,
            population_obj=population_obj,
            rank=rank,
            crowd_dist=crowd_dist,
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
        crowd_dist = self.state.crowd_dist

        pool_idxs = tournament_selection(
            local_random, popsize, poolsize, -crowd_dist, rank
        )
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
        local_random = self.local_random

        population_parm = np.vstack((population_parm, x_gen))
        population_obj = np.vstack((population_obj, y_gen))
        population_parm, population_obj = remove_duplicates(
            population_parm, population_obj
        )

        population_parm, population_obj, rank, crowd_dist = environmental_selection(
            local_random,
            population_parm,
            population_obj,
            popsize,
            nInput,
            nOutput,
            logger=self.logger,
        )

        self.state.population_parm[:] = population_parm
        self.state.population_obj[:] = population_obj
        self.state.rank[:] = rank
        self.state.crowd_dist[:] = crowd_dist

    def get_population_strategy(self):
        pop_x = self.state.population_parm.copy()
        pop_y = self.state.population_obj.copy()

        return pop_x, pop_y


def sortMO(x, y):
    """Non-dominated sort for multi-objective optimization
    x: input parameter matrix
    y: output objectives matrix
    """
    rank = dda_non_dominated_sort(y)
    idxr = rank.argsort()
    rank = rank[idxr]
    x = x[idxr, :]
    y = y[idxr, :]

    return x, y, rank


def normalize(front, extreme):
    """Rescale and normalize first non-dominated front:

    f^{i}_{norm}(S) = [ f_i(S) - z^{min}_{i} ] / a_{i} \forall S \in F1

    where:
      f_{i} objective i for solution S
      z^{min}_{i} minimum value across all solutions in the front
      a_{i} the intercept of the M-dimensional hyperplane with the objective axis f_{i}

    The M-dimensional hyperplane is composed by the extreme vectors

        z^{max}_{i} = max (f_{i}(S) - z^{min}_{i}), S \in F1
        i.e., the largest objective values in F1 after the translation
        toward the origin of the axes.
    """
    m, n = front.shape

    # if system is abnormal, use min-max normalization
    if len(extreme) != len(np.unique(extreme, axis=0)):
        normalization = np.max(front, axis=0)
        return normalization

    # Calculate the intercepts of the hyperplane constructed by the extreme
    # points and the axes

    try:
        hyperplane = np.linalg.solve(front[extreme], np.ones(n))
    except:
        hyperplane = [np.nan]

    if any(np.isnan(hyperplane)) or any(np.isinf(hyperplane)) or any(hyperplane < 0):
        normalization = np.max(front, axis=0)
    else:
        normalization = 1.0 / hyperplane
        if any(np.isnan(normalization)) or any(np.isinf(normalization)):
            normalization = np.max(front, axis=0)

    normalization[np.isclose(normalization, 0.0, rtol=1e-4, atol=1e-4)] = 1.0

    return normalization


def minkowski_distances(A, B, p):
    """workaround for scipy's cdist refusing p<1"""
    i_ind, j_ind = np.meshgrid(np.arange(A.shape[0]), np.arange(B.shape[0]))
    return np.power(np.power(np.abs(A[i_ind] - B[j_ind]), p).sum(axis=2), 1.0 / p)


def get_geometry(front, extreme):
    """approximate p(norm)"""

    m, n = front.shape

    d = point_2_line_distance(front, np.zeros(n), np.ones(n))
    d[extreme] = np.inf
    index = np.argmin(d)
    p = np.log(n) / np.log(1.0 / np.mean(front[index, :]))

    if np.isnan(p) or p <= 0.1:
        p = 1.0
    elif p > 20:
        p = 20.0  # avoid numpy underflow

    return p


def point_2_line_distance(P, A, B):
    d = np.zeros(P.shape[0])

    for i in range(P.shape[0]):
        pa = P[i] - A
        ba = B - A
        t = np.dot(pa, ba) / np.dot(ba, ba)
        d[i] = np.linalg.norm(pa - t * ba, 2)

    return d


def find_corner_solutions(front):
    """Return the indexes of the extreme points."""

    m, n = front.shape

    if m <= n:
        return np.arange(m)

    # let's define the axes of the n-dimensional spaces
    W = 1e-6 + np.eye(n)
    r = W.shape[0]
    indexes = np.zeros(n, dtype=int)
    selected = np.zeros(m, dtype=bool)
    for i in range(r):
        dists = point_2_line_distance(front, np.zeros(n), W[i, :])
        dists[selected] = np.inf  # prevent already selected to be reselected
        index = np.argmin(dists)
        indexes[i] = index
        selected[index] = True

    return indexes


def survival_score(y, front, ideal_point):
    """front: index of non-dominated solutions of rank d"""

    m, n = y[front, :].shape
    crowd_dist = np.zeros(m)

    if m < n:
        p = 1
        normalization = np.max(y[front, :], axis=0)
        normalization[np.isclose(normalization, 0.0, rtol=1e-4, atol=1e-4)] = 1.0
        return normalization, p, crowd_dist

    # shift the ideal point to the origin
    yfront = y[front, :] - ideal_point

    extreme = find_corner_solutions(yfront)
    normalization = normalize(yfront, extreme)
    ynfront = yfront / normalization
    p = get_geometry(ynfront, extreme)

    # set the distance for the extreme solutions
    crowd_dist[extreme] = np.inf
    selected = np.full(m, False)
    selected[extreme] = True

    nn = np.linalg.norm(ynfront, p, axis=1)
    distances = minkowski_distances(ynfront, ynfront, p=p)
    distances = distances / nn[:, None]

    neighbors = 2
    remaining = np.arange(m)
    remaining = list(remaining[~selected])

    for i in range(m - np.sum(selected)):
        mg = np.meshgrid(np.arange(selected.shape[0])[selected], remaining)
        D_mg = distances[
            tuple(mg)
        ]  # avoid Numpy's future deprecation of array special indexing

        if D_mg.shape[1] > 1:
            # equivalent to mink(distances(remaining, selected),neighbors,2); in Matlab
            maxim = np.argpartition(D_mg, neighbors - 1, axis=1)[:, :neighbors]
            tmp = np.sum(np.take_along_axis(D_mg, maxim, axis=1), axis=1)
            index: int = np.argmax(tmp)
            d = tmp[index]
        else:
            index = D_mg[:, 0].argmax()
            d = D_mg[index, 0]

        best = remaining.pop(index)
        selected[best] = True
        crowd_dist[best] = d

    return normalization, p, crowd_dist


def environmental_selection(
    local_random,
    population_parm,
    population_obj,
    pop,
    nInput,
    nOutput,
    feasibility_model=None,
    logger=None,
):
    # get max int value
    max_int = np.iinfo(np.int32).max

    xs, ys, rank = sortMO(population_parm, population_obj)
    rmax = int(np.max(rank[rank != max_int]))
    rmin = int(np.min(rank))

    yn = np.zeros_like(ys)
    crowd_dist = np.zeros_like(rank).astype(np.float32)
    selected = np.zeros_like(rank).astype(bool)

    # get the first front for normalization
    front_1 = np.argwhere(rank == 0).ravel()

    # follows from the definition of the ideal point but with current non-dominated solutions
    ideal_point = np.min(ys[front_1, :], axis=0)

    normalization, p, crowd_dist[front_1] = survival_score(ys, front_1, ideal_point)
    yn[front_1, :] = ys[front_1] / normalization

    count = len(front_1)
    if count < pop:
        selected[front_1] = True
        for r in range(1, rmax + 1):
            front_r = np.argwhere(rank == r).ravel()
            yn[front_r] = ys[front_r] / normalization
            crowd_dist[front_r] = 1.0 / minkowski_distances(
                yn[front_r, :], ideal_point[None, :], p=p
            )
            if (count + len(front_r)) < pop:
                selected[front_r] = True
                count += len(front_r)
            else:
                # Select the solutions in the last front based on their crowding distances
                sort_keys = []
                if feasibility_model is not None:
                    sort_keys.append(-feasibility_model.rank(xs[front_r]))
                sort_keys.append(-crowd_dist[front_r])
                perm = np.lexsort(sort_keys)
                selected[front_r[perm[: pop - count]]] = True
                break

    else:
        sort_keys = []
        if feasibility_model is not None:
            sort_keys.append(-feasibility_model.rank(xs[front_1]))
        sort_keys.append(-crowd_dist[front_1])
        perm = np.lexsort(sort_keys)
        selected[front_1[perm[:pop]]] = True

    assert np.sum(selected) > 0

    # return selected solutions, number of selected should be equal to population size
    return (
        xs[selected].copy(),
        ys[selected].copy(),
        rank[selected].copy(),
        crowd_dist[selected].copy(),
    )
