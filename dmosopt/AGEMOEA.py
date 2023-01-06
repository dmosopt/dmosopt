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
import gc, itertools
from functools import reduce
from dmosopt import sampling
from dmosopt.datatypes import OptHistory
from dmosopt.dda import dda_non_dominated_sort
from dmosopt.MOEA import (
    crossover_sbx,
    mutation,
    tournament_selection,
    remove_duplicates,
)


def optimization(
    model,
    nInput,
    nOutput,
    xlb,
    xub,
    initial=None,
    feasibility_model=None,
    termination=None,
    pop=100,
    gen=100,
    crossover_prob=0.9,
    mutation_prob=0.1,
    mutation_rate=None,
    nchildren=1,
    di_crossover=1.0,
    di_mutation=20.0,
    sampling_method=None,
    local_random=None,
    logger=None,
    **kwargs,
):
    """AGE-MOEA, A multi-objective algorithm based on non-euclidean geometry.
    model: the evaluated model function
    nInput: number of model input
    nOutput: number of output objectives
    xlb: lower bound of input
    xub: upper bound of input
    pop: number of population
    gen: number of generation
    crossover_prob: probability of crossover in each generation
    mutation_prob: probability of mutation in each generation
    di_crossover: distribution index for crossover
    di_mutation: distribution index for mutation
    """

    if local_random is None:
        local_random = default_rng()

    if np.isscalar(di_crossover):
        di_crossover = np.asarray([di_crossover] * nInput)
    if np.isscalar(di_mutation):
        di_mutation = np.asarray([di_mutation] * nInput)

    poolsize = int(round(pop / 2.0))
    # size of mating pool;

    if mutation_rate is None:
        mutation_rate = 1.0 / float(nInput)

    x_initial, y_initial = None, None
    if initial is not None:
        x_initial, y_initial = initial

    if sampling_method is None:
        x = sampling.lh(pop, nInput, local_random)
        x = x * (xub - xlb) + xlb
    elif sampling_method == "sobol":
        x = sampling.sobol(pop, nInput, local_random)
        x = x * (xub - xlb) + xlb
    elif callable(sampling_method):
        sampling_method_params = kwargs.get("sampling_method_params", None)
        if sampling_method_params is None:
            x = sampling_method(local_random, pop, nInput, xlb, xub)
        else:
            x = sampling_method(local_random, **sampling_method_params)
    else:
        raise RuntimeError(f"Unknown sampling method {sampling_method}")

    y = model.evaluate(x).astype(np.float32)

    if x_initial is not None:
        x = np.vstack((x_initial.astype(np.float32), x))
    if y_initial is not None:
        y = np.vstack((y_initial.astype(np.float32), y))

    gen_indexes = []
    gen_indexes.append(np.zeros((x.shape[0],), dtype=np.int32))

    population_parm = x[:pop]
    population_obj = y[:pop]
    population_parm, population_obj, rank, crowd_dist = environmental_selection(
        local_random,
        population_parm,
        population_obj,
        pop,
        nInput,
        nOutput,
        logger=logger,
    )

    x_new = []
    y_new = []

    n_eval = 0
    it = range(1, gen + 1)
    if termination is not None:
        it = itertools.count(1)
    for i in it:
        if termination is not None:
            opt = OptHistory(i, n_eval, population_parm, population_obj, None)
            if termination.has_terminated(opt):
                break
        if logger is not None:
            if termination is not None:
                logger.info(f"AGE-MOEA: generation {i}...")
            else:
                logger.info(f"AGE-MOEA: generation {i} of {gen}...")

        pool_idxs = tournament_selection(local_random, pop, poolsize, -crowd_dist, rank)
        pool = population_parm[pool_idxs, :]

        count = 0
        xs_gen = []
        while count < pop - 1:
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
        y_gen = model.evaluate(x_gen)
        x_new.append(x_gen)
        y_new.append(y_gen)
        gen_indexes.append(np.ones((x_gen.shape[0],), dtype=np.uint32) * i)

        population_parm = np.vstack((population_parm, x_gen))
        population_obj = np.vstack((population_obj, y_gen))
        population_parm, population_obj = remove_duplicates(
            population_parm, population_obj
        )
        population_parm, population_obj, rank, crowd_dist = environmental_selection(
            local_random,
            population_parm,
            population_obj,
            pop,
            nInput,
            nOutput,
            logger=logger,
        )
        gc.collect()
        n_eval += count

    bestx = population_parm.copy()
    besty = population_obj.copy()

    gen_index = np.concatenate(gen_indexes)
    x = np.vstack([x] + x_new)
    y = np.vstack([y] + y_new)

    return bestx, besty, gen_index, x, y


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
    max_int = np.iinfo(np.int).max

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
