#
# Common routines used by multi-objective evolutionary algorithms.
#

import math
import numpy as np
from functools import reduce
from scipy.spatial.distance import cdist
from typing import Any, Union, Dict, List, Tuple, Optional
from dmosopt.dda import dda_ens
from dmosopt import sampling
from dmosopt.indicators import crowding_distance_metric, euclidean_distance_metric


# function sharedmoea(selfunc,μ,λ)
#  \selfunc, selection function to be used.
# μand λ, population and offspring sizes.
# t ←0; P0 ←randompopulation(μ).
# while end criterion not met do
# Poff←applyvariation(Pt,λ).
# Pt+1 ←selfunc(Pt ∪Poff,μ).
# t ←t +1.
# return nondomset(Pt+1), final non-dominated set


class Struct(object):
    def __init__(self, **items):
        self.__dict__.update(items)

    def update(self, items):
        self.__dict__.update(items)

    def items(self):
        return self.__dict__.items()

    def __call__(self):
        return self.__dict__

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, val):
        self.__dict__[key] = val

    def __contains__(self, k):
        return True if k in self.__dict__ else False

    def __repr__(self):
        return f"Struct({self.__dict__})"

    def __str__(self):
        return f"<Struct>"


class MOEA(object):
    def __init__(self, name: str, popsize: int, nInput: int, nOutput: int, **kwargs):
        """Base Class for a Multi-Objective Evolutionary Algorithm."""

        self.name = name
        self.popsize = popsize
        self.nInput = nInput
        self.nOutput = nOutput
        self.opt_params = Struct(**self.default_parameters)
        self.opt_params.update(
            {
                "popsize": popsize,
                "nInput": nInput,
                "nOutput": nOutput,
                "initial_size": popsize,
                "initial_sampling_method": None,
                "initial_sampling_method_params": None,
            }
        )
        for k, v in kwargs.items():
            if k not in self.opt_params:
                self.opt_params[k] = v
            elif v is not None:
                self.opt_params[k] = v
        self.local_random = None
        self.state = None

    @property
    def default_parameters(self) -> Dict[str, Any]:
        """Return default hyper parameters of algorithm."""
        return {}

    @property
    def opt_parameters(self) -> Dict[str, Any]:
        """Return current hyper parameters of algorithm."""
        params = self.opt_params()
        return params

    @property
    def population_objectives(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.get_population_strategy()

    def get_population_strategy(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def initialize_strategy(
        self,
        x: np.ndarray,
        y: np.ndarray,
        bounds: np.ndarray,
        local_random: Optional[np.random.Generator] = None,
        **params,
    ):
        """Initialize the evolutionary algorithm."""

        self.bounds = bounds
        self.local_random = local_random

        # Initialize strategy based on strategy-specific initialize method
        self.state = self.initialize_state(x, y, bounds, local_random)

        return self.state

    def generate_initial(self, bounds, local_random):
        """Generate an initial set of parameters to initialize strategy."""

        xlb = bounds[:, 0]
        xub = bounds[:, 1]

        nInput = self.nInput
        initial_size = self.opt_params.initial_size
        sampling_method = self.opt_params.initial_sampling_method
        sampling_method_params = self.opt_params.initial_sampling_method_params

        if sampling_method is None:
            x = sampling.lh(initial_size, nInput, local_random)
            x = x * (xub - xlb) + xlb
        elif sampling_method == "sobol":
            x = sampling.sobol(initial_size, nInput, local_random)
            x = x * (xub - xlb) + xlb
        elif callable(sampling_method):
            if sampling_method_params is None:
                x = sampling_method(local_random, initial_size, nInput, xlb, xub)
            else:
                x = sampling_method(local_random, **sampling_method_params)
        else:
            raise RuntimeError(f"Unknown sampling method {sampling_method}")

        return x

    def generate(
        self,
        **params,
    ):
        """Generate new parameter candidates to evaluate next."""

        # Generate parameters to be evaluated based on strategy-specific method
        x, state = self.generate_strategy(**params)

        # Clip proposal candidates into allowed range
        x_clipped = np.clip(x, self.bounds[:, 0], self.bounds[:, 1])

        return x_clipped, state

    def update(
        self,
        x: np.ndarray,
        y: np.ndarray,
        state: Dict[Any, Any],
        **params,
    ):
        """Update objective information for given set of parameters and update algorithm statex."""

        # Update the search state based on strategy-specific update
        self.update_strategy(x, y, state, **params)
        return self.state

    def initialize_state(self, **params):
        """Search-specific initialization method. Returns initial state."""
        raise NotImplementedError

    def generate_strategy(self, **params):
        """Search-specific parameter generation. Returns new parameters & updated state."""
        raise NotImplementedError

    def update_strategy(
        self,
        x: np.ndarray,
        y: np.ndarray,
        state: Dict[Any, Any],
        **params,
    ):
        """Search-specific objectives update. Returns updated state."""
        raise NotImplementedError


def mutation(
    local_random, parent, di_mutation, xlb, xub, mutation_rate=0.5, nchildren=1
):
    """Polynomial Mutation in Genetic Algorithm
    muration_rate: mutation rate
    di_mutation: distribution index for mutation
        This determine how well spread the child will be from its parent.
    parent: sample point before mutation
    """
    n = len(parent)
    if np.isscalar(di_mutation):
        di_mutation = np.asarray([di_mutation] * n)
    children = np.ndarray((nchildren, n))
    delta = np.ndarray((n,))
    for i in range(nchildren):
        u = local_random.random(n)
        lo = np.argwhere(u < mutation_rate).ravel()
        hi = np.argwhere(u >= mutation_rate).ravel()
        delta[lo] = (2.0 * u[lo]) ** (1.0 / (di_mutation[lo] + 1)) - 1.0
        delta[hi] = 1.0 - (2.0 * (1.0 - u[hi])) ** (1.0 / (di_mutation[hi] + 1))
        children[i, :] = np.clip(parent + (xub - xlb) * delta, xlb, xub)
    return children


def crossover_sbx(local_random, parent1, parent2, di_crossover, xlb, xub, nchildren=1):
    """SBX (Simulated Binary Crossover) in Genetic Algorithm

    di_crossover: distribution index for crossover
    This determine how well spread the children will be from their parents.
    """
    n = len(parent1)
    if np.isscalar(di_crossover):
        di_crossover = np.asarray([di_crossover] * n)
    children1 = np.ndarray((nchildren, n))
    children2 = np.ndarray((nchildren, n))
    beta = np.ndarray((n,))
    for i in range(nchildren):
        u = local_random.random(n)
        lo = np.argwhere(u <= 0.5).ravel()
        hi = np.argwhere(u > 0.5).ravel()
        beta[lo] = (2.0 * u[lo]) ** (1.0 / (di_crossover[lo] + 1))
        beta[hi] = (1.0 / (2.0 * (1.0 - u[hi]))) ** (1.0 / (di_crossover[hi] + 1))
        children1[i, :] = np.clip(
            0.5 * ((1 - beta) * parent1 + (1 + beta) * parent2), xlb, xub
        )
        children2[i, :] = np.clip(
            0.5 * ((1 + beta) * parent1 + (1 - beta) * parent2), xlb, xub
        )
    return children1, children2


def sortMO(
    x,
    y,
    return_perm=False,
    x_distance_metrics=None,
    y_distance_metrics=None,
):
    """Non-dominated sort for multi-objective optimization
    x: input parameter matrix
    y: output objectives matrix
    return_perm: if True, return permutation indices of original input
    """
    y_distance_functions = []
    if y_distance_metrics is not None:
        y_distance_functions = []
        assert len(y_distance_metrics) > 0
        for distance_metric in y_distance_metrics:
            if callable(distance_metric):
                y_distance_functions.append(distance_metric)
            elif distance_metric == "crowding":
                y_distance_functions.append(crowding_distance_metric)
            elif distance_metric == "euclidean":
                y_distance_functions.append(euclidean_distance_metric)
            else:
                raise RuntimeError(f"sortMO: unknown distance metric {distance_metric}")

    x_distance_functions = []
    if x_distance_metrics is not None:
        for distance_metric in x_distance_metrics:
            if callable(distance_metric):
                x_distance_functions.append(distance_metric)
            else:
                raise RuntimeError(f"sortMO: unknown distance metric {distance_metric}")

    rank = dda_ens(y)

    y_dists = list([np.zeros_like(rank) for _ in y_distance_functions])
    x_dists = list([np.zeros_like(rank) for _ in x_distance_functions])
    for i, y_distance_function in enumerate(y_distance_functions):
        y_dists[i] = y_distance_function(y)
    for i, x_distance_function in enumerate(x_distance_functions):
        x_dists[i] = x_distance_function(x)

    perm = np.lexsort(
        (list([-dist for dist in x_dists]) + list([-dist for dist in y_dists]) + [rank])
    )

    x = x[perm]
    y = y[perm]
    rank = rank[perm]
    y_dists = tuple([dist[perm] for dist in y_dists])

    if return_perm:
        return x, y, rank, y_dists, perm
    else:
        return x, y, rank, y_dists


def orderMO(
    x,
    y,
    x_distance_metrics=None,
    y_distance_metrics=None,
):
    """Returns the ordering for a non-dominated sort for multi-objective optimization
    x: input parameter matrix
    y: output objectives matrix
    """
    y_distance_functions = []
    if y_distance_metrics is not None:
        assert len(y_distance_metrics) > 0
        for distance_metric in y_distance_metrics:
            if callable(distance_metric):
                y_distance_functions.append(distance_metric)
            elif distance_metric == "crowding":
                y_distance_functions.append(crowding_distance_metric)
            elif distance_metric == "euclidean":
                y_distance_functions.append(euclidean_distance_metric)
            else:
                raise RuntimeError(f"sortMO: unknown distance metric {distance_metric}")

    x_distance_functions = []
    if x_distance_metrics is not None:
        for distance_metric in x_distance_metrics:
            if callable(distance_metric):
                x_distance_functions.append(distance_metric)
            else:
                raise RuntimeError(f"sortMO: unknown distance metric {distance_metric}")

    rank = dda_ens(y)

    y_dists = list([np.zeros_like(rank) for _ in y_distance_functions])
    x_dists = list([np.zeros_like(rank) for _ in x_distance_functions])
    for i, y_distance_function in enumerate(y_distance_functions):
        y_dists[i] = y_distance_function(y)
    for i, x_distance_function in enumerate(x_distance_functions):
        x_dists[i] = x_distance_function(x)

    perm = np.lexsort(
        (list([-dist for dist in x_dists]) + list([-dist for dist in y_dists]) + [rank])
    )

    rank = rank[perm]
    y_dists = tuple([dist[perm] for dist in y_dists])

    return perm, rank, y_dists


def top_k_MO(x, y, top_k=None):
    """Returns the top_k elements ordered by a non-dominated sort for multi-objective optimization
    x: input parameter matrix
    y: output objectives matrix
    top_k: number of top elements. If None, all elements will be returned
    """
    if not isinstance(top_k, int):
        return x, y

    if x.shape[0] <= top_k:
        return x, y

    x_, y_, *_ = sortMO(x, y)
    if x_.shape[0] >= top_k:
        x = x_[:top_k]
        y = y_[:top_k]
    else:
        # if sortMO yields less than top-k
        #  we fall back on regular top-k
        x = x[-top_k:]
        y = y[-top_k:]

    return x, y


def tournament_prob(ax, i):
    p = ax[1]
    try:
        p1 = p * (1.0 - p) ** i
    except FloatingPointError:
        p1 = 0.0
    ax[0].append(p1)
    return (ax[0], p)


def tournament_selection(local_random, pop, poolsize, *metrics):
    """tournament selecting the best individuals into the mating pool"""

    candidates = np.arange(pop)
    sorted_candidates = np.lexsort(tuple((metric[candidates] for metric in metrics)))
    prob, _ = reduce(tournament_prob, candidates, ([], 0.5))
    prob = prob / np.sum(prob)
    poolidx = local_random.choice(
        sorted_candidates, size=poolsize, p=np.asarray(prob), replace=False
    )
    return poolidx


def remove_worst(
    population_parm,
    population_obj,
    pop,
    x_distance_metrics=None,
    y_distance_metrics=None,
    return_perm=False,
):
    """remove the worst individuals in the population"""
    population_parm, population_obj, rank, _, perm = sortMO(
        population_parm,
        population_obj,
        x_distance_metrics=x_distance_metrics,
        y_distance_metrics=y_distance_metrics,
        return_perm=True,
    )

    result = (population_parm[0:pop, :], population_obj[0:pop, :], rank[0:pop])
    if return_perm:
        result = (
            population_parm[0:pop, :],
            population_obj[0:pop, :],
            rank[0:pop],
            perm[0:pop],
        )
    return result


def get_duplicates(X, Y=None, eps=1e-16):
    if Y is None:
        Y = X
    D = cdist(X, Y)
    D[np.triu_indices(len(X), m=len(Y))] = np.inf
    D[np.isnan(D)] = np.inf

    is_duplicate = np.zeros((len(X),), dtype=bool)
    is_duplicate[np.any(D <= eps, axis=1)] = True

    return is_duplicate


def remove_duplicates(population_parm, population_obj, eps=1e-16):
    """remove duplicate individuals in the population"""
    is_duplicate = get_duplicates(population_parm, eps=eps)
    return population_parm[~is_duplicate, :], population_obj[~is_duplicate, :]


class EpsilonSort:
    """
    An archive of epsilon-nondominated solutions.
    Allows auxiliary information to tag along for the sort
    process.

    The eps_sort function provides a much more convenient interface than
    the Archive class.

    ----

    Source: https://github.com/matthewjwoodruff/pareto.py/blob/master/pareto.py

    ----

    Copyright (C) 2013 Matthew Woodruff and Jon Herman.

    This script is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This script is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this script. If not, see <http://www.gnu.org/licenses/>.
    """

    def __init__(self, epsilons):
        """
        epsilons: sizes of epsilon boxes to use in the sort.  Number
                  of objectives is inferred by the number of epsilons.
        """
        self.archive = []  # objectives
        self.tagalongs = []  # tag-along data
        self.boxes = []  # remember for efficiency
        self.epsilons = [e if e != 0 and not np.isnan(e) else 1e-8 for e in epsilons]
        self.itobj = range(len(epsilons))  # infer number of objectives

    def add(self, objectives, tagalong, ebox):
        """add a solution to the archive, plus auxiliary information"""
        self.archive.append(objectives)
        self.tagalongs.append(tagalong)
        self.boxes.append(ebox)

    def remove(self, index):
        """remove a solution from the archive"""
        self.archive.pop(index)
        self.tagalongs.pop(index)
        self.boxes.pop(index)

    def sortinto(self, objectives, tagalong=None):
        """
        Sort a solution into the archive.  Add it if it's nondominated
        w.r.t current solutions.

        objectives: objectives by which to sort.  Minimization is assumed.
        tagalong:   data to preserve with the objectives.  Probably the actual
                    solution is here, the objectives having been extracted
                    and possibly transformed.  Tagalong data can be *anything*.
                    We don't inspect it, just keep a reference to it for as
                    long as the solution is in the archive, and then return
                    it in the end.
        """
        # Here's how the early loop exits in this code work:
        # break:    Stop iterating the box comparison for loop because we know
        #           the solutions are in relatively nondominated boxes.
        # continue: Start the next while loop iteration immediately (i.e.
        #           jump ahead to the comparison with the next archive member).
        # return:   The candidate solution is dominated, stop comparing it to
        #           the archive, don't add it, immediately exit the method.
        objectives = np.nan_to_num(objectives)
        ebox = [math.floor(objectives[ii] / self.epsilons[ii]) for ii in self.itobj]

        asize = len(self.archive)

        ai = -1  # ai: archive index
        while ai < asize - 1:
            ai += 1
            adominate = False  # archive dominates
            sdominate = False  # solution dominates
            nondominate = False  # neither dominates

            abox = self.boxes[ai]

            for oo in self.itobj:
                if abox[oo] < ebox[oo]:
                    adominate = True
                    if sdominate:  # nondomination
                        nondominate = True
                        break  # for
                elif abox[oo] > ebox[oo]:
                    sdominate = True
                    if adominate:  # nondomination
                        nondominate = True
                        break  # for

            if nondominate:
                continue  # while
            if adominate:  # candidate solution was dominated
                return
            if sdominate:  # candidate solution dominated archive solution
                self.remove(ai)
                ai -= 1
                asize -= 1
                continue  # while

            # solutions are in the same box
            aobj = self.archive[ai]
            corner = [ebox[ii] * self.epsilons[ii] for ii in self.itobj]
            sdist = sum([(objectives[ii] - corner[ii]) ** 2 for ii in self.itobj])
            adist = sum([(aobj[ii] - corner[ii]) ** 2 for ii in self.itobj])
            if adist < sdist:  # archive dominates
                return
            else:  # solution dominates
                self.remove(ai)
                ai -= 1
                asize -= 1
                # Need a continue here if we ever reorder the while loop.
                continue  # while

        # if you get here, then no archive solution has dominated this one
        self.add(objectives, tagalong, ebox)
