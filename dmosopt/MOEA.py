#
# Common routines used by multi-objective evolutionary algorithms.
#

import numpy as np
from functools import reduce
from dmosopt.dda import dda_non_dominated_sort
from dmosopt import sampling
from scipy.spatial.distance import cdist
from typing import Any, Union, Dict, List, Tuple, Optional

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
    nInput,
    nOutput,
    return_perm=False,
    x_distance_metrics=None,
    y_distance_metrics=["crowding"],
):
    """Non-dominated sort for multi-objective optimization
    x: input parameter matrix
    y: output objectives matrix
    nInput: number of input
    nOutput: number of output
    return_perm: if True, return permutation indices of original input
    """
    y_distance_functions = [crowding_distance]
    if y_distance_metrics is not None:
        y_distance_functions = []
        assert len(y_distance_metrics) > 0
        for distance_metric in y_distance_metrics:
            if distance_metric == None:
                y_distance_functions.append(crowding_distance)
            elif distance_metric == "crowding":
                y_distance_functions.append(crowding_distance)
            elif distance_metric == "euclidean":
                y_distance_functions.append(euclidean_distance)
            elif callable(distance_metric):
                y_distance_functions.append(distance_metric)
            else:
                raise RuntimeError(f"sortMO: unknown distance metric {distance_metric}")

    x_distance_functions = []
    if x_distance_metrics is not None:
        for distance_metric in x_distance_metrics:
            if callable(distance_metric):
                x_distance_functions.append(distance_metric)
            else:
                raise RuntimeError(f"sortMO: unknown distance metric {distance_metric}")

    rank = dda_non_dominated_sort(y)

    y_dists = list([np.zeros_like(rank) for _ in y_distance_functions])
    x_dists = list([np.zeros_like(rank) for _ in x_distance_functions])
    rmax = int(rank.max())
    for front in range(rmax + 1):
        rankidx = rank == front
        for i, y_distance_function in enumerate(y_distance_functions):
            D = y_distance_function(y[rankidx, :])
            y_dists[i][rankidx] = D
        for i, x_distance_function in enumerate(x_distance_functions):
            D = x_distance_function(x[rankidx, :])
            x_dists[i][rankidx] = D

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


def crowding_distance(Y):
    """Crowding distance metric.
    Y is the output data matrix
    [n,d] = size(Y)
    n: number of points
    d: number of dimensions
    """
    n, d = Y.shape
    lb = np.min(Y, axis=0, keepdims=True)
    ub = np.max(Y, axis=0, keepdims=True)

    if n == 1:
        D = np.array([1.0])
    else:
        ub_minus_lb = ub - lb
        ub_minus_lb[ub_minus_lb == 0.0] = 1.0

        U = (Y - lb) / ub_minus_lb

        D = np.zeros(n)
        DS = np.zeros((n, d))

        idx = U.argsort(axis=0)
        US = np.zeros((n, d))
        for i in range(d):
            US[:, i] = U[idx[:, i], i]

        DS[0, :] = 1.0
        DS[n - 1, :] = 1.0

        for i in range(1, n - 1):
            for j in range(d):
                DS[i, j] = US[i + 1, j] - US[i - 1, j]

        for i in range(n):
            for j in range(d):
                D[idx[i, j]] += DS[i, j]
        D[np.isnan(D)] = 0.0

    return D


def euclidean_distance(Y):
    """Row-wise euclidean distance."""
    n, d = Y.shape
    lb = np.min(Y, axis=0)
    ub = np.max(Y, axis=0)
    ub_minus_lb = ub - lb
    ub_minus_lb[ub_minus_lb == 0.0] = 1.0
    U = (Y - lb) / ub_minus_lb
    return np.sqrt(np.sum(U**2, axis=1))


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
    nInput,
    nOutput,
    x_distance_metrics=None,
    y_distance_metrics=None,
):
    """remove the worst individuals in the population"""
    population_parm, population_obj, rank, _ = sortMO(
        population_parm,
        population_obj,
        nInput,
        nOutput,
        x_distance_metrics=x_distance_metrics,
        y_distance_metrics=y_distance_metrics,
    )
    return population_parm[0:pop, :], population_obj[0:pop, :], rank[0:pop]


def get_duplicates(X, eps=1e-16):
    D = cdist(X, X)
    D[np.triu_indices(len(X))] = np.inf
    D[np.isnan(D)] = np.inf

    is_duplicate = np.zeros((len(X),), dtype=bool)
    is_duplicate[np.any(D <= eps, axis=1)] = True

    return is_duplicate


def remove_duplicates(population_parm, population_obj, eps=1e-16):
    """remove duplicate individuals in the population"""
    is_duplicate = get_duplicates(population_parm, eps=eps)
    return population_parm[~is_duplicate, :], population_obj[~is_duplicate, :]
