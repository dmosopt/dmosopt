###
### Multiobjective CMA-ES optimization class based on the papers
### "Efficient Covariance Matrix Update for Variable Metric Evolution Strategies", Suttorp, Hansen, Igel; 2009.
### "Improved Step Size Adaptation for the MO-CMA-ES", Voss, Hansen, Igel; 2010.
###
### Based on code from:
### https://github.com/DEAP/deap/blob/master/deap/cma.py
###

import gc, itertools, math
from functools import partial
import numpy as np
from numpy.random import default_rng
from typing import Any, Union, Dict, List, Tuple, Optional
from dmosopt import sampling
from dmosopt.datatypes import OptHistory
from dmosopt.dda import dda_non_dominated_sort
from dmosopt.MOEA import remove_worst, remove_duplicates
from dmosopt.indicators import Hypervolume


def sortMO(x, y):
    """Non-dominated sort for multi-objective optimization
    x: input parameter matrix
    y: output objectives matrix
    """
    rank = dda_non_dominated_sort(y)

    return rank


def optimization(
    model,
    nInput,
    nOutput,
    xlb,
    xub,
    initial=None,
    gen=100,
    pop=100,
    sigma=0.01,
    di_mutation=1.0,
    mu=None,
    sampling_method=None,
    termination=None,
    local_random=None,
    logger=None,
    **kwargs,
):
    """
    Multiobjective CMA-ES optimization class based on the paper
    Voss, Hansen, Igel, "Improved Step Size Adaptation for the MO-CMA-ES", 2010.

    :param model: the evaluated model function
    :param nInput: number of model parameters
    :param nOutput: number of output objectives
    :param xlb: lower bound of input
    :param xub: upper bound of input
    :param pop: number of population
    :param sampling_method: optional callable for initial sampling of parameters
    :param pop: Size of population of individuals.
    :param sigma: Coordinate-wise standard deviation (step size)
    :param mu: The number of parents to use in the evolution. When not
               provided it defaults to *pop*. (optional)
    :param lambda_: The number of offspring to produce at each generation.
                    (optional, defaults to 1)

    Other parameters can be provided as described in the next table
    +----------------+---------------------------+----------------------------+
    | Parameter      | Default                   | Details                    |
    +================+===========================+============================+
    | ``d``          | ``1.0 + N / 2.0``         | Damping for step-size.     |
    +----------------+---------------------------+----------------------------+
    | ``ptarg``      | ``1.0 / (5 + 1.0 / 2.0)`` | Target success rate.        |
    +----------------+---------------------------+----------------------------+
    | ``cp``         | ``ptarg / (2.0 + ptarg)`` | Step size learning rate.   |
    +----------------+---------------------------+----------------------------+
    | ``cc``         | ``2.0 / (N + 2.0)``       | Cumulation time horizon.   |
    +----------------+---------------------------+----------------------------+
    | ``ccov``       | ``2.0 / (N**2 + 6.0)``    | Covariance matrix learning |
    |                |                           | rate.                      |
    +----------------+---------------------------+----------------------------+
    | ``pthresh``    | ``0.44``                  | Threshold success rate.    |
    +----------------+---------------------------+----------------------------+

    """

    if local_random is None:
        local_random = default_rng()

    x_initial, y_initial = None, None
    if initial is not None:
        x_initial, y_initial = initial

    if mu is None:
        mu = pop

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

    population_parm = x[:pop]
    population_obj = y[:pop]

    bounds = np.column_stack((xlb, xub))
    optimizer = CMAES(
        x=population_parm,
        y=population_obj,
        sigma=sigma,
        mu=mu,
        di_mutation=di_mutation,
        bounds=bounds,
        local_random=local_random,
        **kwargs,
    )

    x_new = []
    y_new = []

    gen_indexes = []
    gen_indexes.append(np.zeros((x_initial.shape[0],), dtype=np.uint32))

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
                logger.info(f"CMAES: generation {i}...")
            else:
                logger.info(f"CMAES: generation {i} of {gen}...")

        count = 0
        xs_gen = []
        ys_gen = []
        while count < pop - 1:
            x_gen_i, p_idx_gen_i = optimizer.generate()

            y_gen_i = model.evaluate(x_gen_i)

            # Update w/ evaluation values.
            optimizer.update(x_gen_i, y_gen_i, p_idx_gen_i)

            xs_gen.append(x_gen_i)
            ys_gen.append(y_gen_i)
            count += x_gen_i.shape[0]

        x_gen = np.vstack(xs_gen)
        y_gen = np.vstack(ys_gen)

        x_new.append(x_gen)
        y_new.append(y_gen)

        population_parm = np.vstack((population_parm, x_gen))
        population_obj = np.vstack((population_obj, y_gen))
        population_parm, population_obj = remove_duplicates(
            population_parm, population_obj
        )
        population_parm, population_obj, rank = remove_worst(
            population_parm, population_obj, pop, nInput, nOutput
        )

        gen_indexes.append(np.ones((x_gen.shape[0],), dtype=np.uint32) * i)

        n_eval += count

    bestx = population_parm.copy()
    besty = population_obj.copy()

    x = np.vstack([x_initial] + x_new)
    y = np.vstack([y_initial] + y_new)

    gen_index = np.concatenate(gen_indexes)

    return bestx, besty, gen_index, x, y


def updateCholesky(A, Ainv, z, psucc, pc, cc, ccov, pthresh):

    alpha = None
    if psucc < pthresh:
        pc = (1.0 - cc) * pc + np.sqrt(cc * (2.0 - cc)) * z
        alpha = 1.0 - ccov
    else:
        pc = (1.0 - cc) * pc
        alpha = (1.0 - ccov) + ccov * cc * (2.0 - cc)

    beta = ccov
    w = np.dot(Ainv, pc)

    # Under this threshold, the update is mostly noise
    if w.max() > 1e-20:
        w_inv = np.dot(w, Ainv)
        a = np.sqrt(alpha)
        norm_w2 = np.sum(w**2)
        root = np.sqrt(1 + beta / alpha * norm_w2)
        b = a / norm_w2 * (root - 1)
        A = a * A + b * np.outer(pc, w.T)

        c = 1.0 / (a * norm_w2) * (1.0 - 1.0 / root)
        Ainv = (1.0 / a) * Ainv - c * w * (w.T * Ainv)

    return A, Ainv, pc


class CMAES:
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        bounds: np.ndarray,
        sigma: float,
        di_mutation: Union[float, np.ndarray],
        mu: Optional[int],
        local_random: Optional[np.random.Generator] = None,
        **params,
    ):

        if local_random is None:
            local_random = default_rng()
        self.local_random = local_random

        self.indicator = Hypervolume

        population_size = x.shape[0]
        nInput = x.shape[1]
        nOutput = y.shape[1]

        self.parents_x = x
        self.parents_y = y

        self.dim = nInput
        self.bounds = bounds

        self.di_mutation = di_mutation
        if np.isscalar(di_mutation):
            self.di_mutation = np.asarray([di_mutation] * self.dim)

        # Selection
        if mu is None:
            mu = population_size // 2
        self.mu = mu
        self.lambda_ = params.get("lambda_", nInput)

        # Step size control
        self.d = params.get("d", 1.0 + self.dim / 2.0)
        self.ptarg = params.get("ptarg", 1.0 / (5.0 + 0.5))
        self.cp = params.get("cp", self.ptarg / (2.0 + self.ptarg))

        # Covariance matrix adaptation
        self.cc = params.get("cc", 2.0 / (self.dim + 2.0))
        self.ccov = params.get("ccov", 2.0 / (self.dim**2 + 6.0))
        self.pthresh = params.get("pthresh", 0.44)

        # Internal parameters associated to the mu parent
        self.sigmas = np.asarray(
            [sigma * (1.0 / (self.di_mutation + 1.0))] * population_size
        )

        # Lower Cholesky matrix (Sampling matrix)
        self.A = np.stack([np.identity(self.dim) for _ in range(population_size)])

        # Inverse Cholesky matrix (Used in the update of A)
        self.Ainv = np.stack([np.identity(self.dim) for _ in range(population_size)])
        self.pc = np.stack([np.zeros(self.dim) for _ in range(population_size)])
        self.psucc = np.asarray([self.ptarg] * population_size)

    def generate(self):
        """
        Generates a population of :math:`\lambda` individuals.
        :returns: A list of individuals.
        """

        arz = self.local_random.normal(size=(self.lambda_, self.dim))
        individuals = None

        # Each parent produces an offspring
        if self.lambda_ == self.mu:
            individuals = self.parents_x + self.sigmas * np.einsum(
                "ijk,ik->ij", self.A, arz
            )
            p_idx_array = np.asarray(range(self.lambda_), dtype=np.int_)
        # Parents producing an offspring are chosen at random from the first front
        else:
            rank = sortMO(self.parents_x, self.parents_y)
            front_1 = np.argwhere(rank == 0).ravel()
            js = self.local_random.choice(len(front_1), size=self.lambda_)
            p_idx_array = front_1[js]
            individuals = self.parents_x[p_idx_array] + self.sigmas[
                p_idx_array
            ] * np.einsum("ijk,ik->ij", self.A[p_idx_array], arz)

        x_new = np.clip(individuals, self.bounds[:, 0], self.bounds[:, 1])
        return x_new, p_idx_array

    def _select(self, candidates_x, candidates_y, candidates_ps, candidates_inds):

        if candidates_x.shape[0] <= self.mu:
            return np.ones_like(candidates_inds, dtype=bool_), np.zeros_like(
                candidates_inds, dtype=bool
            )

        rank = sortMO(candidates_x, candidates_y)

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
            front_r = np.argwhere(rank == r).ravel()
            if chosen_count + len(front_r) <= self.mu and not full:
                chosen[front_r] = True
                chosen_count += len(front_r)
            elif mid_front is None and chosen_count < self.mu:
                mid_front = front_r.copy()
                # With this front, we selected enough individuals
                full = True
            else:
                not_chosen[front_r] = True

        # Separate the mid front to accept only k individuals
        k = self.mu - chosen_count
        if k > 0:
            # reference point is chosen in the complete population
            # as the worst in each dimension +1
            ref = np.max(candidates_y, axis=0) + 1
            indicator = self.indicator(ref_point=ref)

            def contribution(front, i):
                # The contribution of point p_i in point set P
                # is the hypervolume of P without p_i
                return indicator.do(
                    np.concatenate(
                        (candidates_y[front[:i]], candidates_y[front[i + 1 :]])
                    )
                )

            contrib_values = np.fromiter(
                map(partial(contribution, mid_front), range(len(mid_front))),
                dtype=np.float32,
            )
            contrib_order = np.argsort(contrib_values)

            chosen[mid_front[contrib_order[:k]]] = True
            not_chosen[mid_front[contrib_order[k:]]] = True

        return chosen, not_chosen

    def update(self, x, y, p_idxs):
        """Update the current covariance matrix strategies from the population.
        :param x:
        :param y:
        """

        # Every parent i gets assigned a "parent index" of i
        P = self.parents_x.shape[0]
        parent_pidxs = np.asarray(range(P), dtype=np.int_)

        C = x.shape[0]
        candidates_x = np.vstack((x, self.parents_x))
        candidates_y = np.vstack((y, self.parents_y))
        candidates_offspring = np.concatenate(
            (
                np.asarray([True] * C, dtype=bool),
                np.asarray([False] * P, dtype=bool),
            )
        )
        candidates_pidxs = np.concatenate((p_idxs, parent_pidxs))
        chosen, not_chosen = self._select(
            candidates_x, candidates_y, candidates_offspring, candidates_pidxs
        )

        cp, cc, ccov = self.cp, self.cc, self.ccov
        d, ptarg, pthresh = self.d, self.ptarg, self.pthresh

        # Make copies for chosen offspring only
        chosen_offspring = np.logical_and(chosen, candidates_offspring)
        sigmas = np.where(
            chosen_offspring.reshape((-1, 1)),
            self.sigmas[candidates_pidxs],
            np.full((candidates_pidxs.shape[0], self.dim), np.nan),
        )
        last_steps = sigmas.copy()
        Ainv = np.where(
            chosen_offspring.reshape((-1, 1, 1)), self.Ainv[candidates_pidxs], np.nan
        )
        A = np.where(
            chosen_offspring.reshape((-1, 1, 1)), self.A[candidates_pidxs], np.nan
        )
        pc = np.where(
            chosen_offspring.reshape((-1, 1)), self.pc[candidates_pidxs], np.nan
        )
        psucc = np.where(chosen_offspring, self.psucc[candidates_pidxs], np.nan)

        # Update the internal parameters for successful offspring
        for ind in np.nonzero(chosen)[0]:

            is_offspring = candidates_offspring[ind]
            p_idx = candidates_pidxs[ind]

            # Only the offspring update the parameter set
            if is_offspring:
                # Update (Success = 1 since it is chosen)
                psucc[ind] = (1.0 - cp) * psucc[ind] + cp
                sigmas[ind] = sigmas[ind] * np.exp(
                    (psucc[ind] - ptarg) / (d * (1.0 - ptarg))
                )

                xp = candidates_x[ind]
                x = self.parents_x[p_idx]
                z = (xp - x) / last_steps[ind]
                A[ind], Ainv[ind], pc[ind] = updateCholesky(
                    A[ind], Ainv[ind], z, psucc[ind], pc[ind], cc, ccov, pthresh
                )

                self.psucc[p_idx] = (1.0 - cp) * self.psucc[p_idx] + cp
                self.sigmas[p_idx] = self.sigmas[p_idx] * np.exp(
                    (self.psucc[p_idx] - ptarg) / (d * (1.0 - ptarg))
                )

        # It is unnecessary to update the entire parameter set for not chosen individuals
        # Their parameters will not make it to the next generation
        for ind in np.nonzero(not_chosen)[0]:

            is_offspring = candidates_offspring[ind]
            p_idx = candidates_pidxs[ind]

            # Only the offspring update the parameter set
            if is_offspring:
                self.psucc[p_idx] = (1.0 - cp) * self.psucc[p_idx]
                self.sigmas[p_idx] = self.sigmas[p_idx] * np.exp(
                    (self.psucc[p_idx] - ptarg) / (d * (1.0 - ptarg))
                )

        # Make a copy of the internal parameters
        # The parameter is in the temporary variable for offspring and in the original one for parents
        self.parents_x = candidates_x[chosen]
        self.parents_y = candidates_y[chosen]

        self.sigmas = np.where(
            candidates_offspring[chosen].reshape((-1, 1)),
            sigmas[chosen],
            self.sigmas[candidates_pidxs[chosen]],
        )
        self.Ainv = np.where(
            candidates_offspring[chosen].reshape((-1, 1, 1)),
            Ainv[chosen],
            self.Ainv[candidates_pidxs[chosen]],
        )
        self.A = np.where(
            candidates_offspring[chosen].reshape((-1, 1, 1)),
            A[chosen],
            self.A[candidates_pidxs[chosen]],
        )
        self.pc = np.where(
            candidates_offspring[chosen].reshape((-1, 1)),
            pc[chosen],
            self.pc[candidates_pidxs[chosen]],
        )
        self.psucc = np.where(
            candidates_offspring[chosen],
            psucc[chosen],
            self.psucc[candidates_pidxs[chosen]],
        )
