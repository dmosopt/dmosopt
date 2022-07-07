###
### Multiobjective CMA-ES optimization class based on the paper
### "Improved Step Size Adaptation for the MO-CMA-ES", Voss, Hansen, Igel; 2010.
###
### Based on code from 
### https://raw.githubusercontent.com/CyberAgentAILab/cmaes/main/cmaes/_cma.py
### https://github.com/DEAP/deap/blob/master/deap/cma.py
###

import gc, itertools
import numpy as np
from numpy.random import default_rng
from typing import Any, Dict, List, Tuple, Optional
from dmosopt import sampling
from dmosopt.datatypes import OptHistory
from dmosopt.dda import dda_non_dominated_sort
from dmosopt.MOEA import remove_worst, remove_duplicates
from dmosopt.indicators import Hypervolume

_EPS = 1e-8
_MEAN_MAX = 1e32
_SIGMA_MAX = 1e32

def sortMO(x, y):
    ''' Non-dominated sort for multi-objective optimization
        x: input parameter matrix
        y: output objectives matrix
    '''
    rank = dda_non_dominated_sort(y)

    return rank


def optimization(model, nInput, nOutput, xlb, xub, initial=None, gen=100,
                 pop=100, sigma=0.9, mu=None, sampling_method=None, termination=None,
                 local_random=None, logger=None, **kwargs):
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
    elif callable(sampling_method):
        x = sampling_method(pop, nInput, local_random, xlb, xub)
    else:
        raise RuntimeError(f'Unknown sampling method {sampling_method}')
    if x_initial is not None:
        x = np.vstack((x_initial, x))
    
    y = np.zeros((pop, nOutput))
    for i in range(pop):
        y[i,:] = model.evaluate(x[i,:])
    if y_initial is not None:
        y = np.vstack((y_initial, y))

    bounds = np.column_stack((xlb, xub))
    optimizer = CMAES(x=x, y=y, sigma=sigma, mu=mu, bounds=bounds, local_random=local_random, **kwargs)

    population_parm = np.zeros((0, nInput))
    population_obj  = np.zeros((0, nOutput))

    x_new = []
    y_new = []

    gen_indexes = []
    gen_indexes.append(np.zeros((x_initial.shape[0],),dtype=np.uint32))

    n_eval = 0
    it = range(1, gen+1)

    for i in it:
        
        if termination is not None:
            if optimizer.termination():
                break
            
        if logger is not None:
            if termination is not None:
                logger.info(f"CMAES: generation {i}...")
            else:
                logger.info(f"CMAES: generation {i} of {gen}...")

        count = 0
        xs_gen = []
        ys_gen = []
        while (count < pop - 1):
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
        population_obj  = np.vstack((population_obj, y_gen))
        population_parm, population_obj = remove_duplicates(population_parm, population_obj)
        population_parm, population_obj, rank = \
            remove_worst(population_parm, population_obj, pop, nInput, nOutput)

        gen_indexes.append(np.ones((x_gen.shape[0],),dtype=np.uint32)*i)

        n_eval += count

    bestx = population_parm.copy()
    besty = population_obj.copy()

    x = np.vstack([x_initial] + x_new)
    y = np.vstack([y_initial] + y_new)

    gen_index = np.concatenate(gen_indexes)
    
    return bestx, besty, gen_index, x, y
    

class CMAES:
    
    def __init__(self,
                 x: np.ndarray,
                 y: np.ndarray,
                 bounds: np.ndarray,
                 sigma: float,
                 mu: Optional[int],
                 local_random: Optional[np.random.Generator] = None,
                 **params):

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
        
        # Selection
        if mu is None:
            mu = population_size
        self.mu = mu
        self.lambda_ = params.get("lambda_", population_size//10)

        # Step size control
        self.d = params.get("d", 1.0 + self.dim / 2.0)
        self.ptarg = params.get("ptarg", 1.0 / (5.0 + 0.5))
        self.cp = params.get("cp", self.ptarg / (2.0 + self.ptarg))

        # Covariance matrix adaptation
        self.cc = params.get("cc", 2.0 / (self.dim + 2.0))
        self.ccov = params.get("ccov", 2.0 / (self.dim ** 2 + 6.0))
        self.pthresh = params.get("pthresh", 0.44)

        # Internal parameters associated to the mu parent
        self.sigmas = np.asarray([sigma] * population_size)
        # Lower Cholesky matrix (Sampling matrix)
        self.A = np.stack([np.identity(self.dim) for _ in range(population_size)])
        # Inverse Cholesky matrix (Used in the update of A)
        self.invCholesky = np.stack([np.identity(self.dim) for _ in range(population_size)])
        self.pc = np.stack([np.zeros(self.dim) for _ in range(population_size)])
        self.psucc = np.asarray([self.ptarg] * population_size)

        
    def generate(self):
        """
        Generates a population of :math:`\lambda` individuals.
        :returns: A list of individuals.
        """
        arz = self.local_random.normal(size=(self.lambda_, self.dim))
        individuals = list()
        
        p_idxs = []
        
        # Each parent produce an offspring
        if self.lambda_ == self.mu:
            for i in range(self.lambda_):
                individuals.append(self.parents_x[i] + self.sigmas[i] * np.dot(self.A[i], arz[i]))
                p_idxs.append(i)
        # Parents producing an offspring are chosen at random from the first front
        else:
            rank = sortMO(self.parents_x, self.parents_y)
            front_1 = np.argwhere(rank == 0).ravel()
            for i in range(self.lambda_):
                j = self.local_random.integers(0, len(front_1))
                p_idx = front_1[j]
                individuals.append(self.parents_x[p_idx] + self.sigmas[p_idx] * np.dot(self.A[p_idx], arz[i]))
                p_idxs.append(p_idx)

        x_new = np.vstack(individuals)
        x_new = np.clip(x_new, self.bounds[:,0], self.bounds[:,1])
        p_idx_array = np.asarray(p_idxs, dtype=np.int_)
        return x_new, p_idx_array


    def _select(self, candidates_x, candidates_y, candidates_ps, candidates_inds):
        
        if candidates_x.shape[0] <= self.mu:
            return np.ones_like(candidates_inds, dtype=np.bool_), np.zeros_like(candidates_inds, dtype=np.bool_)

        rank = sortMO(candidates_x, candidates_y)

        chosen = np.zeros_like(candidates_inds, dtype=np.bool_)
        not_chosen = np.zeros_like(candidates_inds, dtype=np.bool_)
        mid_front = None

        # Fill the next population (chosen) with the fronts until there is not enough space
        # When an entire front does not fit in the space left we rely on the hypervolume
        # for this front
        # The remaining fronts are explicitly not chosen
        full = False
        rmax = int(np.max(rank))
        chosen_count = 0
        for r in range(rmax+1):
            front_r = np.argwhere(rank == r).ravel()
            if chosen_count + len(front_r) <= self.mu and not full:
                chosen[front_r] = True
                chosen_count += len(front_r)
            elif mid_front is None and chosen_count < self.mu:
                mid_front = front_r.tolist()
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
            
            for _ in range(len(mid_front) - k):
                idx = np.argmax(indicator.do(candidates_y[mid_front]))
                not_chosen[mid_front.pop(idx)] = True

            chosen[mid_front] = True

        return chosen, not_chosen

    def _update_cov(self, invCholesky, A, alpha, beta, v):
        w = np.dot(invCholesky, v)

        # Under this threshold, the update is mostly noise
        if w.max() > 1e-20:
            w_inv = np.dot(w, invCholesky)
            norm_w2 = np.sum(w ** 2)
            a = np.sqrt(alpha)
            root = np.sqrt(1 + beta / alpha * norm_w2)
            b = a / norm_w2 * (root - 1)

            A = a * A + b * np.outer(v, w)
            invCholesky = 1.0 / a * invCholesky - b / (a ** 2 + a * b * norm_w2) * np.outer(w, w_inv)

        return invCholesky, A

    def update(self, x, y, p_idxs):
        """Update the current covariance matrix strategies from the population.
        :param x:
        :param y:
        """

        # Every parent gets assigned a "parent index" of -1
        P = self.parents_x.shape[0]
        parent_pidxs = np.asarray(range(P), dtype=np.int_)

        C = x.shape[0]
        candidates_x = np.vstack((x, self.parents_x))
        candidates_y = np.vstack((y, self.parents_y))
        candidates_offspring = np.concatenate((np.asarray([True]*C, dtype=np.bool_), np.asarray([False]*P, dtype=np.bool_)))
        candidates_pidxs = np.concatenate((p_idxs, parent_pidxs))
        chosen, not_chosen = self._select(candidates_x, candidates_y, candidates_offspring, candidates_pidxs)

        cp, cc, ccov = self.cp, self.cc, self.ccov
        d, ptarg, pthresh = self.d, self.ptarg, self.pthresh

        # Make copies for chosen offspring only
        chosen_offspring = np.logical_and(chosen, candidates_offspring)
        last_steps = np.where(chosen_offspring, self.sigmas[candidates_pidxs], np.nan)
        sigmas = np.where(chosen_offspring, self.sigmas[candidates_pidxs], np.nan)
        invCholesky = np.where(chosen_offspring.reshape((-1, 1, 1)), self.invCholesky[candidates_pidxs], np.nan)
        A = np.where(chosen_offspring.reshape((-1, 1, 1)), self.A[candidates_pidxs], np.nan)
        pc = np.where(chosen_offspring.reshape((-1, 1)), self.pc[candidates_pidxs], np.nan)
        psucc = np.where(chosen_offspring, self.psucc[candidates_pidxs], np.nan)

        # Update the internal parameters for successful offspring
        for i, ind in enumerate(np.nonzero(chosen)[0]):
            
            is_offspring = candidates_offspring[ind]
            p_idx = candidates_pidxs[ind]

            # Only the offspring update the parameter set
            if is_offspring:
                # Update (Success = 1 since it is chosen)
                psucc[i] = (1.0 - cp) * psucc[i] + cp
                sigmas[i] = sigmas[i] * np.exp((psucc[i] - ptarg) / (d * (1.0 - ptarg)))

                if psucc[i] < pthresh:
                    xp = np.array(ind)
                    x = np.array(self.parents_x[p_idx])
                    pc[i] = (1.0 - cc) * pc[i] + np.sqrt(cc * (2.0 - cc)) * (xp - x) / last_steps[i]
                    invCholesky[i], A[i] = self._update_cov(invCholesky[i], A[i], 1 - ccov, ccov, pc[i])
                else:
                    pc[i] = (1.0 - cc) * pc[i]
                    pc_weight = cc * (2.0 - cc)
                    invCholesky[i], A[i] = self._update_cov(invCholesky[i], A[i], 1 - ccov + pc_weight, ccov, pc[i])

                self.psucc[p_idx] = (1.0 - cp) * self.psucc[p_idx] + cp
                self.sigmas[p_idx] = self.sigmas[p_idx] * np.exp((self.psucc[p_idx] - ptarg) / (d * (1.0 - ptarg)))

        # It is unnecessary to update the entire parameter set for not chosen individuals
        # Their parameters will not make it to the next generation
        for ind in np.nonzero(not_chosen)[0]:
            
            is_offspring = candidates_offspring[ind]
            p_idx = candidates_pidxs[ind]

            # Only the offspring update the parameter set
            if is_offspring:
                self.psucc[p_idx] = (1.0 - cp) * self.psucc[p_idx]
                self.sigmas[p_idx] = self.sigmas[p_idx] * np.exp((self.psucc[p_idx] - ptarg) / (d * (1.0 - ptarg)))

        # Make a copy of the internal parameters
        # The parameter is in the temporary variable for offspring and in the original one for parents
        self.parents_x = candidates_x[chosen]
        self.parents_y = candidates_y[chosen]
        
        self.sigmas = np.where(candidates_offspring[chosen], sigmas[chosen], self.sigmas[candidates_pidxs[chosen]])
        self.invCholesky = np.where(candidates_offspring[chosen].reshape((-1,1,1)), invCholesky[chosen], self.invCholesky[candidates_pidxs[chosen]])
        self.A = np.where(candidates_offspring[chosen].reshape((-1,1,1)), A[chosen], self.A[candidates_pidxs[chosen]])
        self.pc = np.where(candidates_offspring[chosen].reshape((-1,1)), pc[chosen], self.pc[candidates_pidxs[chosen]])
        self.psucc = np.where(candidates_offspring[chosen], psucc[chosen], self.psucc[candidates_pidxs[chosen]])

    def termination(self) -> bool:
        B, D = self.invCholesky, self.A
        dC = np.diag(self.pc)

        ## TODO
        return False
    
        # Stop if the std of the normal distribution is smaller than tolx
        # in all coordinates and pc is smaller than tolx in all components.
        if np.all(self._sigma * dC < self._tolx) and np.all(
            self._sigma * self._pc < self._tolx
        ):
            return True

        # Stop if detecting divergent behavior.
        if self._sigma * np.max(D) > self._tolxup:
            return True

        # No effect coordinates: stop if adding 0.2-standard deviations
        # in any single coordinate does not change m.
        if np.any(self._mean == self._mean + (0.2 * self._sigma * np.sqrt(dC))):
            return True

        # No effect axis: stop if adding 0.1-standard deviation vector in
        # any principal axis direction of C does not change m. 
        i = self.generation % self.dim
        if np.all(self._mean == self._mean + (0.1 * self._sigma * D[i] * B[:, i])):
            return True

        # Stop if the condition number of the covariance matrix exceeds 1e14.
        condition_cov = np.max(D) / np.min(D)
        if condition_cov > self._tolconditioncov:
            return True

        return False
    
        
