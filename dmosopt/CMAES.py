###
### Multiobjective CMA-ES optimization class based on the paper
### "Improved Step Size Adaptation for the MO-CMA-ES", Voss, Hansen, Igel; 2010.
###
### Based on code from 
### https://raw.githubusercontent.com/CyberAgentAILab/cmaes/main/cmaes/_cma.py
### https://github.com/DEAP/deap/blob/master/deap/cma.py
###

import math
import numpy as np
from numpy.random import default_rng
from typing import Any, Dict, List, Tuple, Optional
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
                 pop=100, sigma=1.3, sampling_method=None, termination=None,
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
    :param sigma: The initial step size of the complete system.
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
        mu = np.zeros(nInput)

    bounds = np.column_stack((xlb, xub))
    optimizer = CMAES(x=x_initial, y=y_initial, sigma=sigma,
                      bounds=bounds, local_random=local_random, **kwargs)

    population_parm = np.zeros((0, nInput))
    population_obj  = np.zeros((0, nOutput))

    x_new = []
    y_new = []

    gen_indexes = []
    gen_indexes.append(np.zeros((x_initial.shape[0],),dtype=np.uint32))

    for generation in range(gen):
        
        if (termination is not None) and optimizer.should_stop():
            break

        x_gen, offspring_gen, ind_gen = optimizer.generate()

        y_gen = optimizer.evaluate(x_gen)
            
        # Update w/ evaluation values.
        optimizer.update(x_gen, y_gen, offspring_gen, ind_gen)

        x_new.append(x_gen)
        y_new.append(y_gen)

        population_parm = np.vstack((population_parm, x_gen))
        population_obj  = np.vstack((population_obj, y_gen))
        population_parm, population_obj = remove_duplicates(population_parm, population_obj)
        population_parm, population_obj, rank = \
            remove_worst(population_parm, population_obj, pop, nInput, nOutput)

        gen_indexes.append(np.ones((x_gen.shape[0],),dtype=np.uint32)*generation)

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
                 sigma: float,
                 bounds: Optional[np.ndarray] = None,
                 local_random: Optional[np.random.Generator] = None,
                 **params):

        if local_random is None:
            local_random = default_rng()

        self.indicator = Hypervolume
            
        population_size = x.shape[0]
        nInput = x.shape[1]
        nOutput = y.shape[1]
        
        self.parents_x = x
        self.parents_y = y

        self.dim = nInput

        # Selection
        self.mu = params.get("mu", population_size)
        self.lambda_ = params.get("lambda_", 1)

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

        
    def generate(self, ind_init):
        """Generate a population of :math:`\lambda` individuals of type
        *ind_init* from the current strategy.
        :param ind_init: A function object that is able to initialize an
                         individual from a list.
        :returns: A list of individuals with a private attribute :attr:`_ps`.
                  This last attribute is essential to the update function, it
                  indicates that the individual is an offspring and the index
                  of its parent.
        """

        
        arz = self.local_random.normal(self.lambda_, self.dim)
        individuals = list()
        
        inds = []
        
        # Each parent produce an offspring
        if self.lambda_ == self.mu:
            for i in range(self.lambda_):
                individuals.append(ind_init(self.parents[i] + self.sigmas[i] * np.dot(self.A[i], arz[i])))
                inds.append(i)
        # Parents producing an offspring are chosen at random from the first front
        else:
            rank = sortMO(self.parents_x, self.parents_y)
            front_1 = np.argwhere(rank == 0).ravel()
            for i in range(self.lambda_):
                j = self.local_random.integers(0, len(front_1))
                p_idx = self.parent_inds[front_1[j]]
                individuals.append(ind_init(self.parents[p_idx] + self.sigmas[p_idx] * np.dot(self.A[p_idx], arz[i])))
                inds.append(p_idx)

        x_new = np.vstack(individuals)
        ind_array = np.asarray(inds, dtype=np.int)
        return x_new, ind_array


    def _select(self, candidates_x, candidates_y, candidates_ps, candidates_inds):
        
        if candidates_x.shape[0] <= self.mu:
            return np.ones_like(candidates_inds, dtype=np.bool), np.zeros_like(candidates_inds, dtype=np.bool)

        rank = sortMO(candidates_x, candidates_y)

        chosen = np.zeros_like(candidates_inds, dtype=np.bool)
        not_chosen = np.zeros_like(candidates_inds, dtype=np.bool)
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
                chosen_count += len(chosen[-1])
            elif mid_front is None and chosen_count < self.mu:
                mid_front = front_r.tolist()
                # With this front, we selected enough individuals
                full = True
            else:
                not_chosen[front_r] = True

        chosen = np.concatenate(chosen)
        not_chosen = np.concatenate(not_chosen)
                
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

    def _rankOneUpdate(self, invCholesky, A, alpha, beta, v):
        w = np.dot(invCholesky, v)

        # Under this threshold, the update is mostly noise
        if w.max() > 1e-20:
            w_inv = np.dot(w, invCholesky)
            norm_w2 = np.sum(w ** 2)
            a = sqrt(alpha)
            root = np.sqrt(1 + beta / alpha * norm_w2)
            b = a / norm_w2 * (root - 1)

            A = a * A + b * np.outer(v, w)
            invCholesky = 1.0 / a * invCholesky - b / (a ** 2 + a * b * norm_w2) * np.outer(w, w_inv)

        return invCholesky, A

    def update(self, x, y, inds):
        """Update the current covariance matrix strategies from the population.
        :param x:
        :param y:
        """

        # Make sure every parent has an offspring tag and index
        P = self.parents_x.shape[0]
        parent_offspring = [False]*P
        parent_inds = np.asarray(range(P), dtype=np.int)

        C = x.shape[0]
        candidates_x = np.vstack((x, self.parents_x))
        candidates_y = np.vstack((y, self.parents_y))
        candidates_offspring = np.vstack((np.asarray([True]*C, dtype=np.bool), np.asarray([False]*P, dtype=np.bool)))
        candidates_inds = np.vstack((inds, self.parent_inds))
        chosen_ind, not_chosen_ind = self._select(candidates_x, candidates_y, candidates_offspring, candidates_inds)

        cp, cc, ccov = self.cp, self.cc, self.ccov
        d, ptarg, pthresh = self.d, self.ptarg, self.pthresh

        # Make copies for chosen offspring only
        chosen_offspring_inds = inds[np.logical_and(chosen, candidates_offspring)]
        last_steps = self.sigmas[chosen_offspring_inds]
        sigmas = self.sigmas[chosen_offspring_inds]
        invCholesky = self.invCholesky[chosen_offspring_inds].copy()
        A = self.A[chosen_offspring_inds].copy()
        pc = self.pc[chosen_offspring_inds].copy()
        psucc = self.psucc[chosen_offspring_inds].copy()

        # Update the internal parameters for successful offspring
        for i, ind in enumerate(np.nonzero(chosen)[0]):
            
            is_offspring = candidates_offspring[ind]
            p_idx = inds[ind]

            # Only the offspring update the parameter set
            if is_offspring:
                # Update (Success = 1 since it is chosen)
                psucc[i] = (1.0 - cp) * psucc[i] + cp
                sigmas[i] = sigmas[i] * exp((psucc[i] - ptarg) / (d * (1.0 - ptarg)))

                if psucc[i] < pthresh:
                    xp = np.array(ind)
                    x = np.array(self.parents_x[p_idx])
                    pc[i] = (1.0 - cc) * pc[i] + sqrt(cc * (2.0 - cc)) * (xp - x) / last_steps[i]
                    invCholesky[i], A[i] = self._rankOneUpdate(invCholesky[i], A[i], 1 - ccov, ccov, pc[i])
                else:
                    pc[i] = (1.0 - cc) * pc[i]
                    pc_weight = cc * (2.0 - cc)
                    invCholesky[i], A[i] = self._rankOneUpdate(invCholesky[i], A[i], 1 - ccov + pc_weight, ccov, pc[i])

                self.psucc[p_idx] = (1.0 - cp) * self.psucc[p_idx] + cp
                self.sigmas[p_idx] = self.sigmas[p_idx] * exp((self.psucc[p_idx] - ptarg) / (d * (1.0 - ptarg)))

        # It is unnecessary to update the entire parameter set for not chosen individuals
        # Their parameters will not make it to the next generation
        for ind in np.nonzero(not_chosen)[0]:
            
            is_offspring = candidates_offspring[ind]
            p_idx = inds[ind]

            # Only the offspring update the parameter set
            if is_offspring:
                self.psucc[p_idx] = (1.0 - cp) * self.psucc[p_idx]
                self.sigmas[p_idx] = self.sigmas[p_idx] * exp((self.psucc[p_idx] - ptarg) / (d * (1.0 - ptarg)))

        # Make a copy of the internal parameters
        # The parameter is in the temporary variable for offspring and in the original one for parents
        self.parents_x = candidates_x[chosen]
        self.parents_y = candidates_y[chosen]
        self.parents_is_offspring = candidates_offspring[chosen]
        self.parents_inds = candidates_inds[chosen]
        
        self.sigmas = np.where(candidates_offspring, sigmas[chosen], self.sigmas[candidates_inds[chosen]])
        self.invCholesky = np.where(candidates_offspring, invCholesky[chosen], self.invCholesky[candidates_inds[chosen]])
        self.A = np.where(candidates_offspring, A[chosen], self.A[candidates_inds[chosen]])
        self.pc = np.where(candidates_offspring, pc[chosen], self.pc[candidates_inds[chosen]])
        self.psucc = np.where(candidates_offspring, psucc[chosen], self.psucc[candidates_inds[chosen]])


    
        
