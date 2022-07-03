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
from typing import Any, Dict, List, Tuple, Optional, cast
from dmosopt.MOEA import sortMO, remove_worst, remove_duplicates

_EPS = 1e-8
_MEAN_MAX = 1e32
_SIGMA_MAX = 1e32

def optimization(model, nInput, nOutput, xlb, xub, initial=None, gen=100,
                 pop=100, mean=None, sigma=1.3, sampling_method=None, termination=None,
                 local_random=None, logger=None, **kwargs):
    ''' CMA-ES

        model: the evaluated model function
        nInput: number of model parameters
        nOutput: number of output objectives
        xlb: lower bound of input
        xub: upper bound of input
        pop: number of population
        sampling_method: optional callable for initial sampling of parameters
    '''
    
    if local_random is None:
        local_random = default_rng()

    x_initial, y_initial = None, None
    if initial is not None:
        x_initial, y_initial = initial

    if mean is None:
        mean = np.zeros(nInput)

    bounds = np.column_stack((xlb, xub))
    optimizer = CMAES(mu=mu, sigma=sigma, population_size=pop,
                      bounds=bounds, local_random=local_random)

    population_parm = np.zeros((0, nInput))
    population_obj  = np.zeros((0, nOutput))

    x_new = []
    y_new = []

    gen_indexes = []
    gen_indexes.append(np.zeros((x_initial.shape[0],),dtype=np.uint32))

    for generation in range(gen):
        
        if (termination is not None) and optimizer.should_stop():
            break

        x_gen = np.zeros((pop, nInput))
        solutions = []
        for i in range(optimizer.population_size):
            x = optimizer.generate()
            x_gen[i] = x 

        y_gen = model.evaluate(x_gen)

        for i in range(optimizer.population_size):
            solutions.append((x_gen[i], y_gen[i]))
            
        # Update w/ evaluation values.
        optimizer.update(solutions)

        x_new.append(x_gen)
        y_new.append(y_gen)

        population_parm = np.vstack((population_parm, x_gen))
        population_obj  = np.vstack((population_obj, y_gen))
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
    """Multiobjective CMA-ES optimization class based on the paper [Voss2010]_ 
    with generate-and-update interface.

    :param population: An initial population of individuals.
    :param sigma: The initial step size of the complete system.
    :param mu: The number of parents to use in the evolution. When not
               provided it defaults to the length of *population*. (optional)
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
    .. [Voss2010] Voss, Hansen, Igel, "Improved Step Size Adaptation
       for the MO-CMA-ES", 2010.
    
    """

    def __init__(self,
                 population: np.ndarray,
                 nInput: int,
                 sigma: float,
                 bounds: Optional[np.ndarray] = None,
                 local_random: Optional[np.random.Generator] = None,
                 **params):

        if local_random is None:
            local_random = default_rng()

        population_size = population.shape[0]
        self.parents = population
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
        self.sigmas = [sigma] * population_size
        # Lower Cholesky matrix (Sampling matrix)
        self.A = [np.identity(self.dim) for _ in range(population_size)]
        # Inverse Cholesky matrix (Used in the update of A)
        self.invCholesky = [np.identity(self.dim) for _ in range(population_size)]
        self.pc = [np.zeros(self.dim) for _ in range(population_size)]
        self.psucc = [self.ptarg] * population_size

        
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

        # Make sure every parent has a parent tag and index
        for i, p in enumerate(self.parents):
            p._ps = "p", i

        # Each parent produce an offspring
        if self.lambda_ == self.mu:
            for i in range(self.lambda_):
                individuals.append(ind_init(self.parents[i] + self.sigmas[i] * np.dot(self.A[i], arz[i])))
                individuals[-1]._ps = "o", i
        # Parents producing an offspring are chosen at random from the first front
        else:
            ndom = sortMO(self.parents, len(self.parents), first_front_only=True)
            for i in range(self.lambda_):
                j = self.local_random.integers(0, len(ndom))
                _, p_idx = ndom[j]._ps
                individuals.append(ind_init(self.parents[p_idx] + self.sigmas[p_idx] * np.dot(self.A[p_idx], arz[i])))
                individuals[-1]._ps = "o", p_idx

        return individuals


    def _select(self, candidates):
        
        if len(candidates) <= self.mu:
            return candidates, []

        pareto_fronts = sortMO(candidates, len(candidates))

        chosen = list()
        mid_front = None
        not_chosen = list()

        # Fill the next population (chosen) with the fronts until there is not enough space
        # When an entire front does not fit in the space left we rely on the hypervolume
        # for this front
        # The remaining fronts are explicitly not chosen
        full = False
        for front in pareto_fronts:
            if len(chosen) + len(front) <= self.mu and not full:
                chosen += front
            elif mid_front is None and len(chosen) < self.mu:
                mid_front = front
                # With this front, we selected enough individuals
                full = True
            else:
                not_chosen += front

        # Separate the mid front to accept only k individuals
        k = self.mu - len(chosen)
        if k > 0:
            # reference point is chosen in the complete population
            # as the worst in each dimension +1
            ref = np.array([ind.fitness.wvalues for ind in candidates]) * -1
            ref = np.max(ref, axis=0) + 1

            for _ in range(len(mid_front) - k):
                idx = self.indicator(mid_front, ref=ref)
                not_chosen.append(mid_front.pop(idx))

            chosen += mid_front

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

    def update(self, population):
        """Update the current covariance matrix strategies from the
        *population*.
        :param population: A list of individuals from which to update the
                           parameters.
        """
        chosen, not_chosen = self._select(population + self.parents)

        cp, cc, ccov = self.cp, self.cc, self.ccov
        d, ptarg, pthresh = self.d, self.ptarg, self.pthresh

        # Make copies for chosen offspring only
        last_steps = [self.sigmas[ind._ps[1]] if ind._ps[0] == "o" else None for ind in chosen]
        sigmas = [self.sigmas[ind._ps[1]] if ind._ps[0] == "o" else None for ind in chosen]
        invCholesky = [self.invCholesky[ind._ps[1]].copy() if ind._ps[0] == "o" else None for ind in chosen]
        A = [self.A[ind._ps[1]].copy() if ind._ps[0] == "o" else None for ind in chosen]
        pc = [self.pc[ind._ps[1]].copy() if ind._ps[0] == "o" else None for ind in chosen]
        psucc = [self.psucc[ind._ps[1]] if ind._ps[0] == "o" else None for ind in chosen]

        # Update the internal parameters for successful offspring
        for i, ind in enumerate(chosen):
            t, p_idx = ind._ps

            # Only the offspring update the parameter set
            if t == "o":
                # Update (Success = 1 since it is chosen)
                psucc[i] = (1.0 - cp) * psucc[i] + cp
                sigmas[i] = sigmas[i] * exp((psucc[i] - ptarg) / (d * (1.0 - ptarg)))

                if psucc[i] < pthresh:
                    xp = np.array(ind)
                    x = np.array(self.parents[p_idx])
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
        for ind in not_chosen:
            t, p_idx = ind._ps

            # Only the offspring update the parameter set
            if t == "o":
                self.psucc[p_idx] = (1.0 - cp) * self.psucc[p_idx]
                self.sigmas[p_idx] = self.sigmas[p_idx] * exp((self.psucc[p_idx] - ptarg) / (d * (1.0 - ptarg)))

        # Make a copy of the internal parameters
        # The parameter is in the temporary variable for offspring and in the original one for parents
        self.parents = chosen
        self.sigmas = [sigmas[i] if ind._ps[0] == "o" else self.sigmas[ind._ps[1]] for i, ind in enumerate(chosen)]
        self.invCholesky = [invCholesky[i] if ind._ps[0] == "o" else self.invCholesky[ind._ps[1]] for i, ind in enumerate(chosen)]
        self.A = [A[i] if ind._ps[0] == "o" else self.A[ind._ps[1]] for i, ind in enumerate(chosen)]
        self.pc = [pc[i] if ind._ps[0] == "o" else self.pc[ind._ps[1]] for i, ind in enumerate(chosen)]
        self.psucc = [psucc[i] if ind._ps[0] == "o" else self.psucc[ind._ps[1]] for i, ind in enumerate(chosen)]


    
        
