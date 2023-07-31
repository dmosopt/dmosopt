###
### Multiobjective CMA-ES optimization class based on the papers
### "Efficient Covariance Matrix Update for Variable Metric Evolution Strategies", Suttorp, Hansen, Igel; 2009.
### "Improved Step Size Adaptation for the MO-CMA-ES", Voss, Hansen, Igel; 2010.
###
### Based on code from:
### https://github.com/DEAP/deap/blob/master/deap/cma.py
###

from functools import partial
import numpy as np
from dmosopt.dda import dda_non_dominated_sort
from dmosopt.MOEA import (
    Struct,
    MOEA,
    remove_worst,
    remove_duplicates,
)
from dmosopt.indicators import Hypervolume
from typing import Any, Union, Dict, List, Tuple, Optional


class CMAES(MOEA):
    def __init__(
        self,
        popsize: int,
        nInput: int,
        nOutput: int,
        feasibility_model: Optional[Any],
        **kwargs,
    ):
        """
            Multiobjective Covariance Matrix-Assisted Evolutionary Strategy (CMAES) optimization.
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
        | ``ptarg``      | ``1.0 / (5 + 1.0 / 2.0)`` | Target success rate.       |
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
        super().__init__(
            name="CMAES",
            popsize=popsize,
            nInput=nInput,
            nOutput=nOutput,
            **kwargs,
        )

        self.feasibility_model = feasibility_model

        self.x_distance_metrics = None
        if self.feasibility_model is not None:
            self.x_distance_metrics = [self.feasibility_model.rank]

        di_mutation = self.opt_params.di_mutation
        if np.isscalar(di_mutation):
            self.opt_params.di_mutation = np.asarray([di_mutation] * nInput)

        self.feasibility_model = feasibility_model
        self.state = None
        self.indicator = Hypervolume

    @property
    def default_parameters(self) -> Dict[str, Any]:
        """Returns default parameters of CMAES strategy."""

        nInput = self.nInput
        nOutput = self.nOutput
        popsize = self.popsize

        # Selection
        sigma = 0.005
        mu = popsize // 2
        lambda_ = 1

        # Step size control
        d = 1.0 + nOutput / 2.0
        ptarg = 1.0 / (5.0 + 0.5)
        cp = ptarg / (1.0 + ptarg)

        # Covariance matrix adaptation
        cc = 2.0 / (nInput + 2.0)
        ccov = 2.0 / (nInput**2 + 6.0)
        pthresh = 0.44

        params = {
            "sigma": sigma,
            "mu": mu,
            "lambda_": lambda_,
            "d": d,
            "ptarg": ptarg,
            "cp": cp,
            "cc": cc,
            "ccov": ccov,
            "pthresh": pthresh,
            "di_mutation": 1.0,
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
        dim = self.nInput
        population_size = self.popsize
        sigma = self.opt_params.sigma
        di_mutation = self.opt_params.di_mutation
        ptarg = self.opt_params.ptarg

        # Internal parameters associated to the mu parent
        sigmas = np.asarray([sigma * (1.0 / (di_mutation + 1.0))] * population_size)

        # Lower Cholesky matrix (Sampling matrix)
        A = np.stack([np.identity(dim) for _ in range(population_size)])

        # Inverse Cholesky matrix (Used in the update of A)
        Ainv = np.stack([np.identity(dim) for _ in range(population_size)])
        pc = np.stack([np.zeros(dim) for _ in range(population_size)])
        psucc = np.asarray([ptarg] * population_size)

        order, rank = sortMO(x, y, self.x_distance_metrics)
        sorted_rank_idxs = order[:population_size]
        parents_x = x[sorted_rank_idxs].copy()
        parents_y = y[sorted_rank_idxs].copy()

        state = Struct(
            bounds=bounds,
            parents_x=parents_x,
            parents_y=parents_y,
            sigmas=sigmas,
            A=A,
            Ainv=Ainv,
            pc=pc,
            psucc=psucc,
        )

        return state

    def _select(self, candidates_x, candidates_y, candidates_ps, candidates_inds):
        popsize = self.popsize

        if candidates_x.shape[0] <= popsize:
            return np.ones_like(candidates_inds, dtype=bool_), np.zeros_like(
                candidates_inds, dtype=bool
            )

        order, rank = sortMO(candidates_x, candidates_y, self.x_distance_metrics)

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
            if chosen_count + len(front_r) <= popsize and not full:
                chosen[front_r] = True
                chosen_count += len(front_r)
            elif mid_front is None and chosen_count < popsize:
                mid_front = front_r.copy()
                # With this front, we selected enough individuals
                full = True
            else:
                not_chosen[front_r] = True

        # Separate the mid front to accept only k individuals
        k = popsize - chosen_count
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

    def generate_strategy(self, **params):
        """
        Generates a population of :math:`\lambda` individuals.
        :returns: A list of individuals.
        """

        local_random = self.local_random
        dim = self.nInput
        mu = self.opt_params.mu
        lambda_ = self.opt_params.lambda_

        parents_x = self.state.parents_x
        parents_y = self.state.parents_y
        sigmas = self.state.sigmas
        A = self.state.A

        arz = local_random.normal(size=(lambda_ * mu, dim))
        individuals = None

        order, rank = sortMO(parents_x, parents_y, self.x_distance_metrics)
        # mu_front = order[:mu]
        parent_selection = []
        rmax = int(np.max(rank))
        selection_count = 0
        for r in range(rmax + 1):
            front_r = np.argwhere(rank == r).ravel()
            parent_selection.append(front_r)
            selection_count += len(front_r)
            if selection_count >= mu:
                break

        parent_selection = np.concatenate(parent_selection)[:mu]
        js = local_random.choice(len(parent_selection), size=lambda_ * mu)
        p_idx_array = parent_selection[js]
        individuals = parents_x[p_idx_array] + sigmas[p_idx_array] * np.einsum(
            "ijk,ik->ij", A[p_idx_array], arz
        )

        xrng = self.bounds[:, 1] - self.bounds[:, 0]
        x_new = (individuals / np.max(np.abs(individuals))) * xrng + self.bounds[:, 0]
        return x_new, {"p_idx": p_idx_array}

    def update_strategy(
        self,
        x_gen: np.ndarray,
        y_gen: np.ndarray,
        state: Dict[Any, Any],
        **params,
    ):
        """Update the current covariance matrix strategies from the population.
        :param x:
        :param y:
        """

        dim = self.nInput

        p_idxs = state["p_idx"]

        xlb = self.bounds[:, 0]
        xub = self.bounds[:, 1]

        parents_x = self.state.parents_x
        parents_y = self.state.parents_y

        # Every parent i gets assigned a "parent index" of i
        P = parents_x.shape[0]
        parent_pidxs = np.asarray(range(P), dtype=np.int_)

        C = x_gen.shape[0]
        candidates_x = np.vstack((x_gen, parents_x))
        candidates_y = np.vstack((y_gen, parents_y))
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

        cp, cc, ccov = self.opt_params.cp, self.opt_params.cc, self.opt_params.ccov
        d, ptarg, pthresh = (
            self.opt_params.d,
            self.opt_params.ptarg,
            self.opt_params.pthresh,
        )

        # Make copies for chosen offspring only
        chosen_offspring = np.logical_and(chosen, candidates_offspring)
        sigmas = np.where(
            chosen_offspring.reshape((-1, 1)),
            self.state.sigmas[candidates_pidxs],
            np.full((candidates_pidxs.shape[0], dim), np.nan),
        )
        last_steps = sigmas.copy()
        Ainv = np.where(
            chosen_offspring.reshape((-1, 1, 1)),
            self.state.Ainv[candidates_pidxs],
            np.nan,
        )
        A = np.where(
            chosen_offspring.reshape((-1, 1, 1)), self.state.A[candidates_pidxs], np.nan
        )
        pc = np.where(
            chosen_offspring.reshape((-1, 1)), self.state.pc[candidates_pidxs], np.nan
        )
        psucc = np.where(
            chosen_offspring,
            self.state.psucc[candidates_pidxs],
            np.nan,
        )

        # Update the internal parameters for successful offspring
        for ind in np.nonzero(chosen)[0]:
            is_offspring = candidates_offspring[ind]
            p_idx = candidates_pidxs[ind]

            # Only the offspring update the parameter set
            if is_offspring:
                # Update (Success = 1 since it is chosen)
                psucc[ind] = (1.0 - cp) * psucc[ind] + cp
                sigma_factors = np.exp((psucc[ind] - ptarg) / (d * (1.0 - ptarg)))
                sigmas[ind] = sigmas[ind] * sigma_factors

                xp = candidates_x[ind]
                x = parents_x[p_idx]
                z = np.divide(xp - x, xub - xlb) / last_steps[ind]
                A[ind], Ainv[ind], pc[ind] = updateCholesky(
                    A[ind], Ainv[ind], z, psucc[ind], pc[ind], cc, ccov, pthresh
                )

                self.state.psucc[p_idx] = (1.0 - cp) * self.state.psucc[p_idx] + cp
                psigma_factors = np.exp(
                    (self.state.psucc[p_idx] - ptarg) / (d * (1.0 - ptarg))
                )
                self.state.sigmas[p_idx] = self.state.sigmas[p_idx] * psigma_factors

        # Update the entire parameter set for not chosen individuals
        for ind in np.nonzero(not_chosen)[0]:
            is_offspring = candidates_offspring[ind]
            p_idx = candidates_pidxs[ind]

            # Only the offspring update the parameter set
            if is_offspring:
                self.state.psucc[p_idx] = (1.0 - cp) * self.state.psucc[p_idx]
                self.state.sigmas[p_idx] = self.state.sigmas[p_idx] * np.exp(
                    (self.state.psucc[p_idx] - ptarg) / (d * (1.0 - ptarg))
                )

        self.state.parents_x = candidates_x[chosen]
        self.state.parents_y = candidates_y[chosen]

        self.state.sigmas = np.where(
            candidates_offspring[chosen].reshape((-1, 1)),
            sigmas[chosen],
            self.state.sigmas[candidates_pidxs[chosen]],
        )

        self.state.Ainv = np.where(
            candidates_offspring[chosen].reshape((-1, 1, 1)),
            Ainv[chosen],
            self.state.Ainv[candidates_pidxs[chosen]],
        )
        self.state.A = np.where(
            candidates_offspring[chosen].reshape((-1, 1, 1)),
            A[chosen],
            self.state.A[candidates_pidxs[chosen]],
        )
        self.state.pc = np.where(
            candidates_offspring[chosen].reshape((-1, 1)),
            pc[chosen],
            self.state.pc[candidates_pidxs[chosen]],
        )
        self.state.psucc = np.where(
            candidates_offspring[chosen],
            psucc[chosen],
            self.state.psucc[candidates_pidxs[chosen]],
        )

    def get_population_strategy(self):
        population_parm = self.state.parents_x.copy()
        population_obj = self.state.parents_y.copy()

        population_parm, population_obj = remove_duplicates(
            population_parm, population_obj
        )
        population_parm, population_obj, _ = remove_worst(
            population_parm, population_obj, self.popsize, self.nInput, self.nOutput
        )

        return population_parm, population_obj


def sortMO(
    x,
    y,
    x_distance_metrics=None,
):
    """Non-dominated sort for multi-objective optimization
    x: input parameter matrix
    y: output objectives matrix
    """

    x_distance_functions = []
    if x_distance_metrics is not None:
        for distance_metric in x_distance_metrics:
            if callable(distance_metric):
                x_distance_functions.append(distance_metric)
            else:
                raise RuntimeError(f"sortMO: unknown distance metric {distance_metric}")

    rank = dda_non_dominated_sort(y)

    x_dists = list([np.zeros_like(rank) for _ in x_distance_functions])
    rmax = int(rank.max())
    if len(x_dists) > 0:
        for front in range(rmax + 1):
            rankidx = rank == front
            for i, x_distance_function in enumerate(x_distance_functions):
                D = x_distance_function(x[rankidx, :])
                x_dists[i][rankidx] = D

    perm = np.lexsort((list([-dist for dist in x_dists]) + [rank]))

    return perm, rank


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
