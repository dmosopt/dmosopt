# Trust Region Search, multi-objective local optimization algorithm.

import gc, itertools
import numpy as np
from numpy.random import default_rng
from dmosopt.datatypes import OptHistory
from dmosopt.MOEA import (
    sortMO,
    remove_worst,
    remove_duplicates,
)


def optimization(
    model,
    nInput,
    nOutput,
    xlb,
    xub,
    initial=None,
    pop=100,
    gen=100,
    sampling_method=None,
    local_random=None,
    logger=None,
    **kwargs,
):
    """Trust Region Search

    model: the evaluated model function
    nInput: number of model input
    nOutput: number of output objectives
    xlb: lower bound of input
    xub: upper bound of input
    pop: number of population
    gen: number of generation
    """

    if local_random is None:
        local_random = default_rng()

    pop_slices = list([range(p * pop, (p + 1) * pop) for p in range(init_size)])

    x_initial, y_initial = None, None
    if initial is not None:
        x_initial, y_initial = initial

    xs = []
    ys = []
    for sl in pop_slices:
        if sampling_method is None:
            x = sampling.lh(pop, nInput, local_random)
            x_s = x * (xub - xlb) + xlb
        elif sampling_method == "sobol":
            x = sampling.sobol(pop, nInput, local_random)
            x_s = x * (xub - xlb) + xlb
        elif callable(sampling_method):
            sampling_method_params = kwargs.get("sampling_method_params", None)
            if sampling_method_params is None:
                x_s = sampling_method(local_random, pop, nInput, xlb, xub)
            else:
                x_s = sampling_method(local_random, **sampling_method_params)
        else:
            raise RuntimeError(f"Unknown sampling method {sampling_method}")
        x_s = x_s.astype(np.float32)
        y_s = model.evaluate(x_s).astype(np.float32)
        if x_initial is not None:
            x_s = np.vstack((x_initial.astype(np.float32), x_s))
        if y_initial is not None:
            y_s = np.vstack((y_initial.astype(np.float32), y_s))
        xs.append(x_s)
        ys.append(y_s)

        self._idx = np.vstack((self._idx, i * np.ones((self.n_init, 1), dtype=int)))
        
    population_parm = np.zeros((init_size * pop, nInput), dtype=np.float32)
    population_obj = np.zeros((init_size * pop, nOutput), dtype=np.float32)
    
    ranks = []
    for p, sl in enumerate(pop_slices):
        xs[p], ys[p], rank_p, _ = sortMO(
            xs[p],
            ys[p],
            nInput,
            nOutput,
            x_distance_metrics=x_distance_metrics,
            y_distance_metrics=y_distance_metrics,
        )
        population_parm[sl] = xs[p][:pop]
        population_obj[sl] = ys[p][:pop]
        ranks.append(rank_p)
    
    

    return bestx, besty, gen_index, x, y
