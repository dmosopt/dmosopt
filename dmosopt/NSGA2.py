# Nondominated Sorting Genetic Algorithm II (NSGA-II)
# A multi-objective optimization algorithm.

import gc, itertools
import numpy as np
from numpy.random import default_rng
from dmosopt import sampling
from dmosopt.datatypes import OptHistory
from dmosopt.dda import dda_non_dominated_sort
from dmosopt.MOEA import crossover_sbx, mutation, tournament_selection, sortMO, remove_worst, remove_duplicates


def optimization(model, nInput, nOutput, xlb, xub, initial=None, feasibility_model=None, termination=None,
                 distance_metric=None, pop=100, gen=100, crossover_prob = 0.9, mutation_prob = 0.1, mutation_rate = None, nchildren=1,
                 di_crossover=1., di_mutation=20., sampling_method=None, local_random=None, logger=None, **kwargs):
    ''' Nondominated Sorting Genetic Algorithm II

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
        sampling_method: optional callable for initial sampling of parameters
    '''
    
    if local_random is None:
        local_random = default_rng()

    y_distance_metrics = []
    y_distance_metrics.append(distance_metric)
    x_distance_metrics = None
    if feasibility_model is not None:
        x_distance_metrics = [feasibility_model.rank]

    if np.isscalar(di_crossover):
        di_crossover = np.asarray([di_crossover]*nInput)
    if np.isscalar(di_mutation):
        di_mutation = np.asarray([di_mutation]*nInput)
        
    poolsize = int(round(pop/2.)); # size of mating pool;
    toursize = 2;                  # tournament size;

    if mutation_rate is None:
        mutation_rate = 1. / float(nInput)

    x_initial, y_initial = None, None
    if initial is not None:
        x_initial, y_initial = initial

    if sampling_method is None:
        x = sampling.lh(pop, nInput, local_random)
        x = x * (xub - xlb) + xlb
    elif callable(sampling_method):
        sampling_method_params = kwargs.get('sampling_method_params', None)
        if sampling_method_params is None:
            x = sampling_method(local_random, pop, nInput, xlb, xub)
        else:
            x = sampling_method(local_random, **sampling_method_params)
    else:
        raise RuntimeError(f'Unknown sampling method {sampling_method}')
    if x_initial is not None:
        x = np.vstack((x_initial, x))
    
    y = np.zeros((pop, nOutput))
    for i in range(pop):
        y[i,:] = model.evaluate(x[i,:])
    if y_initial is not None:
        y = np.vstack((y_initial, y))

    x, y, rank, _ = sortMO(x, y, nInput, nOutput,
                           x_distance_metrics=x_distance_metrics,
                           y_distance_metrics=y_distance_metrics)
    population_parm = x[:pop]
    population_obj  = y[:pop]

    gen_indexes = []
    gen_indexes.append(np.zeros((x.shape[0],),dtype=np.uint32))

    x_new = []
    y_new = []

    n_eval = 0
    it = range(1, gen+1)
    if termination is not None:
        it = itertools.count(1)
    for i in it:
        if termination is not None:
            opt = OptHistory(i, n_eval, population_parm, population_obj, None)
            if termination.has_terminated(opt):
                break
        if logger is not None:
            if termination is not None:
                logger.info(f"NSGA2: generation {i}...")
            else:
                logger.info(f"NSGA2: generation {i} of {gen}...")
        pool_idxs = tournament_selection(local_random, pop, poolsize, toursize, rank)
        pool = population_parm[pool_idxs,:]
        count = 0
        xs_gen = []
        while (count < pop - 1):
            if (local_random.random() < crossover_prob):
                parentidx = local_random.choice(poolsize, 2, replace = False)
                parent1   = pool[parentidx[0],:]
                parent2   = pool[parentidx[1],:]
                children1, children2 = crossover_sbx(local_random, parent1, parent2, di_crossover, xlb, xub, nchildren=nchildren)
                child1 = children1[0]
                child2 = children2[0]
                xs_gen.extend([child1, child2])
                count += 2
            if (local_random.random() < mutation_prob):
                parentidx = local_random.integers(low=0, high=poolsize)
                parent    = pool[parentidx,:]
                children  = mutation(local_random, parent, di_mutation, xlb, xub, mutation_rate=mutation_rate, nchildren=nchildren)
                child     = children[0]
                xs_gen.append(child)
                count += 1
        x_gen = np.vstack(xs_gen)
        y_gen = model.evaluate(x_gen)
        x_new.append(x_gen)
        y_new.append(y_gen)
        gen_indexes.append(np.ones((x_gen.shape[0],),dtype=np.uint32)*i)

        population_parm = np.vstack((population_parm, x_gen))
        population_obj  = np.vstack((population_obj, y_gen))
        population_parm, population_obj = remove_duplicates(population_parm, population_obj)
        population_parm, population_obj, rank = \
            remove_worst(population_parm, population_obj, pop, nInput, nOutput,
                         x_distance_metrics=x_distance_metrics,
                         y_distance_metrics=y_distance_metrics)
        gc.collect()
        n_eval += count
            
    bestx = population_parm.copy()
    besty = population_obj.copy()

    gen_index = np.concatenate(gen_indexes)
    x = np.vstack([x] + x_new)
    y = np.vstack([y] + y_new)
    
    return bestx, besty, gen_index, x, y




