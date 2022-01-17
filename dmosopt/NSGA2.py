# Nondominated Sorting Genetic Algorithm II (NSGA-II)
# A multi-objective optimization algorithm.

import gc, itertools
import numpy as np
from numpy.random import default_rng
from dmosopt import sampling
from dmosopt.datatypes import OptHistory
from dmosopt.dda import dda_non_dominated_sort
from dmosopt.MOEA import crossover_sbx, crossover_sbx_feasibility_selection, mutation, feasibility_selection, tournament_selection, sortMO, remove_worst


def optimization(model, nInput, nOutput, xlb, xub, initial=None, feasibility_model=None, termination=None,
                 distance_metric=None, pop=100, gen=100, crossover_rate = 0.5, mutation_rate = 0.05,
                 di_crossover=1., di_mutation=20., local_random=None, logger=None):
    ''' Nondominated Sorting Genetic Algorithm II

        model: the evaluated model function
        nInput: number of model input
        nOutput: number of output objectives
        xlb: lower bound of input
        xub: upper bound of input
        pop: number of population
        gen: number of generation
        crossover_rate: ratio of crossover in each generation
        mutation_rate: ratio of muration in each generation
        di_crossover: distribution index for crossover
        di_mutation: distribution index for mutation
    '''
    
    if local_random is None:
        local_random = default_rng()

    poolsize = int(round(pop/2.)); # size of mating pool;
    toursize = 2;                  # tournament size;

    if mutation_rate is None:
        mutation_rate = 1. / float(nInput)

    x_initial, y_initial = None, None
    if initial is not None:
        x_initial, y_initial = initial
        
    x = sampling.lh(pop, nInput, local_random)
    x = x * (xub - xlb) + xlb
    if x_initial is not None:
        x = np.vstack((x_initial, x))
    
    y = np.zeros((pop, nOutput))
    for i in range(pop):
        y[i,:] = model.evaluate(x[i,:])
    if y_initial is not None:
        y = np.vstack((y_initial, y))

    x, y, rank, crowd = sortMO(x, y, nInput, nOutput, distance_metric=distance_metric)
    population_parm = x[:pop]
    population_obj  = y[:pop]

    gen_indexes = []
    gen_indexes.append(np.zeros((x.shape[0],),dtype=np.uint32))

    nchildren=1
    if feasibility_model is not None:
        nchildren = poolsize

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
                logger.info(f"NSGA2: generation {i+1} of {gen}...")
        pool_idxs = tournament_selection(local_random, pop, poolsize, toursize, rank)
        pool = population_parm[pool_idxs,:]
        count = 0
        xs_gen = []
        while (count < pop - 1):
            if (local_random.random() < crossover_rate):
                parentidx = local_random.choice(poolsize, 2, replace = False)
                parent1   = pool[parentidx[0],:]
                parent2   = pool[parentidx[1],:]
                children1, children2 = crossover_sbx(local_random, parent1, parent2, di_crossover, xlb, xub, nchildren=nchildren)
                if feasibility_model is None:
                    child1 = children1[0]
                    child2 = children2[0]
                else:
                    child1, child2 = crossover_sbx_feasibility_selection(local_random, feasibility_model, [children1, children2], logger=logger)
                xs_gen.extend([child1, child2])
                count += 2
            else:
                parentidx = local_random.integers(low=0, high=poolsize)
                parent    = pool[parentidx,:]
                children  = mutation(local_random, parent, mutation_rate, di_mutation, xlb, xub, nchildren=nchildren)
                if feasibility_model is None:
                    child = children[0]
                else:
                    child = feasibility_selection(local_random, feasibility_model, children, logger=logger)
                xs_gen.append(child)
                count += 1
        x_gen = np.vstack(xs_gen)
        y_gen = model.evaluate(x_gen)
        x_new.append(x_gen)
        y_new.append(y_gen)
<<<<<<< HEAD
        population_parm = np.vstack((population_parm, x_gen))
=======
        gen_indexes.append(np.ones((x_gen.shape[0],),dtype=np.uint32)*i)

        population_para = np.vstack((population_para, x_gen))
>>>>>>> master
        population_obj  = np.vstack((population_obj, y_gen))
        population_parm, population_obj, rank = \
            remove_worst(population_parm, population_obj, pop, nInput, nOutput, distance_metric=distance_metric)
        gc.collect()
        n_eval += count
<<<<<<< HEAD
            
    bestx = population_parm.copy()
=======
        
    bestx = population_para.copy()
>>>>>>> master
    besty = population_obj.copy()

    gen_index = np.concatenate(gen_indexes)
    x = np.vstack([x] + x_new)
    y = np.vstack([y] + y_new)
    
    return bestx, besty, gen_index, x, y




