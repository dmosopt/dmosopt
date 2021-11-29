# Nondominated Sorting Genetic Algorithm II (NSGA-II)
# An multi-objective optimization algorithm

import sys, gc
import numpy as np
from functools import reduce
import copy, itertools
from dmosopt import sampling
from dmosopt.datatypes import OptHistory
from dmosopt.dda import dda_non_dominated_sort
from dmosopt.MOEA import crossover_sbx, crossover_sbx_feasibility_selection, mutation, feasibility_selection, tournament_selection, sortMO, remove_worst


def optimization(model, nInput, nOutput, xlb, xub, initial=None, feasibility_model=None, termination=None,
                 distance_metric=None, pop=100, gen=100, crossover_rate = 0.5, mutation_rate = 0.05,
                 di_crossover=1., di_mutation=20., logger=None):
    ''' Nondominated Sorting Genetic Algorithm II, An multi-objective algorithm
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
    poolsize = int(round(pop/2.)); # size of mating pool;
    toursize = 2;                  # tournament size;

    if mutation_rate is None:
        mutation_rate = 1. / float(nInput)

    x_initial, y_initial = None, None
    if initial is not None:
        x_initial, y_initial = initial
        
    x = sampling.lh(pop, nInput)
    x = x * (xub - xlb) + xlb
    if x_initial is not None:
        x = np.vstack((x_initial, x))
    
    y = np.zeros((pop, nOutput))
    for i in range(pop):
        y[i,:] = model.evaluate(x[i,:])
    if y_initial is not None:
        y = np.vstack((y_initial, y))
        
    x, y, rank, crowd = sortMO(x, y, nInput, nOutput, distance_metric=distance_metric)
    population_para = x[:pop]
    population_obj  = y[:pop]

    nchildren=1
    if feasibility_model is not None:
        nchildren = poolsize

    x_new = []
    y_new = []

    n_eval = 0
    it = range(gen)
    if termination is not None:
        it = itertools.count()
    for i in it:
        if termination is not None:
            opt = OptHistory(i, n_eval, population_para, population_obj, None)
            if termination.has_terminated(opt):
                break
        if logger is not None:
            if termination is not None:
                logger.info(f"NSGA2: generation {i+1}...")
            else:
                logger.info(f"NSGA2: generation {i+1} of {gen}...")
        pool = tournament_selection(population_para, population_obj, pop, poolsize, toursize, rank)
        count = 0
        xs_gen = []
        while (count < pop - 1):
            if (np.random.rand() < crossover_rate):
                parentidx = np.random.choice(poolsize, 2, replace = False)
                parent1   = pool[parentidx[0],:]
                parent2   = pool[parentidx[1],:]
                children1, children2 = crossover_sbx(parent1, parent2, di_crossover, xlb, xub, nchildren=nchildren)
                if feasibility_model is None:
                    child1 = children1[0]
                    child2 = children2[0]
                else:
                    child1, child2 = crossover_sbx_feasibility_selection(feasibility_model, [children1, children2], logger=logger)
                xs_gen.extend([child1, child2])
                count += 2
            else:
                parentidx = np.random.randint(poolsize)
                parent    = pool[parentidx,:]
                children  = mutation(parent, mutation_rate, di_mutation, xlb, xub, nchildren=nchildren)
                if feasibility_model is None:
                    child = children[0]
                else:
                    child = feasibility_selection(feasibility_model, children, logger=logger)
                xs_gen.append(child)
                count += 1
        x_gen = np.vstack(xs_gen)
        y_gen = model.evaluate(x_gen)
        x_new.append(x_gen)
        y_new.append(y_gen)
        population_para = np.vstack((population_para, x_gen))
        population_obj  = np.vstack((population_obj, y_gen))
        population_para, population_obj, rank = \
            remove_worst(population_para, population_obj, pop, nInput, nOutput, distance_metric=distance_metric)
        gc.collect()
        n_eval += count
            
    bestx = population_para.copy()
    besty = population_obj.copy()

    x = np.vstack([x] + x_new)
    y = np.vstack([y] + y_new)
        
    return bestx, besty, x, y




