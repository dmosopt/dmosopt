# Nondominated Sorting Genetic Algorithm II (NSGA-II)
# An multi-objective optimization algorithm

import sys, gc
import numpy as np
from functools import reduce
import copy, itertools
from dmosopt import sampling
from dmosopt.datatypes import OptHistory

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
                children1, children2 = crossover(parent1, parent2, di_crossover, xlb, xub, nchildren=nchildren)
                if feasibility_model is None:
                    child1 = children1[0]
                    child2 = children2[0]
                else:
                    child1, child2 = crossover_feasibility_selection(feasibility_model, [children1, children2], logger=logger)
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


def crossover_feasibility_selection(feasibility_model, children_list, logger=None):
    child_selection = []
    for children in children_list:
        fsb_pred, fsb_dist, _ = feasibility_model.predict(children)
        all_feasible = np.argwhere(np.all(fsb_pred > 0, axis=1)).ravel()
        if len(all_feasible) > 0:
            fsb_pred = fsb_pred[all_feasible]
            fsb_dist = fsb_dist[all_feasible]
            children = children[all_feasible]
            sum_dist = np.sum(fsb_dist, axis=1)
            child = children[np.argmax(sum_dist)]
        else:
            childidx = np.random.choice(np.arange(children.shape[0]), size=1)
            child = children[childidx[0]]
        child_selection.append(child)
            
    child1 = child_selection[0]
    child2 = child_selection[1]
    return child1, child2


def feasibility_selection(feasibility_model, children, logger=None):
    fsb_pred, fsb_dist, _ = feasibility_model.predict(children)
    all_feasible = np.argwhere(np.all(fsb_pred > 0, axis=1)).ravel()
    if len(all_feasible) > 0:
        fsb_pred = fsb_pred[all_feasible]
        fsb_dist = fsb_dist[all_feasible]
        children = children[all_feasible]
        sum_dist = np.sum(fsb_dist, axis=1)
        child = children[np.argmax(sum_dist)]
    else:
        childidx = np.random.choice(np.arange(children.shape[0]), size=1)
        child = children[childidx[0]]
    return child


def mutation(parent, mutation_rate, di_mutation, xlb, xub, nchildren=1):
    ''' Polynomial Mutation in Genetic Algorithm
        For more information about PMut refer the NSGA-II paper.
        muration_rate: mutation rate
        di_mutation: distribution index for mutation, default = 20
            This determine how well spread the child will be from its parent.
        parent: sample point before mutation
	'''
    n = len(parent)
    children = np.ndarray((nchildren,n))
    delta = np.ndarray((n,))
    for i in range(nchildren):
        u = np.random.rand(n)
        lo = np.argwhere(u < mutation_rate).ravel()
        hi = np.argwhere(u >= mutation_rate).ravel()
        delta[lo] = (2.0*u[lo])**(1.0/(di_mutation+1)) - 1.0
        delta[hi] = 1.0 - (2.0*(1.0 - u[hi]))**(1.0/(di_mutation+1))
        children[i, :] = np.clip(parent + (xub - xlb) * delta, xlb, xub)
    return children


def crossover(parent1, parent2, di_crossover, xlb, xub, nchildren=1):
    ''' SBX (Simulated Binary Crossover) in Genetic Algorithm
         For more information about SBX refer the NSGA-II paper.
         di_crossover: distribution index for crossover, default = 20
         This determine how well spread the children will be from their parents.
    '''
    n = len(parent1)
    children1 = np.ndarray((nchildren, n))
    children2 = np.ndarray((nchildren, n))
    beta = np.ndarray((n,))
    for i in range(nchildren):
        u = np.random.rand(n)
        lo = np.argwhere(u <= 0.5).ravel()
        hi = np.argwhere(u > 0.5).ravel()
        beta[lo] = (2.0*u[lo])**(1.0/(di_crossover+1))
        beta[hi] = (1.0/(2.0*(1.0 - u[hi])))**(1.0/(di_crossover+1))
        children1[i,:] = np.clip(0.5*((1-beta)*parent1 + (1+beta)*parent2), xlb, xub)
        children2[i,:] = np.clip(0.5*((1+beta)*parent1 + (1-beta)*parent2), xlb, xub)
    return children1, children2




def sortMO(x, y, nInput, nOutput, return_perm=False, distance_metric='crowding'):
    ''' Non domination sorting for multi-objective optimization
        x: input parameter matrix
        y: output objectives matrix
        nInput: number of input
        nOutput: number of output
        return_perm: if True, return permutation indices of original input
    '''
    distance_function = crowding_distance
    if distance_metric is not None:
        if distance_metric == 'crowding':
            distance_function = crowding_distance
        elif distance_metric == 'euclidean':
            distance_function = euclidean_distance
        else:
            raise RuntimeError(f'sortMO: unknown distance metric {distance_metric}')
    rank, dom = fast_non_dominated_sort(y)
    idxr = rank.argsort()
    rank = rank[idxr]
    x = x[idxr,:]
    y = y[idxr,:]
    T = x.shape[0]

    crowd = np.zeros(T)
    rmax = int(rank.max())
    idxt = np.zeros(T, dtype = np.int)
    c = 0
    for k in range(rmax):
        rankidx = (rank == k)
        D = distance_function(y[rankidx,:])
        idxd = D.argsort()[::-1]
        crowd[rankidx] = D[idxd]
        idxtt = np.array(range(len(rank)))[rankidx]
        idxt[c:(c+len(idxtt))] = idxtt[idxd]
        c += len(idxtt)
    x = x[idxt,:]
    y = y[idxt,:]
    perm = idxr[idxt] if return_perm else None
    rank = rank[idxt]

    if return_perm:
        return x, y, rank, crowd, perm
    else:
        return x, y, rank, crowd

def fast_non_dominated_sort(Y):
    ''' a fast non-dominated sorting method
        Y: output objective matrix
    '''
    N, d = Y.shape
    Q = [] # temp array of Pareto front index
    Sp = [] # temp array of points dominated by p
    S = [] # temp array of Sp
    rank = np.zeros(N, dtype=np.uint32) # Pareto rank
    n = np.zeros(N, dtype=np.uint32)  # domination counter of p
    dom = np.zeros((N, N), dtype=np.uint8)  # the dominate matrix, 1: i doms j, 2: j doms i

    # compute the dominate relationship online, much faster
    for i in range(N):
        for j in range(N):
            if i != j:
                if dominates(Y[i,:], Y[j,:]):
                    dom[i,j] = 1
                    Sp.append(j)
                elif dominates(Y[j,:], Y[i,:]):
                    dom[i,j] = 2
                    n[i] += 1
        if n[i] == 0:
            rank[i] = 0
            Q.append(i)
        S.append(Sp)
        Sp = []

    F = []
    F.append(Q)
    k = 0
    while len(F[k]) > 0:
        Q = []
        for i in range(len(F[k])):
            p = F[k][i]
            for j in range(len(S[p])):
                q = S[p][j]
                n[q] -= 1
                if n[q] == 0:
                    rank[q]  = k + 1
                    Q.append(q)
        k += 1
        F.append(Q)

    return rank, dom

def dominates(p,q):
    ''' comparison for multi-objective optimization
        d = True, if p dominates q
        d = False, if p not dominates q
        p and q are 1*nOutput array
    '''
    if np.sum(p > q) == 0:
        d = True
    else:
        d = False
    return d

def crowding_distance(Y):
    ''' compute crowding distance in NSGA-II
        Y is the output data matrix
        [n,d] = size(Y)
        n: number of points
        d: number of dimensions
    '''
    n,d = Y.shape
    lb = np.min(Y, axis=0, keepdims=True)
    ub = np.max(Y, axis=0, keepdims=True)

    if n == 1 or np.min(ub-lb) == 0.0:
        D = np.array([1.])
    else:
        U = (Y - lb) / (ub - lb)

        D = np.zeros(n)
        DS = np.zeros((n,d))

        idx = U.argsort(axis = 0)
        US = np.zeros((n,d))
        for i in range(d):
            US[:,i] = U[idx[:,i],i]

        DS[0,:] = 1.
        DS[n-1,:] = 1.

        for i in range(1,n-1):
            for j in range(d):
                DS[i,j] = US[i+1,j] - US[i-1,j]

        for i in range(n):
            for j in range(d):
                D[idx[i,j]] += DS[i,j]
        D[np.isnan(D)] = 0.0

    return D

def euclidean_distance(Y):
    """Row-wise euclidean distance.
    """
    n, d = Y.shape
    lb = np.min(Y, axis=0)
    ub = np.max(Y, axis=0)
    if np.min(ub-lb) == 0.0:
        return np.array([1.]*n)
    else:
        U = (Y - lb) / (ub - lb)
        return np.sqrt(np.sum(U ** 2, axis=1))


def tournament_prob(ax, i):
    p = ax[1]
    p1 = p*(1. - p)**i
    ax[0].append(p1)
    return (ax[0], p)

def tournament_selection(population_parm, population_obj, pop, poolsize, toursize, *metrics):
    ''' tournament selecting the best individuals into the mating pool'''

    candidates = np.arange(pop)
    sorted_candidates = np.lexsort(tuple((metric[candidates] for metric in metrics)))
    prob, _ = reduce(tournament_prob, candidates, ([], 0.5))
    poolidx = np.random.choice(sorted_candidates, size=poolsize, p=np.asarray(prob), replace=False)
    return population_parm[poolidx,:]


def remove_worst(population_para, population_obj, pop, nInput, nOutput, distance_metric=None):
    ''' remove the worst individuals in the population '''
    population_para, population_obj, rank, crowd = \
        sortMO(population_para, population_obj, nInput, nOutput, distance_metric=distance_metric)
    return population_para[0:pop,:], population_obj[0:pop,:], rank[0:pop]


