#
# Common routines used by multi-objective evolutionary algorithms.
#

import numpy as np
from functools import reduce
from dmosopt.dda import dda_non_dominated_sort
from scipy.spatial.distance import cdist

# function sharedmoea(selfunc,μ,λ)
#  \selfunc, selection function to be used.
# μand λ, population and offspring sizes.
# t ←0; P0 ←randompopulation(μ).
# while end criterion not met do
# Poff←applyvariation(Pt,λ).
# Pt+1 ←selfunc(Pt ∪Poff,μ).
# t ←t +1.
# return nondomset(Pt+1), final non-dominated set



def mutation(local_random, parent, mutation_rate, di_mutation, xlb, xub, nchildren=1):
    ''' Polynomial Mutation in Genetic Algorithm
        muration_rate: mutation rate
        di_mutation: distribution index for mutation
            This determine how well spread the child will be from its parent.
        parent: sample point before mutation
	'''
    n = len(parent)
    if np.isscalar(di_mutation):
        di_mutation = np.asarray([di_mutation]*n)
    children = np.ndarray((nchildren,n))
    delta = np.ndarray((n,))
    for i in range(nchildren):
        u = local_random.random(n)
        lo = np.argwhere(u < mutation_rate).ravel()
        hi = np.argwhere(u >= mutation_rate).ravel()
        delta[lo] = (2.0*u[lo])**(1.0/(di_mutation[lo]+1)) - 1.0
        delta[hi] = 1.0 - (2.0*(1.0 - u[hi]))**(1.0/(di_mutation[hi]+1))
        children[i, :] = np.clip(parent + (xub - xlb) * delta, xlb, xub)
    return children


def crossover_sbx(local_random, parent1, parent2, di_crossover, xlb, xub, nchildren=1):
    ''' SBX (Simulated Binary Crossover) in Genetic Algorithm

         di_crossover: distribution index for crossover
         This determine how well spread the children will be from their parents.
    '''
    n = len(parent1)
    if np.isscalar(di_crossover):
        di_crossover = np.asarray([di_crossover]*n)
    children1 = np.ndarray((nchildren, n))
    children2 = np.ndarray((nchildren, n))
    beta = np.ndarray((n,))
    for i in range(nchildren):
        u = local_random.random(n)
        lo = np.argwhere(u <= 0.5).ravel()
        hi = np.argwhere(u > 0.5).ravel()
        beta[lo] = (2.0*u[lo])**(1.0/(di_crossover[lo]+1))
        beta[hi] = (1.0/(2.0*(1.0 - u[hi])))**(1.0/(di_crossover[hi]+1))
        children1[i,:] = np.clip(0.5*((1-beta)*parent1 + (1+beta)*parent2), xlb, xub)
        children2[i,:] = np.clip(0.5*((1+beta)*parent1 + (1-beta)*parent2), xlb, xub)
    return children1, children2




def sortMO(x, y, nInput, nOutput, return_perm=False, distance_metrics=['crowding']):
    ''' Non-dominated sort for multi-objective optimization
        x: input parameter matrix
        y: output objectives matrix
        nInput: number of input
        nOutput: number of output
        return_perm: if True, return permutation indices of original input
    '''
    distance_functions = [crowding_distance]
    if distance_metrics is not None:
        distance_functions = []
        for distance_metric in distance_metrics:
            if distance_metric == None:
                distance_functions.append(crowding_distance)
            elif distance_metric == 'crowding':
                distance_functions.append(crowding_distance)
            elif distance_metric == 'euclidean':
                distance_functions.append(euclidean_distance)
            elif callable(distance_metric):
                distance_functions.append(distance_metric)
            else:
                raise RuntimeError(f'sortMO: unknown distance metric {distance_metric}')
        
    rank = dda_non_dominated_sort(y)

    dists = list([np.zeros_like(rank) for _ in distance_functions])
    rmax = int(rank.max())
    for front in range(rmax+1):
        rankidx = (rank == front)
        for i, distance_function in enumerate(distance_functions):
            D = distance_function(y[rankidx,:])
            dists[i][rankidx] = D
        
    perm = np.lexsort((list([-dist for dist in dists])+[rank]))
     
    x = x[perm]
    y = y[perm]
    rank = rank[perm]
    dists = tuple([dist[perm] for dist in dists])
    
    if return_perm:
        return x, y, rank, dists, perm
    else:
        return x, y, rank, dists
                
        

def crowding_distance(Y):
    ''' Crowding distance metric.
        Y is the output data matrix
        [n,d] = size(Y)
        n: number of points
        d: number of dimensions
    '''
    n,d = Y.shape
    lb = np.min(Y, axis=0, keepdims=True)
    ub = np.max(Y, axis=0, keepdims=True)

    if n == 1:
        D = np.array([1.])
    else:
        ub_minus_lb = ub - lb
        ub_minus_lb[ub_minus_lb == 0.0] = 1.
        
        U = (Y - lb) / ub_minus_lb

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
    ub_minus_lb = ub - lb
    ub_minus_lb[ub_minus_lb == 0.0] = 1.
    U = (Y - lb) / ub_minus_lb
    return np.sqrt(np.sum(U ** 2, axis=1))


def tournament_prob(ax, i):
    p = ax[1]
    p1 = p*(1. - p)**i
    ax[0].append(p1)
    return (ax[0], p)

def tournament_selection(local_random, pop, poolsize, toursize, *metrics):
    ''' tournament selecting the best individuals into the mating pool'''

    candidates = np.arange(pop)
    sorted_candidates = np.lexsort(tuple((metric[candidates] for metric in metrics)))
    prob, _ = reduce(tournament_prob, candidates, ([], 0.5))
    poolidx = local_random.choice(sorted_candidates, size=poolsize, p=np.asarray(prob), replace=False)
    return poolidx


def remove_worst(population_parm, population_obj, pop, nInput, nOutput, distance_metrics=None):
    ''' remove the worst individuals in the population '''
    population_parm, population_obj, rank, _ = \
        sortMO(population_parm, population_obj, nInput, nOutput, distance_metrics=distance_metrics)
    return population_parm[0:pop,:], population_obj[0:pop,:], rank[0:pop]

def get_duplicates(X, eps=1e-16):
    
    D = cdist(X, X)
    D[np.triu_indices(len(X))] = np.inf
    D[np.isnan(D)] = np.inf

    is_duplicate = np.zeros((len(X),), dtype=bool)
    is_duplicate[np.any(D <= eps, axis=1)] = True
    
    return is_duplicate

def remove_duplicates(population_parm, population_obj, eps=1e-16):
    ''' remove duplicate individuals in the population '''
    is_duplicate = get_duplicates(population_parm, eps=eps)
    return population_parm[~is_duplicate,:], population_obj[~is_duplicate,:]

