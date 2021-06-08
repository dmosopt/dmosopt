# Adaptive evolutionary algorithm based on non-euclidean geometry for
# many-objective optimization. A. Panichella, Proceedings of the
# Genetic and Evolutionary Computation Conference, 2019.  

import numpy as np
import copy
from dmosopt import sampling


def optimization(model, nInput, nOutput, xlb, xub, feasibility_model=None, logger=None, pop=100, gen=100, \
                 crossover_rate = 0.9, mutation_rate = 0.05, mu = 1., mum = 20.):
    ''' AGE-MOEA, A multi-objective algorithm based on non-euclidean geometry.
        model: the evaluated model function
        nInput: number of model input
        nOutput: number of output objectives
        xlb: lower bound of input
        xub: upper bound of input
        pop: number of population
        gen: number of generation
        crossover_rate: ratio of crossover in each generation
        mutation_rate: ratio of muration in each generation
        mu: distribution index for crossover
        mum: distribution index for mutation
    '''
    poolsize = int(round(pop/2.)); # size of mating pool;
    toursize = 2;                  # tournament size;

    if mutation_rate is None:
        mutation_rate = 1. / float(nInput)

    x = sampling.slh(pop, nInput)
    x = x * (xub - xlb) + xlb
    y = np.zeros((pop, nOutput))
    for i in range(pop):
        y[i,:] = model.evaluate(x[i,:])
        
    x, y, rank, crowd = sortMO(x, y, nInput, nOutput)
    population_parm = x.copy()
    population_obj  = y.copy()

    for i in range(gen):
        if logger is not None:
            logger.info(f"AGE-MOEA: iteration {i+1} of {gen}...")
        pool = tournament_selection(population_parm, population_obj, nInput, pop, poolsize, toursize)
        count = 0
        while (count < pop - 1):
            if (np.random.rand() < crossover_rate):
                parentidx = np.random.choice(poolsize, 2, replace = False)
                parent1   = pool[parentidx[0],:]
                parent2   = pool[parentidx[1],:]
                child1, child2 = crossover(parent1, parent2, mu, xlb, xub)
                y1 = model.evaluate(child1)
                y2 = model.evaluate(child2)
                population_parm = np.vstack((population_parm,child1,child2))
                population_obj  = np.vstack((population_obj,y1,y2))
                count += 2
            else:
                parentidx = np.random.randint(poolsize)
                parent    = pool[parentidx,:]
                child     = mutation(parent, mutation_rate, mum, xlb, xub)
                y1 = model.evaluate(child)
                population_parm = np.vstack((population_parm,child))
                population_obj  = np.vstack((population_obj,y1))
                count += 1
                
        population_parm, population_obj, _ = \
            environmental_selection(population_parm, population_obj, pop, nInput, nOutput)
                                    
        bestx = population_parm.copy()
        besty = population_obj.copy()
        
    return bestx, besty, x, y

def sortMO(x, y, nInput, nOutput, return_perm=False):
    ''' Non domination sorting for multi-objective optimization
        x: input parameter matrix
        y: output objectives matrix
        nInput: number of input
        nOutput: number of output
        return_perm: if True, return permutation indices of original input
    '''
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
        D = crowding_distance(y[rankidx,:])
        idxd = D.argsort()[::-1]
        crowd[rankidx] = D[idxd]
        idxtt = np.array(range(len(rank)))[rankidx]
        idxt[c:(c+len(idxtt))] = idxtt[idxd]
        c += len(idxtt)
    x = x[idxt,:]
    y = y[idxt,:]
    perm = idxr[idxt]
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
    rank = np.zeros(N) # Pareto rank
    n = np.zeros(N)  # domination counter of p
    dom = np.zeros((N, N))  # the dominate matrix, 1: i doms j, 2: j doms i
    
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
        S.append(copy.deepcopy(Sp))
        Sp = []

    F = []
    F.append(copy.deepcopy(Q))
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
        F.append(copy.deepcopy(Q))

    return rank, dom

def dominates(p,q):
    ''' comparison for multi-objective optimization
        d = True, if p dominates q
        d = False, if p not dominates q
        p and q are 1*nOutput array
    '''
    if sum(p > q) == 0:
        d = True
    else:
        d = False
    return d

def crowding_distance(Y):
    ''' compute crowding distance in NSGA-II
        Y is the output data matrix
        [n,d] = size(Y)
        n: number of points
        d: number of dimentions
    '''
    n,d = Y.shape
    lb = np.min(Y, axis = 0)
    ub = np.max(Y, axis = 0)

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

def mutation(parent, mutation_rate, mum, xlb, xub):
    ''' Polynomial Mutation in Genetic Algorithm
        For more information about PMut refer the NSGA-II paper.
        muration_rate: mutation rate
        mum: distribution index for mutation, default = 20
            This determine how well spread the child will be from its parent.
        parent: sample point before mutation
	'''
    n     = len(parent)
    delta = np.ndarray(n)
    child = np.ndarray(n)
    u     = np.random.rand(n)
    for i in range(n):
        if (u[i] < mutation_rate):
            delta[i] = (2.0*u[i])**(1.0/(mum+1)) - 1.0
        else:
            delta[i] = 1.0 - (2.0*(1.0 - u[i]))**(1.0/(mum+1))
        child[i] = parent[i] + (xub[i] - xlb[i]) * delta[i]
    child = np.clip(child, xlb, xub)
    return child

def crossover(parent1, parent2, mu, xlb, xub):
    ''' SBX (Simulated Binary Crossover) in Genetic Algorithm
        For more information about SBX refer the NSGA-II paper.
        mu: distribution index for crossover, default = 20
        This determine how well spread the children will be from their parents.
    '''
    n      = len(parent1)
    beta   = np.ndarray(n)
    child1 = np.ndarray(n)
    child2 = np.ndarray(n)
    u = np.random.rand(n)
    for i in range(n):
        if (u[i] <= 0.5):
            beta[i] = (2.0*u[i])**(1.0/(mu+1))
        else:
            beta[i] = (1.0/(2.0*(1.0 - u[i])))**(1.0/(mu+1))
        child1[i] = 0.5*((1-beta[i])*parent1[i] + (1+beta[i])*parent2[i])
        child2[i] = 0.5*((1+beta[i])*parent1[i] + (1-beta[i])*parent2[i])
    child1 = np.clip(child1, xlb, xub)
    child2 = np.clip(child2, xlb, xub)
    return child1, child2


    


def normalize(front, extreme):
    ''' Rescale and normalize first non-dominated front:

        f^{i}_{norm}(S) = [ f_i(S) - z^{min}_{i} ] / a_{i} \forall S \in F1

        where:
          f_{i} objective i for solution S
          z^{min}_{i} minimum value across all solutions in the front
          a_{i} the intercept of the M-dimensional hyperplane with the objective axis f_{i}

        The M-dimensional hyperplane is composed by the extreme vectors

            z^{max}_{i} = max (f_{i}(S) - z^{min}_{i}), S \in F1
            i.e., the largest objective values in F1 after the translation 
            toward the origin of the axes.
    '''
    m, n = front.shape

    # if system is abnormal, use min-max normalization
    if len(extreme) != len(np.unique(extreme, axis=0)):
        normalization = np.max(front, axis=0)
        front = front / normalization
        return front, normalization

    # Calculate the intercepts of the hyperplane constructed by the extreme
    # points and the axes

    hyperplane = np.linalg.solve(front[extreme], np.ones(n))
    if any(np.isnan(hyperplane)) or any(np.isinf(hyperplane)) or any(hyperplane < 0):
        normalization = np.max(front, axis=0)
    else:
        normalization = 1. / hyperplane
        if any(np.isnan(normalization)) or any(np.isinf(normalization)):
            normalization = np.max(front, axis=0)

    return normalization

def minkowski_matrix(A, B, p):
    """workaround for scipy's cdist refusing p<1"""
    i_ind, j_ind = np.meshgrid(np.arange(A.shape[0]), np.arange(B.shape[0]))
    return np.power(np.power(np.abs(A[i_ind] - B[j_ind]), p).sum(axis=2), 1.0/p)


def get_geometry(front, extreme):
    ''' approximate p(norm) '''

    m, n = front.shape
    
    d = point_2_line_distance(front, np.zeros(n), np.ones(n))
    d[extreme] = np.inf
    index = np.argmin(d)
    p = np.log(n) / np.log(1.0 / np.mean(front[index, :]))

    if np.isnan(p) or p <= 0.1:
        p = 1.0
    elif p > 20:
        p = 20.0  # avoid numpy underflow

    return p


def point_2_line_distance(P, A, B):
    
    d = np.zeros(P.shape[0])

    for i in range(P.shape[0]):
        pa = P[i] - A
        ba = B - A
        t = np.dot(pa, ba)/np.dot(ba, ba)
        d[i] = np.linalg.norm(pa - t * ba, 2)

    return d


def find_corner_solutions(front):
    """Return the indexes of the extreme points"""

    m, n = front.shape

    if m <= n:
        return np.arange(m)

    # let's define the axes of the n-dimensional spaces
    W = 1e-6 + np.eye(n)
    r = W.shape[0]
    indexes = np.zeros(n, dtype=np.intp)
    selected = np.zeros(m, dtype=np.bool)
    for i in range(r):
        dists = point_2_line_distance(front, np.zeros(n), W[i, :])
        dists[selected] = np.inf  # prevent already selected to be reselected
        index = np.argmin(dists)
        indexes[i] = index
        selected[index] = True
        
    return indexes


def tournament_selection(population_parm, population_obj, nInput, pop, poolsize, toursize):
    ''' tournament selecting the best individuals into the mating pool'''
    pool    = np.zeros([poolsize,nInput])
    poolidx = np.zeros(poolsize)
    count   = 0
    while (count < poolsize-1):
        candidate = np.random.choice(pop, toursize, replace = False)
        idx = candidate.min()
        if not(idx in poolidx):
            poolidx[count] = idx
            pool[count,:]  = population_parm[idx,:]
            count += 1
    return pool


def survival_score(y, front, ideal_point):
    ''' front: index of non-dominated solutions of rank d
    '''

    m, n = y[front,:].shape
    crowd_dist = np.zeros(m)
    
    if m < n:
        p = 1
        normalization = np.max(y[front, :], axis=0)
        return normalization, p, crowd_dist

    # shift the ideal point to the origin
    yfront = y[front, :] - ideal_point
    
    extreme = find_corner_solutions(yfront)
    normalization = normalize(yfront, extreme)
    ynfront = yfront / normalization
    p = get_geometry(ynfront, extreme)

    # set the distance for the extreme solutions
    crowd_dist[extreme] = np.inf
    selected = np.full(m, False)
    selected[extreme] = True
    
    nn = np.linalg.norm(ynfront, p, axis=1)
    distances = minkowski_matrix(ynfront, ynfront, p=p)
    distances = distances / nn[:, None]

    neighbors = 2
    remaining = np.arange(m)
    remaining = list(remaining[~selected])
    
    for i in range(m - np.sum(selected)):
        mg = np.meshgrid(np.arange(selected.shape[0])[selected], remaining)
        D_mg = distances[tuple(mg)]  # avoid Numpy's future deprecation of array special indexing

        if D_mg.shape[1] > 1:
            # equivalent to mink(distances(remaining, selected),neighbors,2); in Matlab
            maxim = np.argpartition(D_mg, neighbors - 1, axis=1)[:, :neighbors]
            tmp = np.sum(np.take_along_axis(D_mg, maxim, axis=1), axis=1)
            index: int = np.argmax(tmp)
            d = tmp[index]
        else:
            index = D_mg[:, 0].argmax()
            d = D_mg[index, 0]

        best = remaining.pop(index)
        selected[best] = True
        crowd_dist[best] = d

    return normalization, p, crowd_dist


def environmental_selection(population_parm, population_obj, pop, nInput, nOutput):

    xs, ys, rank, _ = sortMO(population_parm, population_obj, nInput, nOutput)
    rmax = int(rank.max())

    yn = np.zeros_like(ys)
    crowd_dist = np.zeros_like(rank)
    selected = rank < rmax
    
    # get the first front for normalization
    front_1 = np.argwhere(rank == 0).ravel()

    # follows from the definition of the ideal point but with current non dominated solutions
    ideal_point = np.min(ys[front_1, :], axis=0)
        
    normalization, p, crowd_dist[front_1] = survival_score(ys, front_1, ideal_point)
    yn[front_1, :] = ys[front_1, :] / normalization
    for r in range(1, rmax+1):
        front_r = np.argwhere(rank == r).ravel()
        yn[front_r] = ys[front_r] / normalization
        crowd_dist[front_r] = 1. / minkowski_matrix(yn[front_r, :], ideal_point[None, :], p=p).squeeze()
        
    # Select the solutions in the last front based on their crowding distances
    last = np.argwhere(rank == rmax).ravel()
    selection_rank = np.argsort(crowd_dist[last])[::-1]
    selected[last[selection_rank[: pop - np.sum(selected)]]] = True

    # return selected solutions, number of selected should be equal to population size
    return population_parm[selected], population_obj[selected], crowd_dist
