#
# Adaptive evolutionary algorithm based on non-euclidean geometry for
# many-objective optimization. A. Panichella, Proceedings of the
# Genetic and Evolutionary Computation Conference, 2019.  
#
# Based on implementations in platEMO and PyMOO (by Ben Crulis):
#
# 
# https://github.com/BenCrulis/pymoo/tree/AGE_MOEA/pymoo
# https://github.com/BIMK/PlatEMO/tree/master/PlatEMO/Algorithms/Multi-objective%20optimization/AGE-MOEA
#

import numpy as np
import copy, gc, itertools
from functools import reduce
from dmosopt import sampling
from dmosopt.datatypes import OptHistory


def optimization(model, nInput, nOutput, xlb, xub, initial=None, feasibility_model=None, logger=None, termination=None,
                 pop=100, gen=100, crossover_rate = 0.9, mutation_rate = 0.05, di_crossover = 1., di_mutation = 20.):
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
        
    population_parm = x[:pop]
    population_obj  = y[:pop]
    population_parm, population_obj, rank, crowd_dist = \
        environmental_selection(population_parm, population_obj, pop, nInput, nOutput)

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
            opt = OptHistory(i, n_eval, population_parm, population_obj, None)
            if termination.has_terminated(opt):
                break
        if logger is not None:
            if termination is not None:
                logger.info(f"AGE-MOEA: generation {i+1}...")
            else:
                logger.info(f"AGE-MOEA: generation {i+1} of {gen}...")

        pool = tournament_selection(population_parm, population_obj, pop, poolsize, toursize, rank, -crowd_dist)
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
                y1 = model.evaluate(child)
                xs_gen.append(child)
                count += 1
        x_gen = np.vstack(xs_gen)
        y_gen = model.evaluate(x_gen)
        x_new.append(x_gen)
        y_new.append(y_gen)
        population_parm = np.vstack((population_parm, x_gen))
        population_obj  = np.vstack((population_obj, y_gen))
        population_parm, population_obj, rank, crowd_dist = \
            environmental_selection(population_parm, population_obj, pop, nInput, nOutput, logger=logger)
        gc.collect()
        n_eval += count
        if termination is not None:
            opt = OptHistory(i, n_eval, population_parm, population_obj, None)
            if termination.has_terminated(opt):
                break

    sorted_population = np.lexsort(tuple((metric for metric in [rank, -crowd_dist])), axis=0)
    bestx = population_parm[sorted_population].copy()
    besty = population_obj[sorted_population].copy()

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


def sortMO(x, y):
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

    return x, y, rank

    
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
    if sum(p > q) == 0:
        d = True
    else:
        d = False
    return d



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
        return normalization

    # Calculate the intercepts of the hyperplane constructed by the extreme
    # points and the axes

    try:
        hyperplane = np.linalg.solve(front[extreme], np.ones(n))
    except:
        hyperplane = [np.nan]
        
    if any(np.isnan(hyperplane)) or any(np.isinf(hyperplane)) or any(hyperplane < 0):
        normalization = np.max(front, axis=0)
    else:
        normalization = 1. / hyperplane
        if any(np.isnan(normalization)) or any(np.isinf(normalization)):
            normalization = np.max(front, axis=0)

    normalization[np.isclose(normalization, 0.0, rtol=1e-4, atol=1e-4)] = 1.0

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
    """Return the indexes of the extreme points. """

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


def environmental_selection(population_parm, population_obj, pop, nInput, nOutput, logger=None):

    xs, ys, rank = sortMO(population_parm, population_obj)
    rmax = int(np.max(rank))
    rmin = int(np.min(rank))

    yn = np.zeros_like(ys)
    crowd_dist = np.zeros_like(rank)
    selected = np.zeros_like(rank).astype(np.bool)
    
    # get the first front for normalization
    front_1 = np.argwhere(rank == 0).ravel()
    #if logger is not None:
    #    logger.info(f"front_1.shape = {front_1.shape}")

    # follows from the definition of the ideal point but with current non dominated solutions
    ideal_point = np.min(ys[front_1, :], axis=0)

    normalization, p, crowd_dist[front_1] = survival_score(ys, front_1, ideal_point)
    yn[front_1, :] = ys[front_1] / normalization
    
    count = len(front_1)
    if count < pop:
        selected[front_1] = True
        for r in range(1, rmax+1):
            front_r = np.argwhere(rank == r).ravel()
            yn[front_r] = ys[front_r] / normalization
            crowd_dist[front_r] = 1. / minkowski_matrix(yn[front_r, :], ideal_point[None, :], p=p).squeeze()
            if (count + len(front_r)) < pop:
                selected[front_r] = True
                count += len(front_r)
            else:
                # Select the solutions in the last front based on their crowding distances
                selection_rank = np.argsort(crowd_dist[front_r])[::-1]
                selected[front_r[selection_rank[: pop - count]]] = True
                break
    else:
        selection_rank = np.argsort(crowd_dist[front_1])[::-1]
        selected[front_1[np.random.choice(selection_rank, size=pop, replace=False)]] = True
            
    assert(np.sum(selected) > 0)
    # return selected solutions, number of selected should be equal to population size
    return xs[selected].copy(), ys[selected].copy(), rank[selected].copy(), crowd_dist[selected].copy()

