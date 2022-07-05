## Speed-constrained multiobjective particle swarm optimization
## SMPSO: A New PSO Metaheuristic for Multi-objective Optimization
##

import gc, itertools
import numpy  as np
from numpy.random import default_rng
from dmosopt import sampling
from dmosopt.datatypes import OptHistory
from dmosopt.dda import dda_non_dominated_sort
from dmosopt.MOEA import mutation, sortMO, crowding_distance, remove_worst, remove_duplicates


def optimization(model, nInput, nOutput, xlb, xub, initial=None, feasibility_model=None, termination=None,
                 distance_metric=None, pop=100, gen=100, nchildren=1,
                 di_mutation=20., swarm_size=5, sampling_method=None, local_random=None, logger=None,
                 **kwargs):
    ''' 
        Speed-constrained multiobjective particle swarm optimization.

        model: the evaluated model function
        nInput: number of model input
        nOutput: number of output objectives
        xlb: lower bound of input
        xub: upper bound of input
        pop: number of population
        gen: number of generation
        di_mutation: distribution index for mutation
    '''

    if local_random is None:
        local_random = default_rng()

    mutation_rate = 1. / float(nInput)

    y_distance_metrics = []
    y_distance_metrics.append(distance_metric)
    x_distance_metrics = None
    if feasibility_model is not None:
        x_distance_metrics = [feasibility_model.rank]

    if np.isscalar(di_mutation):
        di_mutation = np.asarray([di_mutation]*nInput)
    
    pop_slices = list([range(p*pop, (p+1)*pop) for p in range(swarm_size)])

    x_initial, y_initial = None, None
    if initial is not None:
        x_initial, y_initial = initial
    
    xs = []
    ys = []
    for sl in pop_slices:
        if sampling_method is None:
            x = sampling.lh(pop, nInput, local_random)
            x_s = x * (xub - xlb) + xlb
        elif callable(sampling_method):
            x_s = sampling_method(pop, nInput, local_random, xlb, xub)
        else:
            raise RuntimeError(f'Unknown sampling method {sampling_method}')
        if x_initial is not None:
            x_s = np.vstack((x_initial, x_s))
        y_s = model.evaluate(x_s)
        xs.append(x_s)
        ys.append(y_s)

    population_parm = np.zeros((swarm_size*pop, nInput), dtype=np.float32)
    population_obj = np.zeros((swarm_size*pop, nOutput), dtype=np.float32)
    
    velocity = local_random.uniform(size=(swarm_size*pop, nInput)) * (xub - xlb) + xlb

    
    ranks = []
    for p, sl in enumerate(pop_slices):
        xs[p], ys[p], rank_p, _ = sortMO(xs[p], ys[p], nInput, nOutput,
                                         x_distance_metrics=x_distance_metrics,
                                         y_distance_metrics=y_distance_metrics)
        population_parm[sl] = xs[p][:pop]
        population_obj[sl]  = ys[p][:pop]
        ranks.append(rank_p)

    x = population_parm.copy()
    y = population_obj.copy()

    gen_indexes = []
    gen_indexes.append(np.zeros((x.shape[0],),dtype=np.uint32))

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
                logger.info(f"SMPSO: generation {i+1}...")
            else:
                logger.info(f"SMPSO: generation {i+1} of {gen}...")

                                    
        count = 0
        xs_gens = [ [] for _ in range(swarm_size) ]
                                    
        for p, sl in enumerate(pop_slices):
            xs_updated = update_position(population_parm[sl], velocity[sl], xlb, xub)
            xs_gens[p].append(xs_updated)
            
        while count < pop:
            parentidx = local_random.integers(low=0, high=pop, size=(swarm_size, 1))
            for p, sl in enumerate(pop_slices):
                parent = population_parm[sl][parentidx[p,0],:]
                children  = mutation(local_random, parent, di_mutation, xlb, xub, mutation_rate=mutation_rate, nchildren=nchildren)
                child = children[0]
                xs_gens[p].append(child)
            count += 1

        x_gens = np.vstack([np.vstack(x) for x in xs_gens])
        y_gens = model.evaluate(x_gens)
        x_new.append(x_gens)
        y_new.append(y_gens)
        gen_indexes.append(np.ones((x_gens.shape[0],),dtype=np.uint32)*i)
        
        for sl in pop_slices:
            D = crowding_distance(y_gens[sl])
            mean_velocity_before = np.mean(velocity[sl])
            velocity[sl] = velocity_vector(local_random, population_parm[sl], velocity[sl], x_gens[sl], D, xlb, xub)
            
        for p, sl in enumerate(pop_slices):
            population_parm_p = np.vstack((population_parm[sl], x_gens[sl]))
            population_obj_p  = np.vstack((population_obj[sl], y_gens[sl]))
            population_parm_p, population_obj_p = remove_duplicates(population_parm_p, population_obj_p)
            population_parm[sl], population_obj[sl], ranks[p] = \
                remove_worst(population_parm_p, population_obj_p, pop, nInput, nOutput,
                             x_distance_metrics=x_distance_metrics,
                             y_distance_metrics=y_distance_metrics)
        gc.collect()
        n_eval += count
            
    population_parm, population_obj = remove_duplicates(population_parm, population_obj)
    bestx, besty, _ = remove_worst(population_parm, population_obj, pop, nInput, nOutput,
                                   x_distance_metrics=x_distance_metrics,
                                   y_distance_metrics=y_distance_metrics)

    gen_index = np.concatenate(gen_indexes)

    x = np.vstack([x] + x_new)
    y = np.vstack([y] + y_new)
        
    return bestx, besty, gen_index, x, y

                                    
def update_position(parameters, velocity, xlb, xub):
    position = np.clip(parameters + velocity,  xlb, xub)
    return position


def velocity_vector(local_random, position, velocity, archive, crowding, xlb, xub):

    r1  = local_random.uniform(low = 0.0, high = 1.0, size = 1)[0]
    r2  = local_random.uniform(low = 0.0, high = 1.0, size = 1)[0]
    w   = local_random.uniform(low = 0.1, high = 0.5, size = 1)[0]
    c1  = local_random.uniform(low = 1.5, high = 2.5, size = 1)[0]
    c2  = local_random.uniform(low = 1.5, high = 2.5, size = 1)[0]
    phi = 0
    if (c1 + c2 > 4):
        phi = c1 + c2
    else:
        phi = 0
    chi      = 2 / (2 - phi - ( (phi**2) - 4*phi )**(1/2))
    
    output = np.zeros((position.shape[0], velocity.shape[1]))
    delta    = [(xub[i] - xlb[i])/2 for i in range(0, len(xlb))]
    if (archive.shape[0] > 2):
        ind_1, ind_2 = local_random.integers(low=0, high=archive.shape[0], size=2)
        if (crowding[ind_1] < crowding[ind_2]):
            ind_1, ind_2 = ind_2, ind_1
    else:
        ind_1 = 0
        ind_2 = 0

    
    for i in range(0, velocity.shape[0]):
        for j in range(0, velocity.shape[1]):
            output[i,j] = (w*velocity[i,j] + c1*r1*(archive[ind_1, j] - position[i,j]) + c2*r2*(archive[ind_2, j] - position[i,j]))*chi
            output[i,j] = np.clip(output[i,j], -delta[j], delta[j])
            
    return output

