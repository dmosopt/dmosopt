# Multi-Objective Adaptive Surrogate Model-based Optimization
from __future__ import division, print_function, absolute_import
import sys
import numpy as np
from dmosopt import NSGA2, gp, sampling

def optimization(model, nInput, nOutput, xlb, xub, niter, pct, \
                 Xinit = None, Yinit = None, pop = 100, gen = 100, \
                 crossover_rate = 0.9, mutation_rate = None, mum = 20,
                 gpr_optimizer="sceua", logger=None):
    """ 
    Multi-Objective Adaptive Surrogate Modelling-based Optimization
    model: the evaluated model function
    nInput: number of model input
    nOutput: number of output objectives
    xlb: lower bound of input
    xub: upper bound of input
    niter: number of iteration
    pct: percentage of resampled points in each iteration
    Xinit and Yinit: initial samplers for surrogate model construction
    ### options for the embedded NSGA-II of MO-ASMO
        pop: number of population
        gen: number of generation
        crossover_rate: ratio of crossover in each generation
        mu: distribution index for crossover
        mum: distribution index for mutation
    """
    N_resample = int(pop*pct)
    if (Xinit is None and Yinit is None):
        Ninit = nInput * 10
        Xinit = sampling.glp(Ninit, nInput)
        for i in range(Ninit):
            Xinit[i,:] = Xinit[i,:] * (xub - xlb) + xlb
        Yinit = np.zeros((Ninit, nOutput))
        for i in range(Ninit):
            Yinit[i,:] = model.evaluate(Xinit[i,:])
    else:
        Ninit = Xinit.shape[0]
    icall = Ninit
    x = Xinit.copy()
    y = Yinit.copy()

    if mutation_rate is None:
        mutation_rate = 1. / float(nInput)
        
    for i in range(niter):
        print('Surrogate Opt loop: %d' % i)
        sm = gp.GPR_Matern(x, y, nInput, nOutput, x.shape[0], xlb, xub, optimizer=gpr_optimizer, logger=logger)
        bestx_sm, besty_sm, x_sm, y_sm = \
            NSGA2.optimization(sm, nInput, nOutput, xlb, xub, \
                               pop, gen, crossover_rate, mutation_rate, mu, mum, logger=logger)
        D = NSGA2.crowding_distance(besty_sm)
        idxr = D.argsort()[::-1][:N_resample]
        x_resample = bestx_sm[idxr,:]
        y_resample = np.zeros((N_resample,nOutput))
        for j in range(N_resample):
            y_resample[j,:] = model.evaluate(x_resample[j,:])
        icall += N_resample
        x = np.vstack((x, x_resample))
        y = np.vstack((y, y_resample))

    xtmp = x.copy()
    ytmp = y.copy()
    xtmp, ytmp, rank, crowd = NSGA2.sortMO(xtmp, ytmp, nInput, nOutput)
    idxp = (rank == 0)
    bestx = xtmp[idxp,:]
    besty = ytmp[idxp,:]

    return bestx, besty, x, y


def xinit(nEval, nInput, nOutput, xlb, xub, nPrevious=None, maxiter=5):
    """ 
    Initialization for Multi-Objective Adaptive Surrogate Modelling-based Optimization
    model: the evaluated model function
    nInput: number of model input
    nOutput: number of output objectives
    xlb: lower bound of input
    xub: upper bound of input
    """
    Ninit = nInput * nEval
    if nPrevious is not None:
        Ninit -= nPrevious
    
    if Ninit <= 0:
        return None

    Xinit = sampling.glp(Ninit, nInput, maxiter=maxiter)

    for i in range(Ninit):
        Xinit[i,:] = Xinit[i,:] * (xub - xlb) + xlb
    #Yinit = np.zeros((Ninit, nOutput))
    #for i in range(Ninit):
    #    Yinit[i,:] = model.evaluate(Xinit[i,:])

    return Xinit


def onestep(nInput, nOutput, xlb, xub, pct, \
            Xinit, Yinit, C, pop = 100, gen = 100, \
            crossover_rate = 0.9, mutation_rate = None, mu = 15, mum = 20,
            gpr_optimizer="sceua", logger=None):
    """ 
    Multi-Objective Adaptive Surrogate Modelling-based Optimization
    One-step mode for offline optimization
    Do NOT call the model evaluation function
    nInput: number of model input
    nOutput: number of output objectives
    xlb: lower bound of input
    xub: upper bound of input
    pct: percentage of resampled points in each iteration
    Xinit and Yinit: initial samplers for surrogate model construction
    ### options for the embedded NSGA-II of MO-ASMO
        pop: number of population
        gen: number of generation
        crossover_rate: ratio of crossover in each generation
        mutation_rate: ratio of mutation in each generation
        mu: distribution index for crossover
        mum: distribution index for mutation
    """
    if mutation_rate is None:
        mutation_rate = 1. / float(nInput)
    N_resample = int(pop*pct)
    x = Xinit.copy()
    y = Yinit.copy()
    if C is not None:
        feasible = np.argwhere(C > 0.)[:,0]
        if len(feasible) > 0:
            x = x[feasible,:]
            y = y[feasible,:]
    sm = gp.GPR_Matern(x, y, nInput, nOutput, x.shape[0], xlb, xub, optimizer=gpr_optimizer, logger=logger)
    bestx_sm, besty_sm, x_sm, y_sm = \
        NSGA2.optimization(sm, nInput, nOutput, xlb, xub, \
                           pop, gen, crossover_rate, mutation_rate, mu, mum, logger=logger)
    D = NSGA2.crowding_distance(besty_sm)
    idxr = D.argsort()[::-1][:N_resample]
    x_resample = bestx_sm[idxr,:]
    return x_resample

def get_best(x, y, f, c, nInput, nOutput, feasible=True):
    xtmp = x.copy()
    ytmp = y.copy()
    if feasible and c is not None:
        feasible = np.argwhere(c > 0.)[:,0]
        if len(feasible) > 0:
            xtmp = xtmp[feasible,:]
            ytmp = ytmp[feasible,:]
            if f is not None:
                f = f[feasible]
            c = c[feasible,:]
    xtmp, ytmp, rank, crowd, perm = NSGA2.sortMO(xtmp, ytmp, nInput, nOutput, return_perm=True)
    idxp = (rank == 0)
    bestx = xtmp[idxp,:]
    besty = ytmp[idxp,:]
    bestf = None
    if f is not None:
        bestf = f[perm][idxp]
    bestc = None
    if c is not None:
        bestc = c[perm,:][idxp,:]

    return bestx, besty, bestf, bestc
