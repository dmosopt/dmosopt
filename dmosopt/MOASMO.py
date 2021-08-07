# Multi-Objective Adaptive Surrogate Model-based Optimization
import sys, pprint
import numpy as np
from dmosopt import NSGA2, AGEMOEA, gp, sampling
from dmosopt.feasibility import FeasibilityModel

def optimization(model, nInput, nOutput, xlb, xub, niter, pct, \
                 Xinit = None, Yinit = None, nConstraints = None, pop=100,
                 initial_maxiter=5, initial_method="glp",
                 feasibility_model=False,
                 gpr_anisotropic=False, gpr_optimizer="sceua", optimizer="nsga2",
                 optimizer_kwargs= { 'gen': 100,
                                     'crossover_rate': 0.9,
                                     'mutation_rate': None,
                                     'mu': 1., 'mum': 20. },
                 logger=None):
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
    ### options for the embedded NSGA-II optimizer:
        pop: number of population
        gen: number of generation
        crossover_rate: ratio of crossover in each generation
        mu: distribution index for crossover
        mum: distribution index for mutation
    """
    N_resample = int(pop*pct)
    if (Xinit is None and Yinit is None):
        Ninit = nInput * 10
        Xinit = xinit(Ninit, nInput, method=initial_method, maxiter=initial_maxiter, logger=logger)
        for i in range(Ninit):
            Xinit[i,:] = Xinit[i,:] * (xub - xlb) + xlb
        Yinit = np.zeros((Ninit, nOutput))
        C = None
        if nConstraints is not None:
            C = np.zeros((Ninit, nConstraints))
            for i in range(Ninit):
                Yinit[i,:], C[i,:] = model.evaluate(Xinit[i,:])
        else:
            for i in range(Ninit):
                Yinit[i,:] = model.evaluate(Xinit[i,:])
            
    else:
        Ninit = Xinit.shape[0]
    icall = Ninit
    fsbm = None
    if C is not None:
        feasible = np.argwhere(np.all(C > 0., axis=1))
        if feasibility_model:
            fsbm = FeasibilityModel(Xinit,  C)
        if len(feasible) > 0:
            feasible = feasible.ravel()
            x = Xinit[feasible,:].copy()
            y = Yinit[feasible,:].copy()
    else:
        x = Xinit.copy()
        y = Yinit.copy()

    for i in range(niter):
        sm = gp.GPR_Matern(x, y, nInput, nOutput, x.shape[0], xlb, xub, optimizer=gpr_optimizer, anisotropic=gpr_anisotropic, logger=logger)
        if optimizer == 'nsga2':
            bestx_sm, besty_sm, x_sm, y_sm = \
                NSGA2.optimization(sm, nInput, nOutput, xlb, xub, feasibility_model=fsbm, logger=logger, \
                                   pop=pop, **optimizer_kwargs)
        elif optimizer == 'age':
            bestx_sm, besty_sm, x_sm, y_sm = \
                AGEMOEA.optimization(sm, nInput, nOutput, xlb, xub, feasibility_model=fsbm, logger=logger, \
                                     pop=pop, **optimizer_kwargs)
        else:
            raise RuntimeError(f"Unknown optimizer {optimizer}")
        D = NSGA2.crowding_distance(besty_sm)
        idxr = D.argsort()[::-1][:N_resample]
        x_resample = bestx_sm[idxr,:]
        y_resample = np.zeros((N_resample,nOutput))
        c_resample = None
        if C is not None:
            fsbm = FeasibilityModel(x_sm,  C)
            c_resample = np.zeros((N_resample,nConstraints))
            for j in range(N_resample):
                y_resample[j,:], c_resample[j,:] = model.evaluate(x_resample[j,:])
            feasible = np.argwhere(np.all(c_resample > 0., axis=1))
            if len(feasible) > 0:
                feasible = feasible.ravel()
                x_resample = x_resample[feasible,:]
                y_resample = y_resample[feasible,:]
        else:
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


def xinit(nEval, nInput, nOutput, xlb, xub, nPrevious=None, method="glp", maxiter=5, logger=None):
    """ 
    Initialization for Multi-Objective Adaptive Surrogate Modelling-based Optimization
    nEval: number of evaluations per parameter
    nInput: number of model parameters
    nOutput: number of output objectives
    xlb: lower bound of input
    xub: upper bound of input
    """
    Ninit = nInput * nEval

    if nPrevious is not None:
        Ninit -= nPrevious

    if logger is not None:
        logger.info(f"xinit: generating {Ninit} initial parameters...")
        
    if Ninit <= 0:
        return None
    
    if method == "glp":
        Xinit = sampling.glp(Ninit, nInput, maxiter=maxiter)
    elif method == "slh":
        Xinit = sampling.slh(Ninit, nInput, maxiter=maxiter)
    elif method == "lh":
        Xinit = sampling.lh(Ninit, nInput, maxiter=maxiter)
    elif method == "mc":
        Xinit = sampling.mc(Ninit, nInput)
    else:
        raise RuntimeError(f'Unknown method {method}')

    if nPrevious is None:
        nPrevious = 0

    for i in range(nPrevious, Ninit):
        Xinit[i,:] = Xinit[i,:] * (xub - xlb) + xlb

    Xinit = Xinit[nPrevious:, :]

    return Xinit


def onestep(nInput, nOutput, xlb, xub, pct, \
            Xinit, Yinit, C, pop=100,
            feasibility_model=False,
            gpr_anisotropic=False, gpr_optimizer="sceua",
            optimizer="nsga2",
            optimizer_kwargs= { 'gen': 100,
                                'crossover_rate': 0.9,
                                'mutation_rate': None,
                                'mu': 1., 'mum': 20. },
            logger=None):
    """ 
    Multi-Objective Adaptive Surrogate Modelling-based Optimization
    One-step mode for offline optimization.

    nInput: number of model input
    nOutput: number of output objectives
    xlb: lower bound of input
    xub: upper bound of input
    pct: percentage of resampled points in each iteration
    Xinit and Yinit: initial samplers for surrogate model construction
    ### options for the embedded NSGA-II:
        pop: number of population
        gen: number of generation
        crossover_rate: ratio of crossover in each generation
        mutation_rate: ratio of mutation in each generation
        mu: distribution index for crossover
        mum: distribution index for mutation
    """
    N_resample = int(pop*pct)
    x = Xinit.copy()
    y = Yinit.copy()
    fsbm = None
    if C is not None:
        feasible = np.argwhere(np.all(C > 0., axis=1))
        if len(feasible) > 0:
            feasible = feasible.ravel()
            try:
                if feasibility_model:
                    fsbm = FeasibilityModel(Xinit,  C)
                x = x[feasible,:]
                y = y[feasible,:]
                logger.info(f"Found {len(feasible)} feasible solutions")
            except:
                e = sys.exc_info()[0]
                logger.warning(f"Unable to fit feasibility model: {e}")
    sm = gp.GPR_Matern(x, y, nInput, nOutput, x.shape[0], xlb, xub, optimizer=gpr_optimizer, anisotropic=gpr_anisotropic, logger=logger)
    if optimizer == 'nsga2':
        bestx_sm, besty_sm, x_sm, y_sm = \
            NSGA2.optimization(sm, nInput, nOutput, xlb, xub, feasibility_model=fsbm, logger=logger, \
                               pop=pop, **optimizer_kwargs)
    elif optimizer == 'age':
        bestx_sm, besty_sm, x_sm, y_sm = \
            AGEMOEA.optimization(sm, nInput, nOutput, xlb, xub, feasibility_model=fsbm, logger=logger, \
                                 pop=pop, **optimizer_kwargs)
    else:
        raise RuntimeError(f"Unknown optimizer {optimizer}")
        
    D = NSGA2.crowding_distance(besty_sm)
    idxr = D.argsort()[::-1][:N_resample]
    x_resample = bestx_sm[idxr,:]
    return x_resample


def train(nInput, nOutput, xlb, xub, \
          Xinit, Yinit, C, 
          gpr_anisotropic=False, gpr_optimizer="sceua", 
          logger=None):
    """ 
    Multi-Objective Adaptive Surrogate Modelling-based Optimization
    Training of surrogate model.

    nInput: number of model input
    nOutput: number of output objectives
    xlb: lower bound of input
    xub: upper bound of input
    Xinit and Yinit: initial samplers for surrogate model construction
    """

    x = Xinit.copy()
    y = Yinit.copy()

    if C is not None:
        feasible = np.argwhere(np.all(C > 0., axis=1))
        if len(feasible) > 0:
            feasible = feasible.ravel()
            try:
                x = x[feasible,:]
                y = y[feasible,:]
                logger.info(f"Found {len(feasible)} feasible solutions")
            except:
                e = sys.exc_info()[0]
                logger.warning(f"Unable to fit feasibility model: {e}")
                
    sm = gp.GPR_Matern(x, y, nInput, nOutput, x.shape[0], xlb, xub, optimizer=gpr_optimizer, anisotropic=gpr_anisotropic, logger=logger)

    return sm


def get_best(x, y, f, c, nInput, nOutput, feasible=True):
    xtmp = x.copy()
    ytmp = y.copy()
    if feasible and c is not None:
        feasible = np.argwhere(np.all(c > 0., axis=1))
        if len(feasible) > 0:
            feasible = feasible.ravel()
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
