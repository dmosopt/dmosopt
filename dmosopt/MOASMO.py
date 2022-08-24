# Multi-Objective Adaptive Surrogate Model-based Optimization

import sys, pprint
import numpy as np
from numpy.random import default_rng
from dmosopt import MOEA, NSGA2, AGEMOEA, SMPSO, CMAES, gp, sa, sampling
from dmosopt.feasibility import LogisticFeasibilityModel

def optimization(model, param_names, objective_names, xlb, xub, n_epochs, pct, \
                 Xinit = None, Yinit = None, nConstraints = None, pop=100,
                 initial_maxiter=5, initial_method="slh",
                 feasibility_model=False,
                 surrogate_method="gpr",
                 surrogate_options={'anisotropic': False, 'optimizer': "sceua"},
                 optimizer="nsga2",
                 optimizer_kwargs= { 'gen': 100,
                                     'crossover_prob': 0.9,
                                     'mutation_prob': 0.1,
                                     'sampling_method': None,
                                     'di_crossover': 1., 'di_mutation': 20. },
                 sensitivity_method=None,
                 sensitivity_options={},
                 termination=None,
                 local_random=None,
                 logger=None):
    """ 
    Multi-Objective Adaptive Surrogate Modelling-based Optimization
    model: the evaluated model function
    param_names: names of model inputs
    objective_names: names of output objectives
    xlb: lower bound of input
    xub: upper bound of input
    n_epochs: number of epochs
    pct: percentage of resampled points in each iteration
    Xinit and Yinit: initial samplers for surrogate model construction
    ### options for the embedded NSGA-II optimizer:
        pop: number of population
        gen: number of generation
        crossover_prob: probability of crossover in each generation
        di_crossover: distribution index for crossover
        di_mutation: distribution index for mutation
    """
    nInput = len(param_names)
    nOutput = len(objective_names)
    N_resample = int(pop*pct)
    if (surrogate_method is not None) and (Xinit is None and Yinit is None):
        Ninit = nInput * 10
        Xinit = xinit(Ninit, param_names, method=initial_method, maxiter=initial_maxiter, logger=logger)
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

    c = None
    fsbm = None
    if C is not None:
        feasible = np.argwhere(np.all(C > 0., axis=1))
        if feasibility_model:
            fsbm = LogisticFeasibilityModel(Xinit,  C)
        if len(feasible) > 0:
            feasible = feasible.ravel()
            x = Xinit[feasible,:].copy()
            y = Yinit[feasible,:].copy()
            c = C[feasible,:].copy()
        c = c.copy()
    else:
        x = Xinit.copy()
        y = Yinit.copy()
        
    for i in range(n_epochs):
        
        sm = model
        if surrogate_method is not None:
            sm = train(nInput, nOutput, xlb, xub, x, y, c, 
                       surrogate_method=surrogate_method,
                       surrogate_options=surrogate_options,
                       logger=logger)
            if sensitivity_method is not None:
                di_dict = analyze_sensitivity(sm, xlb, xub, param_names, objective_names,
                                              sensitivity_method=sensitivity_method,
                                              sensitivity_options=sensitivity_options,
                                              logger=logger)
                optimizer_kwargs['di_mutation'] = di_dict['di_mutation']
                optimizer_kwargs['di_crossover'] = di_dict['di_crossover']


        if optimizer == 'nsga2':
            bestx_sm, besty_sm, gen_index, x_sm, y_sm = \
                NSGA2.optimization(sm, nInput, nOutput, xlb, xub, feasibility_model=fsbm, logger=logger, \
                                   pop=pop, local_random=local_random, termination=termination, **optimizer_kwargs)
        elif optimizer == 'age':
            bestx_sm, besty_sm, gen_index, x_sm, y_sm = \
                AGEMOEA.optimization(sm, nInput, nOutput, xlb, xub, feasibility_model=fsbm, logger=logger, \
                                     pop=pop, local_random=local_random, termination=termination, **optimizer_kwargs)
        elif optimizer == 'smpso':
            bestx_sm, besty_sm, x_sm, y_sm = \
                SMPSO.optimization(sm, nInput, nOutput, xlb, xub, feasibility_model=fsbm, logger=logger, \
                                   pop=pop, local_random=local_random, termination=termination, **optimizer_kwargs)
        elif optimizer == 'cmaes':
            bestx_sm, besty_sm, x_sm, y_sm = \
                CMAES.optimization(sm, nInput, nOutput, xlb, xub, logger=logger, \
                                   pop=pop, local_random=local_random,
                                   termination=termination, **optimizer_kwargs)
        else:
            raise RuntimeError(f"Unknown optimizer {optimizer}")
        
        if surrogate_method is not None:
            D = MOEA.crowding_distance(besty_sm)
            idxr = D.argsort()[::-1][:N_resample]
            x_resample = bestx_sm[idxr,:]
            y_resample = np.zeros((N_resample,nOutput))
            c_resample = None
            if C is not None:
                fsbm = LogisticFeasibilityModel(x_sm,  C)
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
            x = np.vstack((x, x_resample))
            y = np.vstack((y, y_resample))
            if c_resample is not None:
                c = np.vstack((c, c_resample))
            
    xtmp = x.copy()
    ytmp = y.copy()
    xtmp, ytmp, rank, _ = MOEA.sortMO(xtmp, ytmp, nInput, nOutput)
    idxp = (rank == 0)
    bestx = xtmp[idxp,:]
    besty = ytmp[idxp,:]

    return bestx, besty, x, y


def xinit(nEval, param_names, xlb, xub, nPrevious=None, method="glp", maxiter=5, local_random=None, logger=None):
    """ 
    Initialization for Multi-Objective Adaptive Surrogate Modelling-based Optimization
    nEval: number of evaluations per parameter
    param_names: model parameter names
    xlb: lower bound of input
    xub: upper bound of input
    """
    nInput = len(param_names)
    Ninit = nInput * nEval

    if local_random is None:
        local_random = default_rng()

    if nPrevious is None:
        nPrevious = 0

    if (Ninit <= 0) or (Ninit <= nPrevious):
        return None

    if isinstance(method, dict):
        Xinit = np.column_stack([method[k] for k in param_names])
        for i in range(Xinit.shape[1]):
            in_bounds = np.all(np.logical_and(Xinit[:,i] <= xub[i], Xinit[:,i] >= xlb[i]))
            if not in_bounds:
                logger.error(f'xinit: out of bounds values detected for parameter {param_names[i]}')
            assert(in_bounds)
        return Xinit
      

    if logger is not None:
        logger.info(f"xinit: generating {Ninit} initial parameters...")
             
    if method == "glp":
        Xinit = sampling.glp(Ninit, nInput, local_random=local_random, maxiter=maxiter)
    elif method == "slh":
        Xinit = sampling.slh(Ninit, nInput, local_random=local_random, maxiter=maxiter)
    elif method == "lh":
        Xinit = sampling.lh(Ninit, nInput, local_random=local_random, maxiter=maxiter)
    elif method == "mc":
        Xinit = sampling.mc(Ninit, nInput, local_random=local_random)
    elif method == "sobol":
        Xinit = sampling.sobol(Ninit, nInput, local_random=local_random)
    elif callable(method):
        Xinit = method(Ninit, nInput, local_random)
    else:
        raise RuntimeError(f'Unknown method {method}')

    Xinit = Xinit[nPrevious:,:] * (xub - xlb) + xlb

    return Xinit


def onestep(param_names, objective_names, xlb, xub, pct, \
            Xinit, Yinit, C, pop=100,
            feasibility_model=False,
            optimizer="nsga2",
            optimizer_kwargs= { 'gen': 100,
                                'crossover_prob': 0.9,
                                'mutation_prob': 0.1,
                                'sampling_method': None,
                                'di_crossover': 1.,
                                'di_mutation': 20. },
            surrogate_method="gpr",
            surrogate_options={'anisotropic': False, 'optimizer': "sceua"},
            sensitivity_method=None,
            sensitivity_options={},
            termination=None,
            local_random=None,
            return_sm=False,
            logger=None):
    """ 
    Multi-Objective Adaptive Surrogate Modelling-based Optimization
    One-step mode for offline optimization.


    xlb: lower bound of input
    xub: upper bound of input
    pct: percentage of resampled points in each iteration
    Xinit and Yinit: initial samplers for surrogate model construction
    ### options for the embedded NSGA-II:
        pop: number of population
        gen: number of generation
        crossover_prob: probability of crossover in each generation
        mutation_prob: probability of mutation in each generation
        di_crossover: distribution index for crossover
        di_mutation: distribution index for mutation
    """

    nInput = len(param_names)
    nOutput = len(objective_names)
    
    N_resample = int(pop*pct)
    x = Xinit.copy().astype(np.float32)
    y = Yinit.copy().astype(np.float32)
    fsbm = None
    if C is not None:
        feasible = np.argwhere(np.all(C > 0., axis=1))
        if len(feasible) > 0:
            feasible = feasible.ravel()
            try:
                if feasibility_model:
                    fsbm = LogisticFeasibilityModel(Xinit,  C)
                x = x[feasible,:]
                y = y[feasible,:]
            except:
                e = sys.exc_info()[0]
                logger.warning(f"Unable to fit feasibility model: {e}")
    sm = train(nInput, nOutput, xlb, xub, Xinit, Yinit, C, 
               surrogate_method=surrogate_method,
               surrogate_options=surrogate_options,
               logger=logger)
    
    if sensitivity_method is not None:
        di_dict = analyze_sensitivity(sm, xlb, xub, param_names, objective_names,
                                      sensitivity_method=sensitivity_method,
                                      sensitivity_options=sensitivity_options,
                                      logger=logger)
        optimizer_kwargs['di_mutation'] = di_dict['di_mutation']
        optimizer_kwargs['di_crossover'] = di_dict['di_crossover']

    if optimizer == 'nsga2':
        bestx_sm, besty_sm, gen_index, x_sm, y_sm = \
            NSGA2.optimization(sm, nInput, nOutput, xlb, xub, initial=(x, y), \
                               feasibility_model=fsbm, logger=logger, \
                               pop=pop, local_random=local_random, termination=termination,
                               **optimizer_kwargs)
    elif optimizer == 'age':
        bestx_sm, besty_sm, gen_index, x_sm, y_sm = \
            AGEMOEA.optimization(sm, nInput, nOutput, xlb, xub, initial=(x, y), \
                                 feasibility_model=fsbm, logger=logger, \
                                 pop=pop, local_random=local_random, termination=termination,
                                 **optimizer_kwargs)
    elif optimizer == 'smpso':
        bestx_sm, besty_sm, gen_index, x_sm, y_sm = \
            SMPSO.optimization(sm, nInput, nOutput, xlb, xub, initial=(x, y), \
                               feasibility_model=fsbm, logger=logger, \
                               pop=pop, local_random=local_random, termination=termination,
                               **optimizer_kwargs)
    elif optimizer == 'cmaes':
        bestx_sm, besty_sm, gen_index, x_sm, y_sm = \
            CMAES.optimization(sm, nInput, nOutput, xlb, xub, initial=(x, y), \
                               logger=logger, pop=pop, local_random=local_random, termination=termination,
                               **optimizer_kwargs)
    else:
        raise RuntimeError(f"Unknown optimizer {optimizer}")
        
    D = MOEA.crowding_distance(besty_sm)
    idxr = D.argsort()[::-1][:N_resample]
    x_resample = bestx_sm[idxr,:]
    y_pred = besty_sm[idxr,:]
    if return_sm:
        return x_resample, y_pred, gen_index, x_sm, y_sm
    else:
        return x_resample, y_pred


def train(nInput, nOutput, xlb, xub, \
          Xinit, Yinit, C, 
          surrogate_method="gpr",
          surrogate_options={'anisotropic': False, 'optimizer': "sceua"},
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

    x, y = MOEA.remove_duplicates(x, y)
                
    if surrogate_method == 'gpr':
        gpr_anisotropic = surrogate_options.get('anisotropic', False)
        gpr_optimizer = surrogate_options.get('optimizer', 'sceua')
        gpr_lengthscale_bounds=surrogate_options.get('lengthscale_bounds', (1e-3, 100.0))
        sm = gp.GPR_Matern(x, y, nInput, nOutput, xlb, xub, optimizer=gpr_optimizer,
                           anisotropic=gpr_anisotropic,
                           length_scale_bounds=gpr_lengthscale_bounds,
                           logger=logger)
    elif surrogate_method == 'egp':
        egp_lengthscale_bounds=surrogate_options.get('lengthscale_bounds', None)
        egp_likelihood_sigma=surrogate_options.get('likelihood_sigma', 1.0e-4)
        egp_adam_lr=surrogate_options.get('adam_lr', 0.01)
        egp_n_iter=surrogate_options.get('n_iter', 5000)
        egp_cuda=surrogate_options.get('cuda', 5000)
        sm = gp.EGP_Matern(x, y, nInput, nOutput, x.shape[0], xlb, xub,
                           gp_lengthscale_bounds=egp_lengthscale_bounds,
                           gp_likelihood_sigma=egp_likelihood_sigma,
                           adam_lr=egp_adam_lr, n_iter=egp_n_iter,
                           cuda=egp_cuda, logger=logger)
    elif surrogate_method == 'megp':
        megp_lengthscale_bounds=surrogate_options.get('lengthscale_bounds', None)
        megp_likelihood_sigma=surrogate_options.get('likelihood_sigma', 1.0e-4)
        megp_adam_lr=surrogate_options.get('adam_lr', 0.01)
        megp_n_iter=surrogate_options.get('n_iter', 5000)
        megp_cuda=surrogate_options.get('cuda', 5000)
        sm = gp.MEGP_Matern(x, y, nInput, nOutput, x.shape[0], xlb, xub,
                            gp_lengthscale_bounds=megp_lengthscale_bounds,
                            gp_likelihood_sigma=megp_likelihood_sigma,
                            adam_lr=megp_adam_lr, n_iter=megp_n_iter,
                            cuda=megp_cuda, logger=logger)
    elif surrogate_method == 'vgp':
        vgp_lengthscale_bounds=surrogate_options.get('lengthscale_bounds', (1e-6, 100.0))
        vgp_likelihood_sigma=surrogate_options.get('likelihood_sigma', 1.0e-4)
        vgp_natgrad_gamma=surrogate_options.get('natgrad_gamma', 1.0)
        vgp_adam_lr=surrogate_options.get('adam_lr', 0.01)
        vgp_n_iter=surrogate_options.get('n_iter', 3000)
        sm = gp.VGP_Matern(x, y, nInput, nOutput, x.shape[0], xlb, xub,
                           gp_lengthscale_bounds=vgp_lengthscale_bounds,
                           gp_likelihood_sigma=vgp_likelihood_sigma,
                           natgrad_gamma=vgp_natgrad_gamma,
                           adam_lr=vgp_adam_lr, n_iter=vgp_n_iter,
                           logger=logger)
    elif surrogate_method == 'svgp':
        svgp_batch_size=surrogate_options.get('batch_size', 50)
        svgp_lengthscale_bounds=surrogate_options.get('lengthscale_bounds', (1e-6, 100.0))
        svgp_likelihood_sigma=surrogate_options.get('likelihood_sigma', 1.0e-4)
        svgp_natgrad_gamma=surrogate_options.get('natgrad_gamma', 0.1)
        svgp_adam_lr=surrogate_options.get('adam_lr', 0.01)
        svgp_n_iter=surrogate_options.get('n_iter', 30000)
        sm = gp.SVGP_Matern(x, y, nInput, nOutput, x.shape[0], xlb, xub,
                            batch_size=svgp_batch_size,
                            gp_lengthscale_bounds=svgp_lengthscale_bounds,
                            gp_likelihood_sigma=svgp_likelihood_sigma,
                            natgrad_gamma=svgp_natgrad_gamma,
                            adam_lr=svgp_adam_lr, n_iter=svgp_n_iter,
                            logger=logger)
    elif surrogate_method == 'siv':
        siv_batch_size=surrogate_options.get('batch_size', 50)
        siv_lengthscale_bounds=surrogate_options.get('lengthscale_bounds', (1e-6, 100.0))
        siv_likelihood_sigma=surrogate_options.get('likelihood_sigma', 1.0e-4)
        siv_natgrad_gamma=surrogate_options.get('natgrad_gamma', 0.1)
        siv_adam_lr=surrogate_options.get('adam_lr', 0.01)
        siv_n_iter=surrogate_options.get('n_iter', 30000)
        siv_num_latent_gps=surrogate_options.get('num_latent_gps', None)
        sm = gp.SIV_Matern(x, y, nInput, nOutput, x.shape[0], xlb, xub,
                           batch_size=siv_batch_size,
                           gp_lengthscale_bounds=siv_lengthscale_bounds,
                           gp_likelihood_sigma=siv_likelihood_sigma,
                           natgrad_gamma=siv_natgrad_gamma,
                           adam_lr=siv_adam_lr, n_iter=siv_n_iter,
                           num_latent_gps=siv_num_latent_gps,
                           logger=logger)
    elif surrogate_method == 'crv':
        crv_batch_size=surrogate_options.get('batch_size', 50)
        crv_lengthscale_bounds=surrogate_options.get('lengthscale_bounds', (1e-6, 100.0))
        crv_likelihood_sigma=surrogate_options.get('likelihood_sigma', 1.0e-4)
        crv_natgrad_gamma=surrogate_options.get('natgrad_gamma', 0.1)
        crv_adam_lr=surrogate_options.get('adam_lr', 0.01)
        crv_n_iter=surrogate_options.get('n_iter', 30000)
        crv_num_latent_gps=surrogate_options.get('num_latent_gps', None)
        sm = gp.CRV_Matern(x, y, nInput, nOutput, x.shape[0], xlb, xub,
                           batch_size=crv_batch_size,
                           gp_lengthscale_bounds=crv_lengthscale_bounds,
                           gp_likelihood_sigma=crv_likelihood_sigma,
                           natgrad_gamma=crv_natgrad_gamma,
                           adam_lr=crv_adam_lr, n_iter=crv_n_iter,
                           num_latent_gps=crv_num_latent_gps,
                           logger=logger)
    elif surrogate_method == 'pod':
        sm = pod.POD_RBF(x, y, nInput, nOutput, xlb, xub, logger=logger)
    else:
        raise RuntimeError(f'Unknown surrogate method {surrogate_method}')

    return sm


def analyze_sensitivity(sm, xlb, xub, param_names, objective_names, sensitivity_method=None, sensitivity_options={}, di_min=1.0, di_max=20., logger=None):
    
    di_mutation = None
    di_crossover = None
    if sensitivity_method is not None:
        if sensitivity_method == 'dgsm':
            sens = sa.SA_DGSM(xlb, xub, param_names, objective_names)
            sens_results = sens.analyze(sm)
            S1s = np.vstack(list([sens_results['S1'][objective_name]
                                  for objective_name in objective_names]))
            S1max = np.max(S1s, axis=0)
            S1nmax = S1max / np.max(S1max)
            di_mutation = np.clip(S1nmax * di_max, di_min, None)
            di_crossover = np.clip(S1nmax * di_max, di_min, None)
        elif sensitivity_method == 'fast':
            sens = sa.SA_FAST(xlb, xub, param_names, objective_names)
            sens_results = sens.analyze(sm)
            S1s = np.vstack(list([sens_results['S1'][objective_name]
                                  for objective_name in objective_names]))
            S1max = np.max(S1s, axis=0)
            S1nmax = S1max / np.max(S1max)
            di_mutation = np.clip(S1nmax * di_max, di_min, None)
            di_crossover = np.clip(S1nmax * di_max, di_min, None)
        else:
            RuntimeError(f"Unknown sensitivity method {sensitivity_method}")

    if logger is not None:
        logger.info(f'analyze_sensitivity: di_mutation = {di_mutation}')
        logger.info(f'analyze_sensitivity: di_crossover = {di_crossover}')
    di_dict = {}
    di_dict['di_mutation'] = di_mutation
    di_dict['di_crossover'] = di_crossover

    return di_dict


def get_best(x, y, f, c, nInput, nOutput, epochs=None, feasible=True, return_perm=False, return_feasible=False):
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
            if epochs is not None:
                epochs = epochs[feasible]
    xtmp, ytmp, rank, _, perm = MOEA.sortMO(xtmp, ytmp, nInput, nOutput, return_perm=True)
    idxp = (rank == 0)
    best_x = xtmp[idxp,:]
    best_y = ytmp[idxp,:]
    best_f = None
    if f is not None:
        best_f = f[perm][idxp]
    best_c = None
    if c is not None:
        best_c = c[perm,:][idxp,:]

    best_epoch = None
    if epochs is not None:
        best_epoch = epochs[perm][idxp]

    if not return_perm:
        perm = None
    if return_feasible:
        return best_x, best_y, best_f, best_c, best_epoch, perm, feasible
    else:
        return best_x, best_y, best_f, best_c, best_epoch, perm


def get_feasible(x, y, f, c, nInput, nOutput, epochs=None):
    xtmp = x.copy()
    ytmp = y.copy()
    if c is not None:
        feasible = np.argwhere(np.all(c > 0., axis=1))
        if len(feasible) > 0:
            feasible = feasible.ravel()
            xtmp = xtmp[feasible,:]
            ytmp = ytmp[feasible,:]
            if f is not None:
                f = f[feasible]
            c = c[feasible,:]
            if epochs is not None:
                epochs = epochs[feasible]
    else:
        feasible = None

    perm_x, perm_y, rank, _, perm = MOEA.sortMO(xtmp, ytmp, nInput, nOutput, return_perm=True)
    # x, y are already permutated upon return
    perm_f = f[perm] 
    perm_epoch = epochs[perm]
    perm_c = c[perm]

    uniq_rank, rnk_inv, rnk_cnt = np.unique(rank, return_inverse=True, return_counts=True)

    collect_idx = [[] for i in uniq_rank] 
    for idx, rnk in enumerate(rnk_inv):
        collect_idx[rnk].append(idx)    

    rank_idx = np.array(collect_idx,dtype=np.ndarray)
    for idx, i in enumerate(rank_idx):
        rank_idx[idx] = np.array(i)

    uniq_epc, epc_inv, epc_cnt = np.unique(perm_epoch, return_inverse=True, return_counts=True)

    collect_epoch = [[] for i in uniq_epc]  
    for idx, epc in enumerate(epc_inv):
        collect_epoch[epc].append(idx)    
    epc_idx = np.array(collect_epoch,dtype=np.ndarray)
    for idx, i in enumerate(epc_idx):
        epc_idx[idx] = np.array(i)

    rnk_epc_idx = np.empty(shape=(uniq_rank.shape[0], uniq_epc.shape[0]), dtype=np.ndarray)

    for idx, i in enumerate(rank_idx):
        for jidx, j in enumerate(epc_idx):
            rnk_epc_idx[idx, jidx] = np.intersect1d(i,j, assume_unique=True)

    perm_arrs = (perm_x, perm_y, perm_f, perm_epoch, perm, feasible)
    rnk_arrs = (uniq_rank, rank_idx, rnk_cnt)
    epc_arrs = (uniq_epc, epc_idx, epc_cnt)
    
    return perm_arrs, rnk_arrs, epc_arrs, rnk_epc_idx 
