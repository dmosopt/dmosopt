# Multi-Objective Adaptive Surrogate Model-based Optimization

import sys, pprint
import numpy as np
from numpy.random import default_rng
from dmosopt import MOEA, NSGA2, AGEMOEA, SMPSO, gp, sampling
from dmosopt.feasibility import FeasibilityModel


def optimization(
    model,
    nInput,
    nOutput,
    xlb,
    xub,
    pct,
    Xinit=None,
    Yinit=None,
    nConstraints=None,
    pop=100,
    initial_maxiter=5,
    initial_method="slh",
    feasibility_model=False,
    optimizer="nsga2",
    optimizer_kwargs={
        "gen": 100,
        "crossover_rate": 0.9,
        "mutation_rate": None,
        "sampling_method": None,
        "di_crossover": 1.0,
        "di_mutation": 20.0,
    },
    termination=None,
    local_random=None,
    logger=None,
):
    """
    Multi-Objective Adaptive Surrogate Modelling-based Optimization

    model: the evaluated model function
    nInput: number of model input
    nOutput: number of output objectives
    xlb: lower bound of input
    xub: upper bound of input
    n_epochs: number of epochs
    pct: percentage of resampled points in each iteration
    Xinit and Yinit: initial samplers for surrogate model construction
    ### options for the embedded optimizer:
        pop: number of population
        gen: number of generation
        crossover_rate: ratio of crossover in each generation
        di_crossover: distribution index for crossover
        di_mutation: distribution index for mutation
    """
    Ninit = Xinit.shape[0]

    c = None
    fsbm = None
    if C is not None:
        feasible = np.argwhere(np.all(C > 0.0, axis=1))
        if feasibility_model:
            fsbm = FeasibilityModel(Xinit, C)
        if len(feasible) > 0:
            feasible = feasible.ravel()
            x = Xinit[feasible, :].copy()
            y = Yinit[feasible, :].copy()
            c = C[feasible, :].copy()
        c = c.copy()
    else:
        x = Xinit.copy()
        y = Yinit.copy()

    if optimizer == "nsga2":
        bestx_sm, besty_sm, gen_index, x_sm, y_sm = NSGA2.optimization(
            model,
            nInput,
            nOutput,
            xlb,
            xub,
            feasibility_model=fsbm,
            logger=logger,
            pop=pop,
            local_random=local_random,
            termination=termination,
            **optimizer_kwargs,
        )
    elif optimizer == "age":
        bestx_sm, besty_sm, gen_index, x_sm, y_sm = AGEMOEA.optimization(
            model,
            nInput,
            nOutput,
            xlb,
            xub,
            feasibility_model=fsbm,
            logger=logger,
            pop=pop,
            local_random=local_random,
            termination=termination,
            **optimizer_kwargs,
        )
    elif optimizer == "smpso":
        bestx_sm, besty_sm, x_sm, y_sm = SMPSO.optimization(
            model,
            nInput,
            nOutput,
            xlb,
            xub,
            feasibility_model=fsbm,
            logger=logger,
            pop=pop,
            local_random=local_random,
            termination=termination,
            **optimizer_kwargs,
        )
    else:
        raise RuntimeError(f"Unknown optimizer {optimizer}")

    xtmp = x.copy()
    ytmp = y.copy()
    xtmp, ytmp, rank, crowd = MOEA.sortMO(xtmp, ytmp, nInput, nOutput)
    idxp = rank == 0
    bestx = xtmp[idxp, :]
    besty = ytmp[idxp, :]

    return bestx, besty, x, y


def xinit(
    nEval,
    nInput,
    nOutput,
    xlb,
    xub,
    nPrevious=None,
    method="glp",
    maxiter=5,
    local_random=None,
    logger=None,
):
    """
    Initialization for Multi-Objective Adaptive Surrogate Modelling-based Optimization
    nEval: number of evaluations per parameter
    nInput: number of model parameters
    nOutput: number of output objectives
    xlb: lower bound of input
    xub: upper bound of input
    """
    Ninit = nInput * nEval

    if local_random is None:
        local_random = default_rng()

    if nPrevious is None:
        nPrevious = 0

    if (Ninit <= 0) or (Ninit <= nPrevious):
        return None

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
    elif callable(method):
        Xinit = method(Ninit, nInput, local_random)
    else:
        raise RuntimeError(f"Unknown method {method}")

    Xinit = Xinit[nPrevious:, :] * (xub - xlb) + xlb

    return Xinit


def step(
    nInput,
    nOutput,
    xlb,
    xub,
    pct,
    Xinit,
    Yinit,
    C,
    pop=100,
    feasibility_model=False,
    optimizer="nsga2",
    optimizer_kwargs={
        "gen": 100,
        "crossover_rate": 0.9,
        "mutation_rate": None,
        "sampling_method": None,
        "di_crossover": 1.0,
        "di_mutation": 20.0,
    },
    surrogate_method="gpr",
    surrogate_options={"anisotropic": False, "optimizer": "sceua"},
    termination=None,
    local_random=None,
    return_sm=False,
    logger=None,
):
    """
    Multi-Objective Adaptive Surrogate Modelling-based Optimization
    Perform one epoch of optimization.

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
        di_crossover: distribution index for crossover
        di_mutation: distribution index for mutation
    """
    N_resample = int(pop * pct)
    x = Xinit.copy().astype(np.float32) if Xinit is not None else None
    y = Yinit.copy().astype(np.float32) if Yinit is not None else None
    fsbm = None
    if C is not None:
        feasible = np.argwhere(np.all(C > 0.0, axis=1))
        if len(feasible) > 0:
            feasible = feasible.ravel()
            try:
                if feasibility_model:
                    fsbm = FeasibilityModel(Xinit, C)
                x = x[feasible, :]
                y = y[feasible, :]
            except:
                e = sys.exc_info()[0]
                if logger is not None:
                    logger.warning(f"Unable to fit feasibility model: {e}")

    initial = None
    if (x is not None) and (y is not None):
        initial = (x, y)
    sm = None
    if surrogate_method is not None:
        sm = train(
            nInput,
            nOutput,
            xlb,
            xub,
            Xinit,
            Yinit,
            C,
            surrogate_method=surrogate_method,
            surrogate_options=surrogate_options,
            logger=logger,
        )
    if optimizer == "nsga2":
        gen = NSGA2.optimization(
            nInput,
            nOutput,
            xlb,
            xub,
            initial=initial,
            feasibility_model=fsbm,
            logger=logger,
            pop=pop,
            local_random=local_random,
            termination=termination,
            **optimizer_kwargs,
        )
    elif optimizer == "age":
        gen = AGEMOEA.optimization(
            nInput,
            nOutput,
            xlb,
            xub,
            initial=initial,
            feasibility_model=fsbm,
            logger=logger,
            pop=pop,
            local_random=local_random,
            termination=termination,
            **optimizer_kwargs,
        )
    elif optimizer == "smpso":
        gen = SMPSO.optimization(
            nInput,
            nOutput,
            xlb,
            xub,
            initial=initial,
            feasibility_model=fsbm,
            logger=logger,
            pop=pop,
            local_random=local_random,
            termination=termination,
            **optimizer_kwargs,
        )
    else:
        raise RuntimeError(f"Unknown optimizer {optimizer}")

    res = next(gen)
    while True:
        logger.info(f"MOASMO.step: res = {res}")
        x_gen = res
        if sm is not None:
            y_gen = sm.evaluate(x_gen)
        else:
            y_gen = yield x_gen

        logger.info(
            f"MOASMO.step: gen = {gen} x_gen = {x_gen.shape} y_gen = {y_gen.shape}"
        )
        try:
            res = gen.send(y_gen)
            logger.info(f"MOASMO.step: after send: res = {res}")
        except StopIteration as ex:
            res = ex.args[0]
            bestx_sm = res.best_x
            besty_sm = res.best_y
            gen_index = res.gen_index
            x_sm = res.x
            y_sm = res.y
            break

    D = MOEA.crowding_distance(besty_sm)
    idxr = D.argsort()[::-1][:N_resample]
    x_resample = bestx_sm[idxr, :]
    y_pred = besty_sm[idxr, :]

    if return_sm:
        return x_resample, y_pred, gen_index, x_sm, y_sm
    else:
        return x_resample, y_pred


def train(
    nInput,
    nOutput,
    xlb,
    xub,
    Xinit,
    Yinit,
    C,
    surrogate_method="gpr",
    surrogate_options={"anisotropic": False, "optimizer": "sceua"},
    logger=None,
):
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
        feasible = np.argwhere(np.all(C > 0.0, axis=1))
        if len(feasible) > 0:
            feasible = feasible.ravel()
            try:
                x = x[feasible, :]
                y = y[feasible, :]
                logger.info(f"Found {len(feasible)} feasible solutions")
            except:
                e = sys.exc_info()[0]
                logger.warning(f"Unable to fit feasibility model: {e}")
    else:
        logger.info(f"Found {x.shape[0]} solutions")

    x, y = MOEA.remove_duplicates(x, y)
    if surrogate_method == "gpr":
        gpr_anisotropic = surrogate_options.get("anisotropic", False)
        gpr_optimizer = surrogate_options.get("optimizer", "sceua")
        gpr_lengthscale_bounds = surrogate_options.get(
            "lengthscale_bounds", (1e-3, 100.0)
        )
        sm = gp.GPR_Matern(
            x,
            y,
            nInput,
            nOutput,
            xlb,
            xub,
            optimizer=gpr_optimizer,
            anisotropic=gpr_anisotropic,
            length_scale_bounds=gpr_lengthscale_bounds,
            logger=logger,
        )
    elif surrogate_method == "vgp":
        vgp_lengthscale_bounds = surrogate_options.get(
            "lengthscale_bounds", (1e-6, 100.0)
        )
        vgp_likelihood_sigma = surrogate_options.get("likelihood_sigma", 1.0e-4)
        vgp_natgrad_gamma = surrogate_options.get("natgrad_gamma", 1.0)
        vgp_adam_lr = surrogate_options.get("adam_lr", 0.01)
        vgp_n_iter = surrogate_options.get("n_iter", 3000)
        sm = gp.VGP_Matern(
            x,
            y,
            nInput,
            nOutput,
            x.shape[0],
            xlb,
            xub,
            gp_lengthscale_bounds=vgp_lengthscale_bounds,
            gp_likelihood_sigma=vgp_likelihood_sigma,
            natgrad_gamma=vgp_natgrad_gamma,
            adam_lr=vgp_adam_lr,
            n_iter=vgp_n_iter,
            logger=logger,
        )
    elif surrogate_method == "svgp":
        svgp_lengthscale_bounds = surrogate_options.get(
            "lengthscale_bounds", (1e-6, 100.0)
        )
        svgp_likelihood_sigma = surrogate_options.get("likelihood_sigma", 1.0e-4)
        svgp_natgrad_gamma = surrogate_options.get("natgrad_gamma", 0.1)
        svgp_adam_lr = surrogate_options.get("adam_lr", 0.01)
        svgp_n_iter = surrogate_options.get("n_iter", 30000)
        sm = gp.SVGP_Matern(
            x,
            y,
            nInput,
            nOutput,
            x.shape[0],
            xlb,
            xub,
            gp_lengthscale_bounds=svgp_lengthscale_bounds,
            gp_likelihood_sigma=svgp_likelihood_sigma,
            natgrad_gamma=svgp_natgrad_gamma,
            adam_lr=svgp_adam_lr,
            n_iter=svgp_n_iter,
            logger=logger,
        )
    elif surrogate_method == "siv":
        siv_lengthscale_bounds = surrogate_options.get(
            "lengthscale_bounds", (1e-6, 100.0)
        )
        siv_likelihood_sigma = surrogate_options.get("likelihood_sigma", 1.0e-4)
        siv_natgrad_gamma = surrogate_options.get("natgrad_gamma", 0.1)
        siv_adam_lr = surrogate_options.get("adam_lr", 0.01)
        siv_n_iter = surrogate_options.get("n_iter", 30000)
        siv_num_latent_gps = surrogate_options.get("num_latent_gps", None)
        sm = gp.SIV_Matern(
            x,
            y,
            nInput,
            nOutput,
            x.shape[0],
            xlb,
            xub,
            gp_lengthscale_bounds=siv_lengthscale_bounds,
            gp_likelihood_sigma=siv_likelihood_sigma,
            natgrad_gamma=siv_natgrad_gamma,
            adam_lr=siv_adam_lr,
            n_iter=siv_n_iter,
            num_latent_gps=siv_num_latent_gps,
            logger=logger,
        )
    elif surrogate_method == "pod":
        sm = pod.POD_RBF(x, y, nInput, nOutput, xlb, xub, logger=logger)
    else:
        raise RuntimeError(f"Unknown surrogate method {surrogate_method}")

    return sm


def get_best(
    x,
    y,
    f,
    c,
    nInput,
    nOutput,
    epochs=None,
    feasible=True,
    return_perm=False,
    return_feasible=False,
):
    xtmp = x.copy()
    ytmp = y.copy()
    if feasible and c is not None:
        feasible = np.argwhere(np.all(c > 0.0, axis=1))
        if len(feasible) > 0:
            feasible = feasible.ravel()
            xtmp = xtmp[feasible, :]
            ytmp = ytmp[feasible, :]
            if f is not None:
                f = f[feasible]
            c = c[feasible, :]
            if epochs is not None:
                epochs = epochs[feasible]
    xtmp, ytmp, rank, crowd, perm = MOEA.sortMO(
        xtmp, ytmp, nInput, nOutput, return_perm=True
    )
    idxp = rank == 0
    best_x = xtmp[idxp, :]
    best_y = ytmp[idxp, :]
    best_f = None
    if f is not None:
        best_f = f[perm][idxp]
    best_c = None
    if c is not None:
        best_c = c[perm, :][idxp, :]

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
        feasible = np.argwhere(np.all(c > 0.0, axis=1))
        if len(feasible) > 0:
            feasible = feasible.ravel()
            xtmp = xtmp[feasible, :]
            ytmp = ytmp[feasible, :]
            if f is not None:
                f = f[feasible]
            c = c[feasible, :]
            if epochs is not None:
                epochs = epochs[feasible]
    else:
        feasible = None

    perm_x, perm_y, rank, crowd, perm = MOEA.sortMO(
        xtmp, ytmp, nInput, nOutput, return_perm=True
    )
    # x, y are already permutated upon return
    perm_f = f[perm]
    perm_epoch = epochs[perm]
    perm_c = c[perm]

    uniq_rank, rnk_inv, rnk_cnt = np.unique(
        rank, return_inverse=True, return_counts=True
    )

    collect_idx = [[] for i in uniq_rank]
    for idx, rnk in enumerate(rnk_inv):
        collect_idx[rnk].append(idx)

    rank_idx = np.array(collect_idx, dtype=np.ndarray)
    for idx, i in enumerate(rank_idx):
        rank_idx[idx] = np.array(i)

    uniq_epc, epc_inv, epc_cnt = np.unique(
        perm_epoch, return_inverse=True, return_counts=True
    )

    collect_epoch = [[] for i in uniq_epc]
    for idx, epc in enumerate(epc_inv):
        collect_epoch[epc].append(idx)
    epc_idx = np.array(collect_epoch, dtype=np.ndarray)
    for idx, i in enumerate(epc_idx):
        epc_idx[idx] = np.array(i)

    rnk_epc_idx = np.empty(
        shape=(uniq_rank.shape[0], uniq_epc.shape[0]), dtype=np.ndarray
    )

    for idx, i in enumerate(rank_idx):
        for jidx, j in enumerate(epc_idx):
            rnk_epc_idx[idx, jidx] = np.intersect1d(i, j, assume_unique=True)

    perm_arrs = (perm_x, perm_y, perm_f, perm_epoch, perm, feasible)
    rnk_arrs = (uniq_rank, rank_idx, rnk_cnt)
    epc_arrs = (uniq_epc, epc_idx, epc_cnt)

    return perm_arrs, rnk_arrs, epc_arrs, rnk_epc_idx
