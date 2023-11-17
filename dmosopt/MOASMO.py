# Multi-Objective Adaptive Surrogate Model-based Optimization

import sys, itertools
import numpy as np
from numpy.random import default_rng
from typing import Any, Union, Dict, List, Tuple, Optional
from dmosopt import MOEA, NSGA2, AGEMOEA, SMPSO, CMAES, model, sa, sampling
from dmosopt.feasibility import LogisticFeasibilityModel
from dmosopt.datatypes import OptHistory, EpochResults


def optimize(
    num_generations,
    optimizer,
    model,
    nInput,
    nOutput,
    xlb,
    xub,
    popsize=100,
    initial=None,
    feasibility_model=False,
    termination=None,
    local_random=None,
    logger=None,
    **kwargs,
):
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
    """

    optimizer_kwargs = {}
    optimizer_kwargs.update(kwargs)

    if local_random is None:
        local_random = default_rng()

    bounds = np.column_stack((xlb, xub))

    x = optimizer.generate_initial(bounds, local_random)
    if model is None:
        y = yield x
    else:
        y = model.evaluate(x).astype(np.float32)

    x_initial = None
    y_initial = None
    if initial is not None:
        x_initial, y_initial = initial

    if x_initial is not None:
        x = np.vstack((x_initial.astype(np.float32), x))
    if y_initial is not None:
        y = np.vstack((y_initial.astype(np.float32), y))

    optimizer.initialize_strategy(x, y, bounds, local_random, **optimizer_kwargs)
    if logger is not None:
        logger.info(
            f"{optimizer.name}: optimizer parameters are {repr(optimizer.opt_params)}"
        )

    gen_indexes = []
    gen_indexes.append(np.zeros((x.shape[0],), dtype=np.uint32))

    x_new = []
    y_new = []

    n_eval = 0
    it = range(1, num_generations + 1)
    if termination is not None:
        it = itertools.count(1)
    for i in it:
        if termination is not None:
            pop_x, pop_y = optimizer.population_objectives
            opt = OptHistory(i, n_eval, pop_x, pop_y, None)
            if termination.has_terminated(opt):
                break
        if logger is not None:
            if termination is not None:
                logger.info(f"{optimizer.name}: generation {i}...")
            else:
                logger.info(f"{optimizer.name}: generation {i} of {gen}...")

        ## optimizer generate-update
        x_gen, state_gen = optimizer.generate()

        if model is None:
            y_gen = yield x_gen
        else:
            y_gen = model.evaluate(x_gen)

        optimizer.update(x_gen, y_gen, state_gen)
        count = x_gen.shape[0]
        n_eval += count

        x_new.append(x_gen)
        y_new.append(y_gen)
        gen_indexes.append(np.ones((x_gen.shape[0],), dtype=np.uint32) * i)

    gen_index = np.concatenate(gen_indexes)
    x = np.vstack([x] + x_new)
    y = np.vstack([y] + y_new)
    bestx, besty = optimizer.population_objectives

    results = EpochResults(bestx, besty, gen_index, x, y, optimizer)

    return results


def xinit(
    nEval,
    param_names,
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
            in_bounds = np.all(
                np.logical_and(Xinit[:, i] <= xub[i], Xinit[:, i] >= xlb[i])
            )
            if not in_bounds:
                logger.error(
                    f"xinit: out of bounds values detected for parameter {param_names[i]}"
                )
            assert in_bounds
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
        raise RuntimeError(f"Unknown method {method}")

    Xinit = Xinit[nPrevious:, :] * (xub - xlb) + xlb

    return Xinit


def epoch(
    num_generations,
    param_names,
    objective_names,
    xlb,
    xub,
    pct,
    Xinit,
    Yinit,
    C,
    pop=100,
    sampling_method=None,
    feasibility_model=False,
    optimizer_name="nsga2",
    optimizer_kwargs={},
    surrogate_method="gpr",
    surrogate_options={"anisotropic": False, "optimizer": "sceua"},
    sensitivity_method=None,
    sensitivity_options={},
    termination=None,
    local_random=None,
    logger=None,
):
    """
    Multi-Objective Adaptive Surrogate Modelling-based Optimization
    Performs one epoch of optimization.


    xlb: lower bound of input
    xub: upper bound of input
    pct: percentage of resampled points in each iteration
    Xinit and Yinit: initial samplers for surrogate model construction
    ### options for the embedded NSGA-II:
        pop: number of population
        num_generations: number of generation
        crossover_prob: probability of crossover in each generation
        mutation_prob: probability of mutation in each generation
        di_crossover: distribution index for crossover
        di_mutation: distribution index for mutation
    """

    nInput = len(param_names)
    nOutput = len(objective_names)

    N_resample = int(pop * pct)

    if Xinit is None:
        Xinit, Yinit = yield

    x = Xinit.copy().astype(np.float32)
    y = Yinit.copy().astype(np.float32)

    fsbm = None
    if C is not None:
        feasible = np.argwhere(np.all(C > 0.0, axis=1))
        if len(feasible) > 0:
            feasible = feasible.ravel()
            try:
                if feasibility_model:
                    logger.info(f"Constructing feasibility model...")
                    fsbm = LogisticFeasibilityModel(Xinit, C)
                x = x[feasible, :]
                y = y[feasible, :]
            except:
                e = sys.exc_info()[0]
                logger.warning(f"Unable to fit feasibility model: {e}")

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

    optimizer_kwargs_ = {
        "sampling_method": "slh",
        "mutation_rate": None,
        "nchildren": 1,
    }
    optimizer_kwargs_.update(optimizer_kwargs)

    if sensitivity_method is not None:
        di_dict = analyze_sensitivity(
            sm,
            xlb,
            xub,
            param_names,
            objective_names,
            sensitivity_method=sensitivity_method,
            sensitivity_options=sensitivity_options,
            logger=logger,
        )
        optimizer_kwargs_["di_mutation"] = di_dict["di_mutation"]
        optimizer_kwargs_["di_crossover"] = di_dict["di_crossover"]

    if optimizer_name == "nsga2":
        optimizer = NSGA2.NSGA2(
            nInput=nInput,
            nOutput=nOutput,
            popsize=pop,
            feasibility_model=fsbm,
            distance_metric=None,
            **optimizer_kwargs_,
        )
    elif optimizer_name == "age":
        optimizer = AGEMOEA.AGEMOEA(
            nInput=nInput,
            nOutput=nOutput,
            popsize=pop,
            feasibility_model=fsbm,
            **optimizer_kwargs_,
        )
    elif optimizer_name == "smpso":
        optimizer = SMPSO.SMPSO(
            nInput=nInput,
            nOutput=nOutput,
            popsize=pop,
            feasibility_model=fsbm,
            distance_metric=None,
            **optimizer_kwargs_,
        )
    elif optimizer_name == "cmaes":
        optimizer = CMAES.CMAES(
            nInput=nInput,
            nOutput=nOutput,
            popsize=pop,
            feasibility_model=fsbm,
            **optimizer_kwargs_,
        )
    else:
        raise RuntimeError(f"Unknown optimizer {optimizer_name}")

    opt_gen = optimize(
        num_generations,
        optimizer,
        sm,
        nInput,
        nOutput,
        xlb,
        xub,
        initial=(x, y),
        feasibility_model=fsbm,
        logger=logger,
        popsize=pop,
        local_random=local_random,
        termination=termination,
        **optimizer_kwargs_,
    )

    try:
        res = next(opt_gen)
    except StopIteration as ex:
        opt_gen.close()
        opt_gen = None
        res = ex.args[0]
        best_x = res.best_x
        best_y = res.best_y
        gen_index = res.gen_index
        x = res.x
        y = res.y
    else:
        while True:
            x_gen = res

            if sm is not None:
                y_gen = sm.evaluate(x_gen)
            else:
                _, y_gen = yield x_gen

            try:
                res = opt_gen.send(y_gen)
            except StopIteration as ex:
                opt_gen.close()
                opt_gen = None
                res = ex.args[0]
                best_x = res.best_x
                best_y = res.best_y
                gen_index = res.gen_index
                x = res.x
                y = res.y
                break

    if surrogate_method is not None:
        D = MOEA.crowding_distance(best_y)
        idxr = D.argsort()[::-1][:N_resample]
        x_resample = best_x[idxr, :]
        y_pred = best_y[idxr, :]

        return_dict = {
            "x_resample": x_resample,
            "y_pred": y_pred,
            "gen_index": gen_index,
            "x_sm": x,
            "y_sm": y,
            "optimizer": optimizer,
        }
    else:
        return_dict = {
            "best_x": best_x,
            "best_y": best_y,
            "gen_index": gen_index,
            "x": x,
            "y": y,
            "optimizer": optimizer,
        }

    return return_dict


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
            x = x[feasible, :]
            y = y[feasible, :]
            if logger is not None:
                logger.info(f"Found {len(feasible)} feasible solutions")

    x, y = MOEA.remove_duplicates(x, y)

    if surrogate_method == "gpr":
        gpr_anisotropic = surrogate_options.get("anisotropic", False)
        gpr_optimizer = surrogate_options.get("optimizer", "sceua")
        gpr_lengthscale_bounds = surrogate_options.get(
            "lengthscale_bounds", (1e-3, 100.0)
        )
        sm = model.GPR_Matern(
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
    elif surrogate_method == "egp":
        egp_lengthscale_bounds = surrogate_options.get("lengthscale_bounds", None)
        egp_likelihood_sigma = surrogate_options.get("likelihood_sigma", 1.0e-4)
        egp_adam_lr = surrogate_options.get("adam_lr", 0.01)
        egp_n_iter = surrogate_options.get("n_iter", 5000)
        egp_cuda = surrogate_options.get("cuda", False)
        egp_fast_pred_var = surrogate_options.get("fast_pred_var", True)
        egp_batch_size = surrogate_options.get("batch_size", None)
        egp_min_loss_pct_change = surrogate_options.get("min_loss_pct_change", 1.0)
        sm = model.EGP_Matern(
            x,
            y,
            nInput,
            nOutput,
            x.shape[0],
            xlb,
            xub,
            gp_lengthscale_bounds=egp_lengthscale_bounds,
            gp_likelihood_sigma=egp_likelihood_sigma,
            adam_lr=egp_adam_lr,
            n_iter=egp_n_iter,
            fast_pred_var=egp_fast_pred_var,
            batch_size=egp_batch_size,
            use_cuda=egp_cuda,
            min_loss_pct_change=egp_min_loss_pct_change,
            logger=logger,
        )
    elif surrogate_method == "megp":
        megp_lengthscale_bounds = surrogate_options.get("lengthscale_bounds", None)
        megp_likelihood_sigma = surrogate_options.get("likelihood_sigma", 1.0e-4)
        megp_adam_lr = surrogate_options.get("adam_lr", 0.01)
        megp_n_iter = surrogate_options.get("n_iter", 5000)
        megp_cuda = surrogate_options.get("cuda", False)
        megp_fast_pred_var = surrogate_options.get("fast_pred_var", True)
        megp_batch_size = surrogate_options.get("batch_size", None)
        megp_min_loss_pct_change = surrogate_options.get("min_loss_pct_change", 0.1)
        sm = model.MEGP_Matern(
            x,
            y,
            nInput,
            nOutput,
            xlb,
            xub,
            gp_lengthscale_bounds=megp_lengthscale_bounds,
            gp_likelihood_sigma=megp_likelihood_sigma,
            adam_lr=megp_adam_lr,
            n_iter=megp_n_iter,
            fast_pred_var=megp_fast_pred_var,
            batch_size=megp_batch_size,
            use_cuda=megp_cuda,
            min_loss_pct_change=megp_min_loss_pct_change,
            logger=logger,
        )
    elif surrogate_method == "mdgp":
        mdgp_num_hidden_dims = surrogate_options.get("num_hidden_dims", 3)
        mdgp_num_inducing_points = surrogate_options.get("num_inducing_points", 128)
        mdgp_lengthscale_bounds = surrogate_options.get("lengthscale_bounds", None)
        mdgp_likelihood_sigma = surrogate_options.get("likelihood_sigma", 1.0e-4)
        mdgp_adam_lr = surrogate_options.get("adam_lr", 0.01)
        mdgp_n_iter = surrogate_options.get("n_iter", 2000)
        mdgp_cuda = surrogate_options.get("cuda", False)
        mdgp_fast_pred_var = surrogate_options.get("fast_pred_var", True)
        mdgp_batch_size = surrogate_options.get("batch_size", 10)
        mdgp_min_loss_pct_change = surrogate_options.get("mdgp_loss_pct_change", 1.0)
        sm = model.MDGP_Matern(
            x,
            y,
            nInput,
            nOutput,
            xlb,
            xub,
            num_hidden_dims=mdgp_num_hidden_dims,
            num_inducing_points=mdgp_num_inducing_points,
            gp_lengthscale_bounds=mdgp_lengthscale_bounds,
            gp_likelihood_sigma=mdgp_likelihood_sigma,
            adam_lr=mdgp_adam_lr,
            n_iter=mdgp_n_iter,
            fast_pred_var=mdgp_fast_pred_var,
            batch_size=mdgp_batch_size,
            use_cuda=mdgp_cuda,
            min_loss_pct_change=mdgp_min_loss_pct_change,
            logger=logger,
        )
    elif surrogate_method == "mdspp":
        mdspp_num_hidden_dims = surrogate_options.get("num_hidden_dims", 3)
        mdspp_num_inducing_points = surrogate_options.get("num_inducing_points", 128)
        mdspp_lengthscale_bounds = surrogate_options.get("lengthscale_bounds", None)
        mdspp_likelihood_sigma = surrogate_options.get("likelihood_sigma", 1.0e-4)
        mdspp_adam_lr = surrogate_options.get("adam_lr", 0.1)
        mdspp_n_iter = surrogate_options.get("n_iter", 2000)
        mdspp_cuda = surrogate_options.get("cuda", False)
        mdspp_fast_pred_var = surrogate_options.get("fast_pred_var", True)
        mdspp_batch_size = surrogate_options.get("batch_size", 10)
        mdspp_min_loss_pct_change = surrogate_options.get("min_loss_pct_change", 1.0)
        sm = model.MDSPP_Matern(
            x,
            y,
            nInput,
            nOutput,
            xlb,
            xub,
            num_hidden_dims=mdspp_num_hidden_dims,
            num_inducing_points=mdspp_num_inducing_points,
            gp_lengthscale_bounds=mdspp_lengthscale_bounds,
            gp_likelihood_sigma=mdspp_likelihood_sigma,
            adam_lr=mdspp_adam_lr,
            n_iter=mdspp_n_iter,
            fast_pred_var=mdspp_fast_pred_var,
            batch_size=mdspp_batch_size,
            use_cuda=mdspp_cuda,
            min_loss_pct_change=mdspp_min_loss_pct_change,
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
        vgp_min_elbo_pct_change = surrogate_options.get("min_elbo_pct_change", 1.0)
        sm = model.VGP_Matern(
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
            min_elbo_pct_change=vgp_min_elbo_pct_change,
            logger=logger,
        )
    elif surrogate_method == "svgp":
        svgp_batch_size = surrogate_options.get("batch_size", 50)
        svgp_lengthscale_bounds = surrogate_options.get(
            "lengthscale_bounds", (1e-6, 100.0)
        )
        svgp_likelihood_sigma = surrogate_options.get("likelihood_sigma", 1.0e-4)
        svgp_natgrad_gamma = surrogate_options.get("natgrad_gamma", 0.1)
        svgp_adam_lr = surrogate_options.get("adam_lr", 0.01)
        svgp_n_iter = surrogate_options.get("n_iter", 30000)
        svgp_min_elbo_pct_change = surrogate_options.get("min_elbo_pct_change", 1.0)
        sm = model.SVGP_Matern(
            x,
            y,
            nInput,
            nOutput,
            x.shape[0],
            xlb,
            xub,
            batch_size=svgp_batch_size,
            gp_lengthscale_bounds=svgp_lengthscale_bounds,
            gp_likelihood_sigma=svgp_likelihood_sigma,
            natgrad_gamma=svgp_natgrad_gamma,
            adam_lr=svgp_adam_lr,
            n_iter=svgp_n_iter,
            min_elbo_pct_change=svgp_min_elbo_pct_change,
            logger=logger,
        )
    elif surrogate_method == "spv":
        spv_batch_size = surrogate_options.get("batch_size", 50)
        spv_lengthscale_bounds = surrogate_options.get(
            "lengthscale_bounds", (1e-6, 100.0)
        )
        spv_likelihood_sigma = surrogate_options.get("likelihood_sigma", 1.0e-4)
        spv_natgrad_gamma = surrogate_options.get("natgrad_gamma", 0.1)
        spv_adam_lr = surrogate_options.get("adam_lr", 0.01)
        spv_n_iter = surrogate_options.get("n_iter", 30000)
        spv_num_latent_gps = surrogate_options.get("num_latent_gps", None)
        spv_min_elbo_pct_change = surrogate_options.get("min_elbo_pct_change", 0.1)
        sm = model.SPV_Matern(
            x,
            y,
            nInput,
            nOutput,
            x.shape[0],
            xlb,
            xub,
            batch_size=spv_batch_size,
            gp_lengthscale_bounds=spv_lengthscale_bounds,
            gp_likelihood_sigma=spv_likelihood_sigma,
            natgrad_gamma=spv_natgrad_gamma,
            adam_lr=spv_adam_lr,
            n_iter=spv_n_iter,
            num_latent_gps=spv_num_latent_gps,
            min_elbo_pct_change=spv_min_elbo_pct_change,
            logger=logger,
        )
    elif surrogate_method == "siv":
        siv_batch_size = surrogate_options.get("batch_size", 50)
        siv_lengthscale_bounds = surrogate_options.get(
            "lengthscale_bounds", (1e-6, 100.0)
        )
        siv_likelihood_sigma = surrogate_options.get("likelihood_sigma", 1.0e-4)
        siv_natgrad_gamma = surrogate_options.get("natgrad_gamma", 0.1)
        siv_adam_lr = surrogate_options.get("adam_lr", 0.01)
        siv_n_iter = surrogate_options.get("n_iter", 30000)
        siv_num_latent_gps = surrogate_options.get("num_latent_gps", None)
        siv_min_elbo_pct_change = surrogate_options.get("min_elbo_pct_change", 0.1)
        sm = model.SIV_Matern(
            x,
            y,
            nInput,
            nOutput,
            x.shape[0],
            xlb,
            xub,
            batch_size=siv_batch_size,
            gp_lengthscale_bounds=siv_lengthscale_bounds,
            gp_likelihood_sigma=siv_likelihood_sigma,
            natgrad_gamma=siv_natgrad_gamma,
            adam_lr=siv_adam_lr,
            n_iter=siv_n_iter,
            num_latent_gps=siv_num_latent_gps,
            min_elbo_pct_change=siv_min_elbo_pct_change,
            logger=logger,
        )
    elif surrogate_method == "crv":
        crv_batch_size = surrogate_options.get("batch_size", 50)
        crv_lengthscale_bounds = surrogate_options.get(
            "lengthscale_bounds", (1e-6, 100.0)
        )
        crv_likelihood_sigma = surrogate_options.get("likelihood_sigma", 1.0e-4)
        crv_natgrad_gamma = surrogate_options.get("natgrad_gamma", 0.1)
        crv_adam_lr = surrogate_options.get("adam_lr", 0.01)
        crv_n_iter = surrogate_options.get("n_iter", 30000)
        crv_num_latent_gps = surrogate_options.get("num_latent_gps", None)
        crv_min_elbo_pct_change = surrogate_options.get("min_elbo_pct_change", 0.1)
        sm = model.CRV_Matern(
            x,
            y,
            nInput,
            nOutput,
            x.shape[0],
            xlb,
            xub,
            batch_size=crv_batch_size,
            gp_lengthscale_bounds=crv_lengthscale_bounds,
            gp_likelihood_sigma=crv_likelihood_sigma,
            natgrad_gamma=crv_natgrad_gamma,
            adam_lr=crv_adam_lr,
            n_iter=crv_n_iter,
            num_latent_gps=crv_num_latent_gps,
            min_elbo_pct_change=crv_min_elbo_pct_change,
            logger=logger,
        )
    else:
        raise RuntimeError(f"Unknown surrogate method {surrogate_method}")

    return sm


def analyze_sensitivity(
    sm,
    xlb,
    xub,
    param_names,
    objective_names,
    sensitivity_method=None,
    sensitivity_options={},
    di_min=1.0,
    di_max=20.0,
    logger=None,
):
    di_mutation = None
    di_crossover = None
    if sensitivity_method is not None:
        if sensitivity_method == "dgsm":
            sens = sa.SA_DGSM(xlb, xub, param_names, objective_names)
            sens_results = sens.analyze(sm)
            S1s = np.vstack(
                list(
                    [
                        sens_results["S1"][objective_name]
                        for objective_name in objective_names
                    ]
                )
            )
            S1max = np.max(S1s, axis=0)
            S1nmax = S1max / np.max(S1max)
            di_mutation = np.clip(S1nmax * di_max, di_min, None)
            di_crossover = np.clip(S1nmax * di_max, di_min, None)
        elif sensitivity_method == "fast":
            sens = sa.SA_FAST(xlb, xub, param_names, objective_names)
            sens_results = sens.analyze(sm)
            S1s = np.vstack(
                list(
                    [
                        sens_results["S1"][objective_name]
                        for objective_name in objective_names
                    ]
                )
            )
            S1max = np.max(S1s, axis=0)
            S1nmax = S1max / np.max(S1max)
            di_mutation = np.clip(S1nmax * di_max, di_min, None)
            di_crossover = np.clip(S1nmax * di_max, di_min, None)
        else:
            RuntimeError(f"Unknown sensitivity method {sensitivity_method}")

    if logger is not None:
        logger.info(f"analyze_sensitivity: di_mutation = {di_mutation}")
        logger.info(f"analyze_sensitivity: di_crossover = {di_crossover}")
    di_dict = {}
    di_dict["di_mutation"] = di_mutation
    di_dict["di_crossover"] = di_crossover

    return di_dict


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
    delete_duplicates=True,
):
    xtmp = x
    ytmp = y

    if feasible and c is not None:
        feasible = np.argwhere(np.all(c > 0.0, axis=1)).ravel()
        if len(feasible) > 0:
            feasible = feasible.ravel()
            xtmp = x[feasible, :]
            ytmp = y[feasible, :]
            if f is not None:
                f = f[feasible]
            c = c[feasible, :]
            if epochs is not None:
                epochs = epochs[feasible]

    if delete_duplicates:
        is_duplicate = MOEA.get_duplicates(ytmp)

        xtmp = xtmp[~is_duplicate]
        ytmp = ytmp[~is_duplicate]
        if f is not None:
            f = f[~is_duplicate]
        if c is not None:
            c = c[~is_duplicate]

    xtmp, ytmp, rank, _, perm = MOEA.sortMO(
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

    perm_x, perm_y, rank, _, perm = MOEA.sortMO(
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
