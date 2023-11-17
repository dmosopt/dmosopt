import sys, logging
import numpy as np
from scipy.integrate import solve_ivp
from dmosopt import dmosopt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def lorenz(t, X, s, r, b):
    """This is the Lorenz Dynamical System"""
    x, y, z = X
    xdot = s * (y - x)
    ydot = x * (r - z) - y
    zdot = x * y - b * z
    return xdot, ydot, zdot


def get_lorenz_target(
    sim_t0=0.0,
    sim_tmax=100.0,
    tar_t0=20.0,
    tar_tmax=100.0,
    tar_dt=0.1,
    p=(10, 28, 8 / 3),
):
    """p = (sigma, rho, beta)
    Discard points in [sim_t0, tar_t0) to
    """
    tar_t0 = max(tar_t0, sim_t0)
    tar_tmax = min(tar_tmax, sim_tmax)
    tar_t = np.arange(tar_t0, tar_tmax, tar_dt)

    target_soln = solve_ivp(
        lorenz,
        (sim_t0, sim_tmax),
        X0,
        args=p,
        method="Radau",
        dense_output=True,
        vectorized=True,
    )

    target_lorenz = target_soln.sol(tar_t)
    return target_lorenz, tar_t


X0 = (-0.5, 1, 0.5)
lorenz_target, lorenz_target_time = get_lorenz_target()


def obj_fun(pp):
    """Objective function to be minimized."""

    t0 = 0
    tmax = 100

    #    param_values = np.asarray([pp[k] for k in sorted(pp)])
    param_values = tuple([pp[k] for k in ["s", "r", "b"]])

    eval_soln = solve_ivp(
        lorenz,
        (t0, tmax),
        X0,
        args=param_values,
        method="Radau",
        dense_output=True,
        vectorized=True,
    )

    eval_X = eval_soln.sol(lorenz_target_time)

    # Using absolute norm wrt time
    res = np.abs(eval_X - lorenz_target).sum(axis=-1)

    logger.info(f"Iter: \t pp:{pp}, result:{res}")
    return res


if __name__ == "__main__":
    space = {"s": [5.0, 15], "r": [15, 35], "b": [1, 10]}
    problem_parameters = {}
    objective_names = ["x", "y", "z"]

    # Create an optimizer
    dmosopt_params = {
        "opt_id": "dmosopt_lorenz",
        "obj_fun_name": "obj_fun",
        "obj_fun_module": "example_dmosopt_lorenz",
        "problem_parameters": problem_parameters,
        "optimizer": "nsga2",
        "population_size": 200,
        "num_generations": 200,
        "optimizer": "nsga2",
        "termination_conditions": True,
        "space": space,
        "objective_names": objective_names,
        "n_initial": 100,
        "n_epochs": 4,
        "save_surrogate_eval": True,
        "save": True,
        "file_path": "results/lorenz.h5",
        "resample_fraction": 1.00,
    }

    best = dmosopt.run(dmosopt_params, verbose=True)
#    if best is not None:
#        import matplotlib.pyplot as plt
#        bestx, besty = best
#        x, y = dmosopt.sopt_dict['dmosopt_zdt3'].optimizer_dict[0].get_evals()
#        besty_dict = dict(besty)
#
#        # plot results
#        plt.plot(y[:,0],y[:,1],'b.',label='Evaluated points')
#        plt.plot(besty_dict['y1'],besty_dict['y2'],'r.',label='Best solutions')
#
#        y_true = zdt3_pareto()
#        plt.plot(y_true[:,0],y_true[:,1],'ko',fillstyle='none',alpha=0.5,label='True Pareto')
#        plt.legend()
#
#        plt.savefig("example_dmosopt_zdt3.svg")
#
