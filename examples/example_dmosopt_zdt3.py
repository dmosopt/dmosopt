import sys, logging
import numpy as np
from dmosopt import dmosopt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def zdt3(x):
    """This is the Zitzler-Deb-Thiele Function - type A
    Bound: XUB = [1,1,...]; XLB = [0,0,...]
    dim = 30
    """
    num_variables = len(x)
    f = np.zeros(2)
    f[0] = x[0]
    g = 1.0 + 9.0 / float(num_variables - 1) * np.sum(x[1:])
    h = 1.0 - np.sqrt(f[0] / g)
    j = (x[0] / g) * np.sin(10 * np.pi * x[0])
    f[1] = g * h - j
    return f


def obj_fun(pp):
    """Objective function to be minimized."""
    param_values = np.asarray([pp[k] for k in sorted(pp)])
    res = zdt3(param_values)
    logger.info(f"Iter: \t pp:{pp}, result:{res}")
    return res


def zdt3_pareto(n_points=100, flatten=True):
    regions = [
        [0, 0.0830015349],
        [0.182228780, 0.2577623634],
        [0.4093136748, 0.4538821041],
        [0.6183967944, 0.6525117038],
        [0.8233317983, 0.8518328654],
    ]

    pf = []

    for r in regions:
        x1 = np.linspace(r[0], r[1], int(n_points / len(regions)))
        x2 = 1 - np.sqrt(x1) - x1 * np.sin(10 * np.pi * x1)
        pf.append(np.array([x1, x2]).T)

    if not flatten:
        pf = np.concatenate([pf[None, ...] for pf in pf])
    else:
        pf = np.row_stack(pf)

    return pf


if __name__ == "__main__":
    space = {}
    for i in range(30):
        space["x%d" % (i + 1)] = [0.0, 1.0]
    problem_parameters = {}
    objective_names = ["y1", "y2"]

    # Create an optimizer
    dmosopt_params = {
        "opt_id": "dmosopt_zdt3",
        "obj_fun_name": "obj_fun",
        "obj_fun_module": "example_dmosopt_zdt3",
        "problem_parameters": problem_parameters,
        "optimizer": "nsga2",
        "surrogate_options": {"lengthscale_bounds": (1e-4, 1000.0)},
        "population_size": 200,
        "num_generations": 200,
        "termination_conditions": True,
        "space": space,
        "objective_names": objective_names,
        "n_initial": 5,
        "n_epochs": 4,
        "save_surrogate_eval": True,
        "save": True,
        "file_path": "results/zdt3.h5",
        "resample_fraction": 1.00,
    }

    best = dmosopt.run(dmosopt_params, verbose=True)
    if best is not None:
        import matplotlib.pyplot as plt

        bestx, besty = best
        x, y = dmosopt.sopt_dict["dmosopt_zdt3"].optimizer_dict[0].get_evals()
        besty_dict = dict(besty)

        # plot results
        plt.plot(y[:, 0], y[:, 1], "b.", label="Evaluated points")
        plt.plot(besty_dict["y1"], besty_dict["y2"], "r.", label="Best solutions")

        y_true = zdt3_pareto()
        plt.plot(
            y_true[:, 0],
            y_true[:, 1],
            "ko",
            fillstyle="none",
            alpha=0.5,
            label="True Pareto",
        )
        plt.legend()

        plt.savefig("example_dmosopt_zdt3.svg")
