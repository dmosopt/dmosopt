import sys, logging
import numpy as np
from dmosopt import dmosopt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def tnk(x):
    """Tanaka, Watanabe, Furukawa, & Tanino, 1995.
    Bound: XUB = [pi, pi]; XLB = [0, 0]
    dim = 2
    """
    f = np.zeros(2)
    c = np.zeros(2)
    f[0] = x[0]
    f[1] = x[1]
    c[0] = -(
        np.square(x[0])
        + np.square(x[1])
        - 1.0
        - 0.1 * np.cos(16.0 * np.arctan(x[0] / x[1]))
    )
    c[1] = 2 * (np.square(x[0] - 0.5) + np.square(x[1] - 0.5)) - 1

    return f, c


def obj_fun(pp):
    """Objective function to be minimized."""
    param_values = np.asarray([pp[k] for k in sorted(pp)])
    res = tnk(param_values)
    logger.info(f"Iter: \t pp:{pp}, result:{res}")
    return res


def tnk_pareto(n_points=100):
    f = np.zeros([n_points * n_points, 2])
    x = np.zeros([n_points * n_points, 2])
    c = np.zeros([n_points * n_points, 2])

    x_mesh = np.meshgrid(
        np.linspace(0, np.pi, n_points), np.linspace(0, np.pi, n_points)
    )
    x[:, 0] = x_mesh[0].flat
    x[:, 1] = x_mesh[1].flat
    f[:, 0] = x[:, 0]
    f[:, 1] = x[:, 1]

    c[:, 0] = -(
        np.square(x[:, 0])
        + np.square(x[:, 1])
        - 1.0
        - 0.1 * np.cos(16.0 * np.arctan(x[:, 0] / x[:, 1]))
    )
    c[:, 1] = 2 * (np.square(x[:, 0] - 0.5) + np.square(x[:, 1] - 0.5)) - 1

    feasible = np.argwhere(np.all(c > 0.0, axis=1))

    return f[feasible.flat, :]


if __name__ == "__main__":
    space = {}
    for i in range(2):
        space["x%d" % (i + 1)] = [0.0, np.pi]
    problem_parameters = {}
    objective_names = ["y1", "y2"]
    constraint_names = ["c1", "c2"]

    # Create an optimizer
    dmosopt_params = {
        "opt_id": "dmosopt_tnk",
        "obj_fun_name": "example_dmosopt_tnk.obj_fun",
        "problem_parameters": problem_parameters,
        "space": space,
        "objective_names": objective_names,
        "constraint_names": constraint_names,
        "population_size": 200,
        "num_generations": 100,
        "initial_maxiter": 10,
        "surrogate_method_name": "gpr",
        "optimizer_name": "nsga2",
        "optimizer_kwargs": [
            {
                "crossover_prob": 0.9,
                "mutation_prob": 0.1,
            },
            {},
        ],
        "termination_conditions": True,
        "n_initial": 3,
        "n_epochs": 5,
        "save_surrogate_eval": True,
        "save": True,
        "file_path": "results/tnk.h5",
    }

    best = dmosopt.run(dmosopt_params, verbose=True)
    if best is not None:
        import matplotlib.pyplot as plt

        bestx, besty = best
        x, y, c = (
            dmosopt.dopt_dict["dmosopt_tnk"]
            .optimizer_dict[0]
            .get_evals(return_constraints=True)
        )
        besty_dict = dict(besty)
        feasible = np.argwhere(np.all(c > 0.0, axis=1))
        # plot results
        plt.plot(
            y[feasible, 0], y[feasible, 1], "b.", label="feasible evaluated points"
        )
        plt.plot(besty_dict["y1"], besty_dict["y2"], "r.", label="best points")

        y_true = tnk_pareto()
        plt.plot(
            y_true[:, 0],
            y_true[:, 1],
            "ko",
            fillstyle="none",
            alpha=0.5,
            label="True Pareto",
        )
        plt.legend()

        plt.savefig("example_dmosopt_tnk.svg")
