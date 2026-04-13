import logging
import numpy as np
from dmosopt import dmosopt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def zdt6(x):
    """Zitzler-Deb-Thiele Function - type 6

    ZDT6 is designed to test:
    - Non-uniform distribution of Pareto optimal solutions
    - Low density of solutions near the Pareto front
    - Non-convex Pareto front

    Bound: XUB = [1,1,...]; XLB = [0,0,...]
    dim = 10 (standard, though can be varied)

    Pareto optimal solutions: x1 varies in [0,1], x2...xn = 0
    This means optimal solutions are at the BOUNDARY for x[1:],
    making it a good test for boundary handling with min_sigma.
    """
    num_variables = len(x)
    f = np.zeros(2)

    # f1 has non-uniform density (hard to sample)
    f[0] = 1.0 - np.exp(-4.0 * x[0]) * (np.sin(6.0 * np.pi * x[0]) ** 6)

    # g depends on other variables
    g = 1.0 + 9.0 * (np.sum(x[1:]) / float(num_variables - 1)) ** 0.25

    # Non-convex Pareto front
    h = 1.0 - (f[0] / g) ** 2

    f[1] = g * h

    return f


def obj_fun(pp):
    """Objective function to be minimized."""
    param_values = np.asarray([pp[k] for k in sorted(pp)])
    res = zdt6(param_values)
    logger.info(f"Iter: \t pp:{pp}, result:{res}")
    return res


def zdt6_pareto(n_points=100):
    """Analytical Pareto front for ZDT6.

    The Pareto optimal set is:
    - x1 in [0, 1] (parameterizes the front)
    - x2 = x3 = ... = xn = 0 (at boundary!)

    The Pareto front in objective space is:
    - f1 = 1 - exp(-4*x1) * sin^6(6*pi*x1)
    - f2 = 1 - f1^2

    Note: f1 ranges approximately [0.28, 1.0] due to the sin^6 term,
    creating non-uniform density.
    """
    # Sample x1 to get f1 values
    x1 = np.linspace(0, 1, n_points)
    f1 = 1.0 - np.exp(-4.0 * x1) * (np.sin(6.0 * np.pi * x1) ** 6)

    # Compute f2 from f1 (when g=1, which is optimal)
    f2 = 1.0 - f1**2

    f = np.zeros([n_points, 2])
    f[:, 0] = f1
    f[:, 1] = f2

    return f


if __name__ == "__main__":
    # ZDT6 typically uses 10 variables (vs 30 for ZDT1)
    space = {}
    for i in range(10):
        space["x%d" % (i + 1)] = [0.0, 1.0]

    problem_parameters = {}
    objective_names = ["y1", "y2"]

    # Create an optimizer
    dmosopt_params = {
        "opt_id": "dmosopt_zdt6",
        "obj_fun_name": "example_dmosopt_zdt6.obj_fun",
        "problem_parameters": problem_parameters,
        "space": space,
        "objective_names": objective_names,
        "population_size": 200,
        "num_generations": 100,
        "initial_maxiter": 10,
        "optimizer_name": "age",
        "surrogate_method_name": None,
        "termination_conditions": True,
        "optimize_mean_variance": False,
        "n_initial": 3,
        "n_epochs": 2,
        "save_surrogate_eval": True,
        "save": True,
        "file_path": "results/zdt6.h5",
    }

    best = dmosopt.run(dmosopt_params, verbose=True)
    if best is not None:
        import matplotlib.pyplot as plt

        bestx, besty = best
        x, y = dmosopt.dopt_dict["dmosopt_zdt6"].optimizer_dict[0].get_evals()
        besty_dict = dict(besty)

        # Plot results
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(y[:, 0], y[:, 1], "b.", label="evaluated points", alpha=0.5)
        ax.plot(
            besty_dict["y1"], besty_dict["y2"], "r.", label="best points", markersize=4
        )

        y_true = zdt6_pareto()
        ax.plot(y_true[:, 0], y_true[:, 1], "k-", linewidth=2, label="True Pareto")

        ax.set_xlabel("f1")
        ax.set_ylabel("f2")
        ax.set_title("ZDT6: Non-uniform Pareto Front")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("example_dmosopt_zdt6.svg")

        print("\n=== ZDT6 Results ===")
        print(f"Evaluated points: {len(y)}")
        print(f"Best points: {len(besty_dict['y1'])}")
        print(f"f1 range: [{y_true[:, 0].min():.3f}, {y_true[:, 0].max():.3f}]")
        print(f"f2 range: [{y_true[:, 1].min():.3f}, {y_true[:, 1].max():.3f}]")
