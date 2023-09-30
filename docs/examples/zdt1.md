# ZDT1

Let's consider a hello-world example using the ZDT1 objective. 

Below, we define our own function and perform minimization.

```python
import sys, logging
import numpy as np
from dmosopt import dmosopt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def zdt1(x):
    """This is the Zitzler-Deb-Thiele Function - type A
    Bound: XUB = [1,1,...]; XLB = [0,0,...]
    dim = 30
    """
    num_variables = len(x)
    f = np.zeros(2)
    f[0] = x[0]
    g = 1.0 + 9.0 / float(num_variables - 1) * np.sum(x[1:])
    h = 1.0 - np.sqrt(f[0] / g)
    f[1] = g * h
    return f


def obj_fun(pp):
    """Objective function to be minimized."""
    param_values = np.asarray([pp[k] for k in sorted(pp)])
    res = zdt1(param_values)
    logger.info(f"Iter: \t pp:{pp}, result:{res}")
    return res


def zdt1_pareto(n_points=100):
    f = np.zeros([n_points, 2])
    f[:, 0] = np.linspace(0, 1, n_points)
    f[:, 1] = 1.0 - np.sqrt(f[:, 0])
    return f


if __name__ == "__main__":
    space = {}
    for i in range(30):
        space["x%d" % (i + 1)] = [0.0, 1.0]
    problem_parameters = {}
    objective_names = ["y1", "y2"]

    # Create an optimizer
    dmosopt_params = {
        "opt_id": "dmosopt_zdt1",
        "obj_fun_name": "obj_fun",
        "obj_fun_module": "example_dmosopt_zdt1",
        "problem_parameters": problem_parameters,
        "space": space,
        "objective_names": objective_names,
        "population_size": 200,
        "num_generations": 200,
        "initial_maxiter": 10,
        "optimizer": "nsga2",
        "termination_conditions": True,
        "n_initial": 3,
        "n_epochs": 2,
    }

    best = dmosopt.run(dmosopt_params, verbose=True)
    if best is not None:
        import matplotlib.pyplot as plt

        bestx, besty = best
        x, y = dmosopt.sopt_dict["dmosopt_zdt1"].optimizer_dict[0].get_evals()
        besty_dict = dict(besty)

        # plot results
        plt.plot(y[:, 0], y[:, 1], "b.", label="evaluated points")
        plt.plot(besty_dict["y1"], besty_dict["y2"], "r.", label="best points")

        y_true = zdt1_pareto()
        plt.plot(y_true[:, 0], y_true[:, 1], "k-", label="True Pareto")
        plt.legend()

        plt.savefig("example_dmosopt_zdt1.svg")

```
