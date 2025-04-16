import logging
import numpy as np
from typing import List
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
    group1 = pp["group1"]
    group2 = pp["group2"]
    param_values = np.concatenate(
        (
            np.asarray([group1[k] for k in sorted(group1)]),
            np.asarray([group2[k] for k in sorted(group2)]),
        )
    )

    res = zdt1(param_values)
    logger.info(f"Iter: \t pp:{pp}, result:{res}")
    return res


def zdt1_pareto(n_points=100):
    f = np.zeros([n_points, 2])
    f[:, 0] = np.linspace(0, 1, n_points)
    f[:, 1] = 1.0 - np.sqrt(f[:, 0])
    return f


def test_solution_quality(
    solutions: List[np.ndarray], epsilon: float = 0.01, num_pareto_points: int = 1000
) -> dict:
    """
    Test if solutions lie on or near the Pareto front.

    Args:
        solutions: List of solution vectors
        epsilon: Maximum allowed distance from Pareto front
        num_pareto_points: Number of points to generate for Pareto front
    Returns:
        Dictionary containing test results
    """
    # Calculate objective values for all solutions
    objective_values = np.vstack([zdt1(x) for x in solutions])

    print(f"objective_values.shape = {objective_values.shape}")
    # Get analytical Pareto front
    pareto_front = zdt1_pareto(num_pareto_points)
    print(f"pareto_front.shape = {pareto_front.shape}")

    # Calculate minimum distance to Pareto front for each solution
    distances = []
    for point in objective_values:
        dist = np.min(np.sqrt(np.sum((pareto_front - point) ** 2, axis=1)))
        #        print(f"point = {point} dist = {dist} pareto_front = {pareto_front[:10]}")
        distances.append(dist)

    distances = np.array(distances)

    return {
        "all_solutions_on_front": np.all(distances <= epsilon),
        "mean_distance": float(np.mean(distances)),
        "max_distance": float(np.max(distances)),
        "min_distance": float(np.min(distances)),
        "num_solutions_on_front": int(np.sum(distances <= epsilon)),
        "percent_solutions_on_front": float(np.mean(distances <= epsilon) * 100),
    }


if __name__ == "__main__":
    space = {"group1": {}, "group2": {}}
    for i in range(15):
        space["group1"]["x%d" % (i + 1)] = [0.0, 1.0]
    for i in range(15, 30):
        space["group2"]["x%d" % (i + 1)] = [0.0, 1.0]

    problem_parameters = {}
    objective_names = ["y1", "y2"]

    # Create an optimizer
    dmosopt_params = {
        "opt_id": "dmosopt_zdt1",
        "obj_fun_name": "test_zdt1_age_nested.obj_fun",
        "problem_parameters": problem_parameters,
        "space": space,
        "nested_parameter_space": True,
        "objective_names": objective_names,
        "population_size": 200,
        "num_generations": 100,
        "initial_maxiter": 10,
        "surrogate_method_name": "gpr",
        "optimizer_name": ["age"],
        "optimizer_kwargs": [
            {
                "crossover_prob": 0.9,
                "mutation_prob": 0.1,
                "adaptive_population_size": False,
            },
        ],
        "termination_conditions": True,
        "optimize_mean_variance": False,
        "n_initial": 3,
        "n_epochs": 4,
        "file_path": "./zdt1_nested.h5",
        "save": True,
        "save_surrogate_eval": False,
    }

    best = dmosopt.run(dmosopt_params, verbose=True)
    if best is not None:
        bestx, besty = best
        x, y = dmosopt.dopt_dict["dmosopt_zdt1"].optimizer_dict[0].get_evals()
        solution_quality = test_solution_quality(x)
        print(solution_quality)
        assert solution_quality["num_solutions_on_front"] >= 30
