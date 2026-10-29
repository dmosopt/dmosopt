import logging
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


def solution_quality(
    objective_values: np.ndarray, epsilon: float = 0.01, num_pareto_points: int = 1000
) -> dict:
    """
    Measure how close objective vectors lie to the ZDT1 Pareto front.

    Args:
        objective_values: Array of objective vectors, one per row
        epsilon: Maximum allowed distance from Pareto front
        num_pareto_points: Number of points to generate for Pareto front
    Returns:
        Dictionary containing test results
    """
    objective_values = np.asarray(objective_values)

    # Get analytical Pareto front
    pareto_front = zdt1_pareto(num_pareto_points)

    # Calculate minimum distance to Pareto front for each solution
    distances = []
    for point in objective_values:
        dist = np.min(np.sqrt(np.sum((pareto_front - point) ** 2, axis=1)))
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


def best_objectives(best, objective_names):
    """Stack the best objective values returned by dmosopt.run into an
    array with one row per solution."""
    besty = dict(best[1])
    return np.column_stack([besty[name] for name in objective_names])


def run_zdt1_nsga2_trs():
    """Optimize ZDT1 with alternating NSGA-II and TRS epochs and return
    the best solutions."""
    space = {}
    for i in range(30):
        space["x%d" % (i + 1)] = [0.0, 1.0]
    problem_parameters = {}
    objective_names = ["y1", "y2"]

    # Create an optimizer
    dmosopt_params = {
        "opt_id": "zdt1_nsga2_trs",
        "obj_fun_name": f"{__name__}.obj_fun",
        "problem_parameters": problem_parameters,
        "space": space,
        "objective_names": objective_names,
        "population_size": 200,
        "num_generations": 100,
        "initial_maxiter": 10,
        "surrogate_method_name": "gpr",
        "optimizer_name": ["nsga2", "trs"],
        "optimizer_kwargs": [
            {
                "crossover_prob": 0.9,
                "mutation_prob": 0.1,
                "adaptive_population_size": False,
            },
            {},
        ],
        "termination_conditions": True,
        "optimize_mean_variance": False,
        "n_initial": 3,
        "n_epochs": 4,
        "save": False,
        "save_surrogate_eval": False,
    }

    return dmosopt.run(dmosopt_params, verbose=True)


def test_zdt1_nsga2_trs():
    """
    NSGA-II/TRS on ZDT1 (30 inputs) with the GPR surrogate.

    The test checks that the best solutions returned by the optimizer have
    converged toward the analytical front f2 = 1 - sqrt(f1). The quality
    measure uses the best (non-dominated) set rather than all evaluations,
    since the initial space-filling samples lie far from the front.

    With 4 epochs the residual g(x) - 1 term keeps the best set a few
    hundredths away from the front: across repeated runs the mean distance
    of the best set was 0.010 - 0.032 and the largest single distance was
    0.075. The tolerance of 0.1 gives about three times headroom over the
    worst observed mean, while still failing if the search does not
    approach the front (random samples lie at distances of 2 - 4).
    """
    best = run_zdt1_nsga2_trs()
    assert best is not None, "ZDT1 NSGA2/TRS: optimization returned no solutions."

    epsilon = 0.1
    quality = solution_quality(best_objectives(best, ["y1", "y2"]), epsilon=epsilon)
    logger.info(f"ZDT1 NSGA2/TRS solution quality: {quality}")
    assert quality["mean_distance"] < epsilon, (
        "ZDT1 NSGA2/TRS: best solutions are too far from the Pareto front "
        f"(mean dist={quality['mean_distance']:.4f}, limit {epsilon})."
    )
    assert quality["num_solutions_on_front"] >= 20, (
        "ZDT1 NSGA2/TRS: too few best solutions near the Pareto front "
        f"({quality['num_solutions_on_front']} within {epsilon}, need at least 20)."
    )


if __name__ == "__main__":
    test_zdt1_nsga2_trs()
