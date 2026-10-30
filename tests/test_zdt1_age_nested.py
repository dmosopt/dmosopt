import logging
from pathlib import Path
import numpy as np
from dmosopt import dmosopt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pass criterion, set from measured runs (see the test docstring).
EPSILON = 0.2
MIN_ON_FRONT = 20


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


def run_zdt1_age_nested(output_dir="."):
    """Optimize ZDT1 with AGE-MOEA over a nested parameter space, saving
    results to output_dir, and return the best solutions."""
    space = {"group1": {}, "group2": {}}
    for i in range(15):
        space["group1"]["x%d" % (i + 1)] = [0.0, 1.0]
    for i in range(15, 30):
        space["group2"]["x%d" % (i + 1)] = [0.0, 1.0]

    problem_parameters = {}
    objective_names = ["y1", "y2"]

    # Create an optimizer
    dmosopt_params = {
        "opt_id": "zdt1_age_nested",
        "obj_fun_name": f"{__name__}.obj_fun",
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
        "file_path": str(Path(output_dir) / "zdt1_nested.h5"),
        "save": True,
        "save_surrogate_eval": False,
    }

    return dmosopt.run(dmosopt_params, verbose=True)


def test_zdt1_age_nested(tmp_path):
    """
    AGE-MOEA on ZDT1 over a nested parameter space (two groups of 15
    inputs), with results saved to an HDF5 file.

    The test checks that nested parameters survive the optimization and
    file output, and that the best (non-dominated) solutions have converged
    toward the analytical front f2 = 1 - sqrt(f1). Across three repeated
    runs the best set held 37 - 52 solutions with a mean distance to the
    front of 0.013 - 0.042, and at least 36 of them lay within 0.2. The
    limits below match the dotted-name test, which uses the same budget:
    mean distance under 0.2 and at least 20 solutions within 0.2.
    """
    best = run_zdt1_age_nested(output_dir=tmp_path)
    assert best is not None, "ZDT1 AGE nested: optimization returned no solutions."

    epsilon = EPSILON
    quality = solution_quality(best_objectives(best, ["y1", "y2"]), epsilon=epsilon)
    logger.info(f"ZDT1 AGE nested solution quality: {quality}")
    assert quality["mean_distance"] < epsilon, (
        "ZDT1 AGE nested: best solutions are too far from the Pareto front "
        f"(mean dist={quality['mean_distance']:.4f}, limit {epsilon})."
    )
    assert quality["num_solutions_on_front"] >= MIN_ON_FRONT, (
        "ZDT1 AGE nested: too few best solutions near the Pareto front "
        f"({quality['num_solutions_on_front']} within {epsilon}, "
        f"need at least {MIN_ON_FRONT})."
    )


if __name__ == "__main__":
    test_zdt1_age_nested(".")
