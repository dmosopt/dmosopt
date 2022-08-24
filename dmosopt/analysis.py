from SALib.sample import saltelli
from SALib.analyze import sobol

bounds = list(zip(lo_bounds, hi_bounds))

problem = {"num_vars": len(param_names), "names": param_names, "bounds": bounds}

logger.info(f"problem: {problem}")

# Generate samples
param_values = saltelli.sample(problem, 8192, calc_second_order=True)

# Run model (example)
Y = sm.evaluate(param_values)
