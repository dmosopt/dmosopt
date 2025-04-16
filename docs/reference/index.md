# Reference documentation

This page contains detailed API reference documentation. It is intended to be an in-depth resource for understanding the implementation details of dmosopt's interfaces. You may prefer reviewing the more explanatory [guide](../guide/introduction.md) before consulting this reference.

## dmosopt API Reference

### `dmosopt.run`

```python
dmosopt.run(
    dopt_params,
    time_limit=None,
    feasible=True,
    return_features=False,
    return_constraints=False,
    spawn_workers=False,
    sequential_spawn=False,
    spawn_startup_wait=None,
    spawn_executable=None,
    spawn_args=[],
    nprocs_per_worker=1,
    collective_mode="gather",
    verbose=True,
    worker_debug=False
)
```

The `run` function is the main entry point for running an optimization
using dmosopt. It takes the optimization parameters `dopt_params` and
various configuration options.

**Parameters:**
- `dopt_params` (dict): A dictionary specifying the optimization parameters, including the objective function, parameter space, and optimization settings.
- `time_limit` (float, optional): The time limit for the optimization in seconds. If not provided, the optimization will run until the specified number of epochs is reached.
- `feasible` (bool, optional): If True, only feasible solutions will be returned. Default is True.
- `return_features` (bool, optional): If True, the function will also return the features of the best solution. Default is False.
- `return_constraints` (bool, optional): If True, the function will also return the constraint values of the best solution. Default is False.
- `spawn_workers` (bool, optional): If True, the function will spawn worker processes for distributed optimization. Default is False.
- `sequential_spawn` (bool, optional): If True, worker processes will be spawned sequentially. Default is False.
- `spawn_startup_wait` (float, optional): The waiting time in seconds before spawning the next worker process. Default is None.
- `spawn_executable` (str, optional): The executable to use for spawning worker processes. Default is None.
- `spawn_args` (list, optional): Additional arguments to pass to the spawned worker processes. Default is an empty list.
- `nprocs_per_worker` (int, optional): The number of processes to use per worker. Default is 1.
- `collective_mode` (str, optional): The collective mode for distributed optimization. Can be "gather" or "scatter". Default is "gather".
- `verbose` (bool, optional): If True, verbose output will be enabled. Default is True.
- `worker_debug` (bool, optional): If True, worker debugging mode will be enabled. Default is False.

**Returns:**
- If `return_features` and `return_constraints` are both False, returns a tuple `(best_params, best_objectives)`, where `best_params` is a dictionary of the best parameters found and `best_objectives` is the corresponding objective value(s).
- If `return_features` is True and `return_constraints` is False, returns a tuple `(best_params, best_objectives, best_features)`, where `best_features` is the features of the best solution.
- If `return_features` is False and `return_constraints` is True, returns a tuple `(best_params, best_objectives, best_constraints)`, where `best_constraints` is the constraint values of the best solution.
- If both `return_features` and `return_constraints` are True, returns a tuple `(best_params, best_objectives, best_features, best_constraints)`.

The `run` function distributes the optimization tasks across multiple
workers based on the provided configuration. It initializes the
necessary components, such as the objective function and surrogate
model, and runs the optimization for the specified number of epochs or
until the time limit is reached.

During the optimization, the workers evaluate the objective function
for different parameter configurations, and the results are collected
and processed by the controller. The surrogate model is updated based
on the evaluated points, and new points are selected for evaluation
using the specified sampling strategy.

After the optimization is complete, the `run` function returns the
best parameters found, along with the corresponding objective
value(s), and optionally the features and constraint values if
requested.

### `dmosopt.DistOptimizer`

```python
class dmosopt.DistOptimizer(
    opt_id,
    obj_fun,
    obj_fun_args=None,
    objective_names=None,
    feature_dtypes=None,
    constraint_names=None,
    n_initial=10,
    initial_maxiter=5,
    initial_method="slh",
    dynamic_initial_sampling=None,
    dynamic_initial_sampling_kwargs=None,
    verbose=False,
    reduce_fun=None,
    reduce_fun_args=None,
    problem_ids=None,
    problem_parameters=None,
    space=None,
    population_size=100,
    num_generations=200,
    resample_fraction=0.25,
    distance_metric=None,
    n_epochs=10,
    save_eval=10,
    file_path=None,
    save=False,
    save_surrogate_evals=False,
    save_optimizer_params=True,
    metadata=None,
    nested_parameter_space=False,
    surrogate_method_name="gpr",
    surrogate_method_kwargs={"anisotropic": False, "optimizer": "sceua"},
    surrogate_custom_training=None,
    surrogate_custom_training_kwargs=None,
    optimizer_name="nsga2",
    optimizer_kwargs={
        "mutation_prob": 0.1,
        "crossover_prob": 0.9,
    },
    sensitivity_method_name=None,
    sensitivity_method_kwargs={},
    optimize_mean_variance=False,
    local_random=None,
    random_seed=None,
    feasibility_method_name=None,
    feasibility_method_kwargs=None,
    termination_conditions=None,
    controller=None,
    **kwargs
)
```

The `DistOptimizer` class represents a distributed optimizer object. It encapsulates the optimization parameters, objective function, surrogate model, and other settings required for running an optimization.

**Parameters:**
- `opt_id` (str): A unique identifier for the optimization.
- `obj_fun` (callable): The objective function to be optimized.
- `obj_fun_args` (tuple, optional): Additional arguments to pass to the objective function. Default is None.
- `objective_names` (list of str, optional): Names of the objectives. Default is None.
- `feature_dtypes` (list of tuples, optional): Data types of the features. Default is None.
- `constraint_names` (list of str, optional): Names of the constraints. Default is None.
- `n_initial` (int, optional): Number of initial evaluations. Default is 10.
- `initial_maxiter` (int, optional): Maximum number of iterations for initial evaluations. Default is 5.
- `initial_method` (str, optional): Method for initial evaluations. Default is "slh".
- `dynamic_initial_sampling` (str, optional): Custom dynamic initial sampling function. Default is None.
- `dynamic_initial_sampling_kwargs` (dict, optional): Keyword arguments for the dynamic initial sampling function. Default is None.
- `verbose` (bool, optional): If True, verbose output will be enabled. Default is False.
- `reduce_fun` (callable, optional): Function for reducing evaluation results. Default is None.
- `reduce_fun_args` (tuple, optional): Additional arguments to pass to the reduce function. Default is None.
- `problem_ids` (set, optional): Set of problem IDs. Default is None.
- `problem_parameters` (dict, optional): Problem parameters. Default is None.
- `space` (dict, optional): Parameter space to optimize over. Default is None.
- `population_size` (int, optional): Size of the population in each generation. Default is 100.
- `num_generations` (int, optional): Number of generations to run the optimization. Default is 200.
- `resample_fraction` (float, optional): Fraction of the population to resample in each generation. Default is 0.25.
- `distance_metric` (str or callable, optional): Distance metric for comparing solutions. Default is None.
- `n_epochs` (int, optional): Number of epochs to run the optimization. Default is 10.
- `save_eval` (int, optional): Frequency of saving evaluations. Default is 10.
- `file_path` (str, optional): File path for saving and loading optimization data. Default is None.
- `save` (bool, optional): If True, optimization data will be saved periodically. Default is False.
- `save_surrogate_evals` (bool, optional): If True, surrogate evaluations will be saved. Default is False.
- `save_optimizer_params` (bool, optional): If True, optimizer parameters will be saved. Default is True.
- `metadata` (dict, optional): Dictionary of values representing metadata associated with the optimization. Default is None.
- `nested_parameter_space` (bool, optional): If True, create nested parameter spaces for parameter of the form x.y. Default is False.
- `surrogate_method_name` (str, optional): Name of the surrogate modeling method. Default is "gpr".
- `surrogate_method_kwargs` (dict, optional): Keyword arguments for the surrogate modeling method. Default is {"anisotropic": False, "optimizer": "sceua"}.
- `surrogate_custom_training` (str, optional): Custom surrogate training function. Default is None.
- `surrogate_custom_training_kwargs` (dict, optional): Keyword arguments for the custom surrogate training function. Default is None.
- `optimizer_name` (str or list of str, optional): Name(s) of the optimizer(s) to use. Default is "nsga2".
- `optimizer_kwargs` (dict or list of dict, optional): Keyword arguments for the optimizer(s). Default is {"mutation_prob": 0.1, "crossover_prob": 0.9}.
- `sensitivity_method_name` (str, optional): Name of the sensitivity analysis method. Default is None.
- `sensitivity_method_kwargs` (dict, optional): Keyword arguments for the sensitivity analysis method. Default is an empty dictionary.
- `optimize_mean_variance` (bool, optional): If True, both mean and variance will be optimized. Default is False.
- `local_random` (numpy.random.Generator, optional): Local random number generator. Default is None.
- `random_seed` (int, optional): Random seed for reproducibility. Default is None.
- `feasibility_method_name` (str, optional): Name of the feasibility check method. Default is None.
- `feasibility_method_kwargs` (dict, optional): Keyword arguments for the feasibility check method. Default is None.
- `termination_conditions` (dict, optional): Termination conditions for the optimization. Default is None.
- `controller` (distwq.MPIController, optional): MPI controller for distributed optimization. Default is None.
- `**kwargs`: Additional keyword arguments.

The `DistOptimizer` class provides various methods for running the
optimization, retrieving the best solutions, saving and loading
optimization data, and managing the distributed optimization process.

Key methods include:
- `initialize_strategy()`: Initializes the optimization strategy.
- `run_epoch()`: Runs a single epoch of the optimization.
- `get_best()`: Retrieves the best solutions found during the optimization.
- `save_evals()`: Saves the evaluation results.
- `save_surrogate_evals()`: Saves the surrogate evaluation results.
- `save_optimizer_params()`: Saves the optimizer parameters.
- `save_stats()`: Saves the optimization statistics.

The `DistOptimizer` class interacts with the distributed task queue
(`distwq`) for parallel evaluation of the objective function across
multiple workers. It also utilizes surrogate modeling techniques to
approximate the objective function and guide the optimization process.

### Additional Functions

dmosopt also provides several utility functions for loading and saving
optimization data, initializing optimization strategies, and
evaluating objective functions. Some of the key functions include:

- `dmosopt.dopt_init()`: Initializes the distributed optimizer.
- `dmosopt.dopt_ctrl()`: Controller function for distributed optimization.
- `dmosopt.dopt_work()`: Worker function for distributed optimization.
- `dmosopt.eval_fun()`: Evaluates the objective function.
- `dmosopt.h5_load_raw()`: Loads raw optimization data from an HDF5 file.
- `dmosopt.h5_load_all()`: Loads all optimization data from an HDF5 file.
- `dmosopt.save_to_h5()`: Saves optimization data to an HDF5 file.
- `dmosopt.save_optimizer_params_to_h5()`: Saves optimizer parameters to an HDF5 file.
- `dmosopt.save_surrogate_evals_to_h5()`: Saves surrogate evaluations to an HDF5 file.
- `dmosopt.save_stats_to_h5()`: Saves optimization statistics to an HDF5 file.


:::
