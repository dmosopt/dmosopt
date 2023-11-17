# Setting up the optimization

In dmosopt, optimizations are specified in terms of Python callables. To get started you have to define an objective in your code and point to it in the  configuration dictionary passed to the `dmosopt.run` function. For example:

::: code-group

```python [example.py]
from dmosopt import dmosopt

def my_objective(x):
    # method to optimize; must return features for the given configuration
    ...

dmosopt.run({
    "opt_id": "example_optimization",
    "obj_fun_name": 'my_objective'   # point to callable
    ...
})
```

:::

If the objective callable does not reside in the `__main__` script, you can additionally pass an `obj_fun_module`. Furthermore, you can specify additional arguments to the objective function via `obj_fun_args`. dmosopt will automatically attempt to resolve and import the callable object, and fail with an exception if the specified objective can not be imported.

## Initialization options

If you need more fine-grained control over the objective initialization, you can instead provide an `obj_fun_init_name` and optionally `obj_fun_init_args` and `obj_fun_init_module`. If provided, dmosopt will call this function with specified arguments to allow for dynamic construction of the objective. It will additionally receive the `worker` argument to indify the worker process (or `None` for the controller). Note that the `obj_fun_init_name` function must return a callable objective, but will be ignored if `obj_fun_name` is not `None`.

There is an additional option to hook into dmosopt initialization process via `controller_init_fun_name` (and optionally `controller_init_fun_module` and `controller_init_fun_args`). If provided, the function is getting called by the controlling process at initialization. This can be useful to perform set up work indepedent of each objective evaluation performed by the worker processes. 

Lastly, you can specify an optional `broker_fun_name` (and `broker_module_name`). This callback will receive the `distwq.MPICollectiveBroker` instance which can be useful in advanced use cases.

Remember that dmosopt scripts are using MPI and need to be invoked accordingly:

```bash
$ mpirun -n 4 python example.py
```

dmosopt will allocate a control process as well as workers to evaluate the objective. By default, dmosopt will allocate 1 process per worker; this can be adjusted via the `nprocs_per_workers` option. If `nprocs_per_workers` > 1, a reduce function must be specified via the `reduce_fun_name` option (and optionally `reduce_fun_module`). dmosopt will call the specified function with different worker results to obtain a single reduced result.

### Result file

By default, dmosopt will try to generate an appropriate filename for the results which can be overwritten using the `file_name` option.

Set `save`, `save_eval`, `save_optimizer_params` and `save_surrogate_evals` to `True` to save settings and progress, evaluations, optimization parameters and surrogate evaluations periodically.

Additionally, you can use `metadata` to pass additional HDF5-serializable data that will be stored to the result file.

## Optimizer options

- `optimizer`: One of 'nsga2', 'age', 'smpso', 'cmaes'
- `optimizer_options`: Optimizer specific options
- `objective_names`: List of the objective names
- `space`: Hyperparameters to optimize over. Entries should be of the form:
`parameter: (Low_Bound, High_Bound)` (e.g. `{'alpha': (0.65, 0.85), 'gamma': (1, 8)}`). If both bounds for a parameter are Ints, then only integers within the (inclusive) range will be sampled and tested.
- `problem_parameters`: All hyperparameters and their values for the objective function, including those not being optimized over (for example `{'beta': 0.44}`). Can be an empty `dict`. Can include hyperparameters being optimized over, but does not need to.If a hyperparameter is specified in both 'problem_parameters' and 'space', its value in 'problem_parameters' will be overridden.
- `population_size`: Size of the population.
- `num_generations`: Number of generations.
- `resample_fraction`: Resample fraction.
- `distance_metric`: Distance metric.
- `constraint_names`: List of constraint names
- `feature_dtypes`: List of tuples defining the feature names and their dtypes, e.g. `[('feature', float)]`
- `n_epochs`: Number of epochs to sample and test params
- `problem_ids`: Use to specify a set of problem IDs for solving sets of related problems with the same set of parameters. It defaults to `set([0])` and it is otherwise expected that the objective function will return a dictionary of the form `{ problem_id: value }`.

### Surrogate strategy

- `surrogate_method`: One of 'gpr', 'egp', 'megp', 'mdgp', 'mdspp', 'vgp', 'svgp', 'spv', 'siv', 'crv'
- `surrogate_options`: Method specific options.

### Sensitivity

- `sensitivity_method`: One of 'dgsm', 'fast'
- `sensitivity_options`: Method specific options.

## Sampling strategy

- `n_initial`: Number of evaluations per parameter
- `initial_method`: Sampling strategy. One of 'glp', 'slh', 'lh', 'mc', 'sobol' or dict containing the samples for each parameter names, or a callable returning samples.
- `initial_maxiter`: Number of iterations for sampler

## Other options

- `termination_conditions`: Optional settings to overwrite termination condition defaults (i.e. `{"x_tol": 1e-6,"f_tol": 0.0001,"nth_gen": 5, "n_max_gen": num_generations,"n_last": 50,}`)
- `feasibility_model`: Wether to use a feasibility model
- `local_random`: PRNG
- `random_seed`: Random seed. Do not use if 'local_random' is provided.
- `verbose`: Set to `True` for verbose log output
