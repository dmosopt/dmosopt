# Setting up the optimization

In dmosopt, optimizations are specified in terms of importable Python objects. To get started you have to define an objective in your code and point to it in the configuration dictionary passed to the `dmosopt.run` function. For example:

::: code-group

```python [example.py]
from dmosopt import dmosopt

def my_objective(x):
    # method to optimize; returns objective values for the given configuration
    ...

dmosopt.run({
    "opt_id": "example_optimization",
    "obj_fun_name": 'my_objective'   # point to importable object
    ...
})
```

:::

If the objective callable does not reside in the `__main__` script, you can provide its full import path, for example `path.to_module.my_objective`. Furthermore, you can specify additional arguments to the objective function via `obj_fun_args`. dmosopt will automatically attempt to resolve and import the callable object, and fail with an exception if the specified objective can not be imported.

## Defining the objective

The objective function receives a dictionary with a key for each parameter name and the corresponding value for which the objective is ought to be evaluated. In its most basic form, the objective function needs to return an array with the objective values for each of the considered objectives. It is recommended that you provide a name of each objective using `objective_names`. 

The objective function can additionally return corresponding features and contraint values. Use `constraint_names` to specify the list of considered constraints and `feature_dtypes` to provide the datatypes of the returned feature values (this should be a list of tuples defining the feature names and their dtypes, e.g. `[('feature', float)]`).

[TODO: explain what dmopt does with the returned contr_values and feature_values]

## Initialization options

If you need more fine-grained control over the objective initialization, you can instead provide an `obj_fun_init_name` and optionally `obj_fun_init_args`. If provided, dmosopt will call this function with specified arguments to allow for dynamic construction of the objective. It will additionally receive the `worker` argument to indify the worker process (or `None` for the controller). Note that the `obj_fun_init_name` function must return a callable objective, but will be ignored if `obj_fun_name` is not `None`.

There is an additional option to hook into dmosopt initialization process via `controller_init_fun_name` (and optionally `controller_init_fun_args`). If provided, the function is getting called by the controlling process at initialization. This can be useful to perform set up work indepedent of each objective evaluation performed by the worker processes. 

Lastly, you can specify an optional `broker_fun_name` (and `broker_module_name`). This callback will receive the `distwq.MPICollectiveBroker` instance which can be useful in advanced use cases.

Remember that dmosopt scripts are using MPI and need to be invoked accordingly, for example:

```bash
$ mpirun -n 4 python example.py
```

dmosopt will allocate a control process as well as workers to evaluate the objective. By default, dmosopt will allocate 1 process per worker; this can be adjusted via the `nprocs_per_workers` option. If `nprocs_per_workers` > 1, a reduce function must be specified via the `reduce_fun_name` option. dmosopt will call the specified function with different worker results to obtain a single reduced result.

Note that you can control the verbosity of the log by toggling the `verbose` option. Furthermore, you can specify a `random_seed` or a PRNG via the `local_random` option. Do not pass these options in conjunction.

### Result file

By default, dmosopt will try to generate an appropriate filename for the results H5 file. This can be overwritten using the `file_name` option.

Set `save`, `save_eval`, `save_optimizer_params` and `save_surrogate_evals` to `True` to periodically save settings and progress, evaluations, optimization parameters and surrogate evaluations respectively.

Moreover, you can use `metadata` to pass additional HDF5-serializable data that will be stored to the result file.

[Learn more about how results are stored](./results)


## Optimizer options

dmosopt implements a number of optimization strategies that can be selected by specifing `optimizer_name`. [A number of strategies are supported out of the box](./optimizers). To pass optimizer-specific options use `optimizer_kwargs`.

Each optimization relies on a distance metric [TODO: ADD DESCRIPTION OF WHAT THIS MEANS]. To specify the used `distance_metric`, pass [TODO: DESCRIBE THE OPTIONS HERE]. 

To determine the overall number of epochs to sample and test params, you can set the `n_epochs` option.

The optimization will terminate [TODO: describe when]. By default, dmsopt uses the following termination criteria:

```yaml
x_tol: 1e-6
f_tol: 0.0001
nth_gen: 5
n_max_gen: num_generations
n_last: 50
```

 You can adjust these settings by passing a dictionary to `termination_conditions`.

### Sampling strategy

The effectiveness of the optimization will greatly depend on the [sampling strategy](./sampling) that can be configured using the following parameters:

- `space`: Hyperparameters to optimize over. Entries should be of the form: `parameter: (Low_Bound, High_Bound)` (e.g. `{'alpha': (0.65, 0.85), 'gamma': (1, 8)}`). If both bounds for a parameter are Ints, then only integers within the (inclusive) range will be sampled and tested.
- `problem_parameters`: All hyperparameters and their values for the objective function, including those not being optimized over (for example `{'beta': 0.44}`). Can be an empty `dict`. Can include hyperparameters being optimized over, but does not need to.If a hyperparameter is specified in both 'problem_parameters' and 'space', its value in 'problem_parameters' will be overridden.
- `population_size`: Size of the population.
- `num_generations`: Number of generations.  [TODO: EXPAND EXPLANATION]
- `resample_fraction`: Resample fraction. [TODO: EXPAND EXPLANATION]
- `n_initial`: Number of evaluations per parameter [TODO: EXPAND EXPLANATION]
- `initial_method`: Sampling strategy. One of 'glp', 'slh', 'lh', 'mc', 'sobol' (see [sampling strategies](./sampling)) or dict containing the samples for each parameter names, or a callable returning samples.
- `initial_maxiter`: Number of iterations for sampler

dmosopt supports evaluation different problems with the same set of parameters. Use `problem_ids` to specify a set of problem IDs (otherwise  it defaults to `set([0])`). The objective function must return a dictionary of the form `{ problem_id: value }`.

### Surrogate strategy

[Surrogate models](./surrogates) can greatly improve sampling effectiveness and convergence. Use `surrogate_method_name` to point to a strategy, such as: 'gpr', 'egp', 'megp', 'mdgp', 'mdspp', 'vgp', 'svgp', 'spv', 'siv', 'crv'.  Method specific options can be passed via `surrogate_method_kwargs`.

[TODO: DESCRIBE HERE HOW EPOCHS ARE COUNTED IF A SURROGATE IS PRESENT]

## Sensitivity

[TODO: EXPLAIN WHAT THIS IS DOING]

Select a `sensitivity_method_name` such as 'dgsm' or 'fast', and pass method specific options to `sensitivity_method_kwargs`.

## Feasibility model

[TODO: DESCRIBE WHAT THIS IS DOING AND WHAT THE MODEL IS]

To use the feasibility model, set `feasibility_model` to `True`.
