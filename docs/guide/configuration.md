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

If the objective callable does not reside in the `__main__` script, you can provide its full import path, for example, `path.`to_module.my_objective`. Furthermore, you can specify additional arguments to the objective function via `obj_fun_args`. dmosopt will automatically attempt to resolve and import the callable object, and fail with an exception if the specified objective can not be imported.

## Defining the objective

The objective function receives a dictionary with a key for each parameter name and the corresponding value for which the objective ought to be evaluated. In its most basic form, the objective function needs to return an array with the objective values for each of the considered objectives. It is recommended that you provide a name for each objective using `objective_names`. 

The objective function can additionally return corresponding features and constraint values. Use `constraint_names` to specify the list of considered constraints and `feature_names` and `feature_dtypes` to provide the datatypes of the returned feature values (this should be a list of tuples defining the feature names and their dtypes, e.g. `[('feature', float)]`). The following list summarizes the required return values of the objective function:

- `feature_names` and `constraint_names` given: return `values, features, constraints`
- only `feature_names` given: return `values, features`
- only `constraint_names` given: return `values, constraints`
- otherwise: return `values` only

## Initialization options

If you need more fine-grained control over the objective initialization, you can instead provide an `obj_fun_init_name` and optionally `obj_fun_init_args`. If provided, dmosopt will call this function with specified arguments to allow for the dynamic construction of the objective. It will additionally receive the `worker` argument to identify the worker process (or `None` for the controller). Note that the `obj_fun_init_name` function must return a callable objective, but will be ignored if `obj_fun_name` is not `None`.

There is an additional option to hook into dmosopt initialization process via `controller_init_fun_name` (and optionally `controller_init_fun_args`). If provided, the function is getting called by the controlling process at initialization. This can be useful to perform set-up work independent of each objective evaluation performed by the worker processes. 

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

Moreover, you can use `metadata` to pass additional HDF5-serializable data that will be stored in the result file.

[Learn more about how results are stored](./results)


## Optimizer options

You can choose the optimization strategy by specifying `optimizer_name` to point to a Python object. [Several strategies are supported out of the box](./optimizers). To pass optimizer-specific options use `optimizer_kwargs`.

Each optimization relies on a distance metric to sort objective values and select the most promising parameters. To specify the used `distance_metric`, pass `crowding`, `euclidean` or a custom callable (if `None` the crowding distance will be used by default).

To determine the overall number of epochs to sample and test params, you can set the `n_epochs` option.

The optimization will terminate if any of the termination criteria is met, for example, if the maximum number of generations is met, the mean parameter distance is below a given tolerance, etc. By default, dmsopt uses the following termination criteria:

```yaml
# Metric tolerance parameters
x_tol: 1e-6     
f_tol: 0.0001
# Each n-th generation the termination should be checked for
nth_gen: 5
# Maximum number of generations
n_max_gen: num_generations
# Sliding window size, i.e. the last generations that should be considered during the calculations
n_last: 50
```

You can adjust these settings by passing a dictionary to `termination_conditions`, overwriting some or all of these options.

### Sampling strategy

The effectiveness of the optimization will greatly depend on the [sampling strategy](./sampling) that can be configured using the following parameters:

- `space`: Hyperparameters to optimize over. Entries should be of the form: `parameter: (Low_Bound, High_Bound)` (e.g. `{'alpha': (0.65, 0.85), 'gamma': (1, 8)}`). If both bounds for a parameter are Ints, then only integers within the (inclusive) range will be sampled and tested.
- `problem_parameters`: All hyperparameters and their values for the objective function, including those not being optimized over (for example `{'beta': 0.44}`). Can be an empty `dict`. Can include hyperparameters being optimized over, but does not need to. If a hyperparameter is specified in both 'problem_parameters' and 'space', its value in 'problem_parameters' will be overridden.
- `population_size`: Size of the population.
- `num_generations`: Number of generations.
- `resample_fraction`: Percentage of resampled points in each iteration
- `n_initial`: Determines the number of samples and thus the number of evaluations per parameter.
- `initial_method`: Sampling strategy, see [sampling strategies](./sampling) for in-built options. You may pass a dict containing the samples for each parameter name, or a custom callable returning samples.
- `initial_maxiter`: Number of iterations for sampler

dmosopt supports evaluating different problems with the same set of parameters. Use `problem_ids` to specify a set of problem IDs (otherwise, it defaults to `set([0])`). The objective function must return a dictionary of the form `{ problem_id: ... }` for each ID.

Furthermore, it is possible to implement dynamic sampling strategies via the `dynamic_initial_sampling` option (and `dynamic_initial_sampling_kwargs`). [Learn more](./sampling)

### Surrogate strategy

[Surrogate models](./surrogates) can greatly improve sampling effectiveness and convergence. Use `surrogate_method_name` to point to a strategy; method specific options can be passed via `surrogate_method_kwargs`. Moreover, to use a custom training method, you can pass its Python import path to `surrogate_custom_training` (and additional arguments to `surrogate_custom_training_kwargs`).

## Sensitivity

dmosopt supports [sensitivity analysis](https://salib.readthedocs.io/en/latest/user_guide/basics.html) to understand outcome uncertainty concerning the varied inputs. Provide a `sensitivity_method_name` such as 'dgsm' or 'fast', and pass method specific options to `sensitivity_method_kwargs`. [Learn more](https://salib.readthedocs.io/en/latest/index.html).

## Feasibility model

If the optimization is using constraints, dmosopt can construct and fit a model to predict if samples are satisfying the constraints. To use the feasibility model, set `feasibility_model` to `True`; this will construct a Logistic Regression model that will be passed to the optimizer. 
