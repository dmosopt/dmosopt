# Setting up the optimization

To set up an optimization, you can define the objective as a basic Python function. For example:

::: code-group

```python [example.py]
from dmosopt import dmosopt

def my_objective(x):
    # method to optimize; must return features for the given configuration
    ...

dmosopt.run({
    "opt_id": "example_optimization",
    "obj_fun_name": 'my_objective'
    ...  #  additional configuation options
})
```

:::

If the objective function does not reside in the same script, you may additionally pass an `obj_fun_module`. dmosopt will automatically attempt to resolve and import the callable object, and fail with an exception if the specified objective can not be imported.

If you need more fine-grained control over the objective initialization, you can instead provide an `obj_fun_init_name` and optionally `obj_fun_init_args` and `obj_fun_init_module`. If provided, dmosopt will call this function with specified arguments to allow for dynamic construction of the objective. Note that the `obj_fun_init_name` function must return a callable objective, but will be ignored if a `obj_fun_name` has been provided.

There is an additional option to hook into the initialization via `controller_init_fun_name` (and optionally `controller_init_fun_module` and `controller_init_fun_args`). If provided, the function is getting called by the controlling rank at initialization. This can be useful to perform set up work once indepedent of each objective evaluation performed by the working ranks. 

To lauch the optimization, invoke the script using MPI, for example:

```bash
$ mpirun -n 4 python example.py
```

dmosopt will allocate a controller rank as well as workers to evaluate the objective. By default, dmosopt will allocate 1 process per worker; this can be adjusted via the `nprocs_per_workers` option. If `nprocs_per_workers` > 1, a reduce function must be specified via the `reduce_fun_name` option (and optionally `reduce_fun_module`). dmosopt will call the specified function with the process outcome which in turn must return the reduced result for the worker. 

## dopt_params

Coming soon!