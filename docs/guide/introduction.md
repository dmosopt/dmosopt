# Welcome!

*dmosopt* is a powerful Python library for performing distributed
multi-objective optimization, with a focus on surrogate-based
approaches. It provides a flexible and efficient framework for
optimizing complex objective functions, especially in scenarios where
evaluations are expensive or time-consuming.

## Introduction to the dmosopt Optimization API

### Key Features

- **Distributed Optimization**: `dmosopt` is designed to run
  optimization tasks across multiple workers, allowing for efficient
  parallelization and scalability. It leverages the `distwq` library
  for distributed task management.

- **Surrogate Modeling**: The library supports various surrogate
  modeling techniques, such as Gaussian Process Regression (GPR), to
  approximate the objective function. This enables efficient
  exploration of the search space without requiring extensive
  evaluations of the actual objective.

- **Adaptive Sampling**: `dmosopt` implements adaptive sampling
  strategies to intelligently select the next points to evaluate based
  on the surrogate model's predictions and uncertainties. This helps
  balance exploration and exploitation during the optimization
  process.

- **Flexible Objective Functions**: The API allows users to define
  their own objective functions, which can be single- or
  multi-objective. 

- **Termination Criteria**: `dmosopt` provides configurable
  termination criteria, such as a maximum number of iterations or a
  convergence threshold, to control when the optimization process
  should stop.

- **Detailed recording of model parameters and objectives**: the
  output file format is
  [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format),
  supporting high-performance, parallel data access.

### Basic Usage

To use `dmosopt`, you need to define your objective function and the
parameter space to be optimized. Here's a basic example:

```python
import dmosopt

def objective_function(params):
    # Evaluate the objective function for the given parameters
    # Return a scalar value or a dictionary of objectives

parameter_space = {
    'param1': (0, 10),  # Continuous parameters
    'param2': (1, 5),
}

dopt_params = {
    "opt_id": "example_optimization",
    'obj_fun': objective_function,
    'space': parameter_space,
    'num_epochs': 100,  # Number of optimization epochs
    'population_size': 50,  # Size of the population in each epoch
}

best_params, best_objectives = dmosopt.run(dopt_params)
```

In this example:

1. We define the `objective_function` that takes a dictionary of
   parameters and returns the objective value(s).

2. We specify the `parameter_space`, indicating the ranges and types
   of the parameters to optimize.
   
3. We create a dictionary `dopt_params` with the necessary settings,
   including the objective function, parameter space, number of
   epochs, and population size.

4. The `opt_id` serves a unique namespace for the resulting output
   file that captures the best solutions as well as various meta data
   about the optimization progress.
   
5. Finally, we call `dmosopt.run()` with the optimization parameters
   and retrieve the best parameters and corresponding objective
   values.

### Advanced Features

`dmosopt` offers many advanced features and customization options, such as:
- Specifying problem parameters and constraints
- Configuring the surrogate model and optimization algorithms
- Saving and loading optimization progress
- Defining custom initialization and reduction functions
- Controlling the distribution and parallelization settings

By leveraging these features, users can fine-tune the optimization
process to suit their specific needs and problem characteristics.



## Where to go from here

::: info Installation

To get started, run `pip install dmosopt`.

[Learn more](./installation)

:::

::: info [Continue with the Guide](./run)

To dive into available configuration options and features.

:::

::: info [Check out the examples](../examples/zdt1)

Explore examples that demonstrate dmosopt' optimization capabilities.

:::

::: info [Consult the Reference](../reference/index)

Describes available APIs in full detail.

:::
