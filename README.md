# dmosopt

Distributed surrogate based multi-objective optimization.

*dmosopt* is a powerful Python library for performing distributed
multi-objective optimization, with a focus on surrogate-based
approaches. It provides a flexible and efficient framework for
optimizing complex objective functions, especially in scenarios where
evaluations are expensive or time-consuming.

## Introduction to the dmosopt Optimization API

### Key Features

- **Distributed Optimization**: `dmosopt` is designed to run
  optimization tasks across multiple workers, allowing for efficient
  parallelization and scalability. It leverages the
  [distwq](https://github.com/iraikov/distwq) library for distributed
  task management.

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


[Documentation](https://dmosopt.github.io/dmosopt/)


## Quick start example

```python
import sys, logging
import numpy as np
from dmosopt import dmosopt
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def zdt1(x):
    ''' This is the Zitzler-Deb-Thiele Function - type A
        Bound: XUB = [1,1,...]; XLB = [0,0,...]
        dim = 30
    '''
    num_variables = len(x)
    f = np.zeros(2)
    f[0] = x[0]
    g = 1. + 9./float(num_variables-1)*np.sum(x[1:])
    h = 1. - np.sqrt(f[0]/g)
    f[1] = g*h
    return f


def obj_fun(pp):
    """ Objective function to be minimized. """
    param_values = np.asarray([pp[k] for k in sorted(pp)])
    res = zdt1(param_values)
    logger.info(f"Iter: \t pp:{pp}, result:{res}")
    return res


def zdt1_pareto(n_points=100):
    f = np.zeros([n_points,2])
    f[:,0] = np.linspace(0,1,n_points)
    f[:,1] = 1.0 - np.sqrt(f[:,0])
    return f

if __name__ == '__main__':
    space = {}
    for i in range(30):
        space['x%d' % (i+1)] = [0.0, 1.0]
    problem_parameters = {}
    objective_names = ['y1', 'y2']
    
    # Create an optimizer
    dmosopt_params = {'opt_id': 'dmosopt_zdt1',
                      'obj_fun_name': 'example_dmosopt_zdt1.obj_fun',
                      'problem_parameters': problem_parameters,
                      'space': space,
                      'objective_names': objective_names,
                      'population_size': 200,
                      'num_generations': 200,
                      'initial_maxiter': 10,
                      'optimizer': 'nsga2',
                      'termination_conditions': True,
                      'n_initial': 3,
                      'n_epochs': 2}
    
    best = dmosopt.run(dmosopt_params, verbose=True)
    if best is not None:
        import matplotlib.pyplot as plt
        bestx, besty = best
        x, y = dmosopt.sopt_dict['dmosopt_zdt1'].optimizer_dict[0].get_evals()
        besty_dict = dict(besty)
        
        # plot results
        plt.plot(y[:,0],y[:,1],'b.',label='evaluated points')
        plt.plot(besty_dict['y1'],besty_dict['y2'],'r.',label='best points')
    
        y_true = zdt1_pareto()
        plt.plot(y_true[:,0],y_true[:,1],'k-',label='True Pareto')
        plt.legend()
        
        plt.savefig("example_dmosopt_zdt1.svg")
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


## Acknowledgements

dmosopt is based on MO-ASMO as described in the following paper:

Gong, W., Q. Duan, J. Li, C. Wang, Z. Di, A. Ye, C. Miao, and Y. Dai
(2016), Multiobjective adaptive surrogate modeling-based optimization
for parameter estimation of large, complex geophysical models, Water
Resour. Res., 52(3), 1984-2008. doi:10.1002/2015WR018230.
