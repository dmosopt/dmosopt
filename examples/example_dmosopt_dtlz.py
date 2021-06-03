import sys, math, logging
import numpy as np
from dmosopt import dmosopt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# K. Deb, L. Thiele, M.Laumanns,and E. Zitzler.
# Scalable test problems for evolutionary multi-objective optimization.
# In A. Abraham,L. Jain,and R. Goldberg, eds.,
# Evolutionary Multiobjective Optimization, pp. 105-145.
# London: Springer-Verlag, 2005.
         
def g(x, num_objectives):
    """The g function of DTLZ7."""
    g = 0.0
    n = len(x)
    k = n - num_objectives + 1
    for i in range(n - k + 1, n + 1):
        g += x[i-1]
    return 1.0 + 9.0 * g / k


def dtlz7(x, num_objectives=3):
    num_variables = len(x)
    g_value = g(x, num_objectives)
    obj_values = np.asarray([x[i] for i in range(num_objectives)])
    h = 0
    for j in range(1, num_objectives):
        h += obj_values[j-1] / (1.0 + g_value) * (1.0 + math.sin(3.0 * math.pi * obj_values[j-1]))
        h = num_objectives - h
        obj_values[num_objectives - 1] = (1.0 + g_value) * h
    return obj_values
    
         
         
def obj_fun(pp, num_objectives=3):
    """ Objective function to be minimized. """
    param_values = np.asarray([pp[k] for k in sorted(pp)])
    res = dtlz7(param_values, num_objectives=num_objectives)
    logger.info(f"Iter: \t pp:{pp}, result:{res}")
    return res
         
         
         

if __name__ == '__main__':
         
    space = {}
    for i in range(20):
        space['x%d' % (i+1)] = [0.0, 1.0]
        
    problem_parameters = {}
    objective_names = ['y1', 'y2', 'y3']
         
    # Create an optimizer
    dmosopt_params = {'opt_id': 'dmosopt_dtlz7',
                      'obj_fun_name': 'obj_fun',
                      'obj_fun_module': 'example_dmosopt_dtlz',
                      'problem_parameters': problem_parameters,
                      'space': space,
                      'objective_names': objective_names,
                      'population_size': 200,
                      'initial_maxiter': 10,
                      'n_initial': 4,
                      'n_iter': 5, }
         
    best = dmosopt.run(dmosopt_params, verbose=True)
    if best is not None:
        
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        bestx, besty = best
        x, y = dmosopt.sopt_dict['dmosopt_dtlz7'].optimizer_dict[0].get_evals()
        besty_dict = dict(besty)
         
        # plot results
        ax.scatter(y[:,0],y[:,1],y[:,2],c='b',label='evaluated points')
        ax.scatter(besty_dict['y1'],besty_dict['y2'],besty_dict['y3'],c='r',label='MO-ASMO')
             
        plt.savefig("example_dmosopt_dtlz.svg")
         
         
         
