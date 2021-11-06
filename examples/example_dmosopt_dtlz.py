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


def dtlz7(x, num_objectives=2):
    num_variables = len(x)
    g_value = g(x, num_objectives)
    obj_values = np.asarray([x[i] for i in range(num_objectives)])
    h = 0
    for j in range(1, num_objectives):
        h += obj_values[j-1] / (1.0 + g_value) * (1.0 + math.sin(3.0 * math.pi * obj_values[j-1]))
        h = num_objectives - h
        obj_values[num_objectives - 1] = (1.0 + g_value) * h
    return obj_values
    

def dtlz7_pareto(n_points=100):
    
    regions = [[0, 0.2514118360],
               [0.6316265307, 0.8594008566],
               [1.3596178367, 1.5148392681],
               [2.0518383519, 2.116426807]]

    pf = []

    for r in regions:
        x1 = np.linspace(r[0], r[1], int(n_points / len(regions)))
        x2 = 4 - x1*(1 + np.sin(3 * np.pi * x1))
        pf.append(np.array([x1, x2]).T)

    pf = np.row_stack(pf)

    return pf
         
         
def obj_fun(pp, num_objectives=2):
    """ Objective function to be minimized. """
    param_values = np.asarray([pp[k] for k in sorted(pp)])
    res = dtlz7(param_values, num_objectives=num_objectives)
    logger.info(f"Iter: \t pp:{pp}, result:{res}")
    return res
         
         
         

if __name__ == '__main__':
         
    space = {}
    for i in range(10):
        space['x%d' % (i+1)] = [0.0, 1.0]
        
    problem_parameters = {}
    objective_names = ['y1', 'y2']
         
    # Create an optimizer
    dmosopt_params = {'opt_id': 'dmosopt_dtlz7',
                      'obj_fun_name': 'obj_fun',
                      'obj_fun_module': 'example_dmosopt_dtlz',
                      'problem_parameters': problem_parameters,
                      'space': space,
                      'objective_names': objective_names,
                      'population_size': 200,
                      'initial_maxiter': 10,
                      'num_generations': 1000,
                      'termination_conditions': True,
                      'n_initial': 5,
                      'n_epochs': 4, }
         
    best = dmosopt.run(dmosopt_params, verbose=True)
    if best is not None:
        
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111)

        bestx, besty = best
        x, y = dmosopt.sopt_dict['dmosopt_dtlz7'].optimizer_dict[0].get_evals()
        besty_dict = dict(besty)
         
        # plot results
        plt.plot(y[:,0],y[:,1],'b.',label='Evaluated points')
        plt.plot(besty_dict['y1'],besty_dict['y2'],'r.',label='Best solutions')
    
        y_true = dtlz7_pareto()
        plt.plot(y_true[:,0],y_true[:,1],'ko',fillstyle='none',label='True Pareto')
        plt.legend()
             
        plt.savefig("example_dmosopt_dtlz.svg")
         
         
         
