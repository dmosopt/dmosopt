        
def zdt6(x):
    ''' This is the Zitzler-Deb-Thiele Function - type A
        Bound: XUB = [1,1,...]; XLB = [0,0,...]
        dim = 10
    '''
    f = np.zeros(2)
    f[0] = 1 - anp.exp(-4 * x[:, 0]) * anp.power(anp.sin(6 * anp.pi * x[:, 0]), 6)
    g = 1 + 9.0 * anp.power(anp.sum(x[:, 1:], axis=1) / (self.n_var - 1.0), 0.25)
    f[1] = g * (1 - anp.power(f1 / g, 2))
    
    return f


def obj_fun(pp):
    """ Objective function to be minimized. """
    param_values = np.asarray([pp[k] for k in sorted(pp)])
    res = zdt6(param_values)
    logger.info(f"Iter: \t pp:{pp}, result:{res}")
    return res


def zdt6_pareto():
    n = 100
    x = np.linspace(0.2807753191, 1, n)
    return np.array([x, 1 - np.power(x, 2)]).T


if __name__ == '__main__':

    space = {}
    for i in range(10):
        space['x%d' % (i+1)] = [0.0, 1.0]
    problem_parameters = {}
    objective_names = ['y1', 'y2']
    
    # Create an optimizer
    dmosopt_params = {'opt_id': 'dmosopt_zdt6',
                      'obj_fun_name': 'obj_fun',
                      'obj_fun_module': 'example_dmosopt_zdt6',
                      'problem_parameters': problem_parameters,
                      'space': space,
                      'objective_names': objective_names,
                      'population_size': 200,
                      'initial_maxiter': 10,
                      'optimizer': 'age',
                      'termination_conditions': True,
                      'n_initial': 3,
                      'n_epochs': 2}
    
    best = dmosopt.run(dmosopt_params, verbose=True)
    if best is not None:
        import matplotlib.pyplot as plt
        bestx, besty = best
        x, y = dmosopt.sopt_dict['dmosopt_zdt6'].optimizer_dict[0].get_evals()
        besty_dict = dict(besty)
        
        # plot results
        plt.plot(y[:,0],y[:,1],'b.',label='evaluated points')
        plt.plot(besty_dict['y1'],besty_dict['y2'],'r.',label='MO-ASMO')
    
        y_true = zdt6_pareto()
        plt.plot(y_true[:,0],y_true[:,1],'k-',label='True Pareto')
        plt.legend()
        
        plt.savefig("example_dmosopt_zdt6.svg")

        

