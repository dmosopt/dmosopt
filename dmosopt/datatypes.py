import os, sys, logging, pprint
from functools import partial
from collections import namedtuple
import numpy as np  



ParamSpec = namedtuple('ParamSpec',
                       ['bound1',
                        'bound2',
                        'is_integer',
                       ])


EvalEntry = namedtuple('EvalEntry',
                       ['epoch',
                        'parameters',
                        'objectives',
                        'features',
                        'constraints'
                       ])

OptHistory = namedtuple('OptHistory',
                        ['n_gen',
                         'n_eval',
                         'x',
                         'y',
                         'c',
                        ])


class OptProblem(object):

    __slots__ = ( 'dim', 'lb', 'ub', 'int_var', 'eval_fun', 'param_names', 'objective_names',
                  'feature_dtypes', 'constraint_names', 'n_objectives', 'n_features',
                  'n_constraints' )
     
    def __init__(self, param_names, objective_names, feature_dtypes, constraint_names, spec, eval_fun):

        self.dim = len(spec.bound1)
        assert(self.dim > 0)
        self.lb = spec.bound1
        self.ub = spec.bound2
        self.int_var = spec.is_integer
        self.eval_fun = eval_fun
        self.param_names = param_names
        self.objective_names = objective_names
        self.feature_dtypes = feature_dtypes
        self.constraint_names = constraint_names
        self.n_objectives = len(objective_names)
        self.n_features = len(feature_dtypes) if feature_dtypes is not None else None
        self.n_constraints = len(constraint_names) if constraint_names is not None else None
