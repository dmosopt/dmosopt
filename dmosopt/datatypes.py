import os, sys, logging, pprint
from functools import partial
from collections import namedtuple
import numpy as np  

class Struct(object):
    def __init__(self, **items):
        self.__dict__.update(items)

    def update(self, items):
        self.__dict__.update(items)

    def __call__(self):
        return self.__dict__

    def __getitem__(self, key):
        return self.__dict__[key]

    def __repr__(self):
        return f'Struct({self.__dict__})'

    def __str__(self):
        return f'<Struct>'


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
                        'constraints',
                        'prediction',
                       ])

EvalRequest = namedtuple('EvalRequest',
                         ['parameters',
                          'prediction',
                         ])

OptHistory = namedtuple('OptHistory',
                        ['n_gen',
                         'n_eval',
                         'x',
                         'y',
                         'c',
                        ])

EpochResults = namedtuple('EpochResult'
                          ['best_x',
                           'best_y',
                           'gen_index',
                           'x',
                           'y'])

class OptProblem(object):

    __slots__ = ( 'dim', 'lb', 'ub', 'int_var', 'eval_fun', 'param_names', 'objective_names',
                  'feature_dtypes', 'constraint_names', 'n_objectives', 'n_features',
                  'n_constraints', 'logger')
     
    def __init__(self, param_names, objective_names, feature_dtypes, constraint_names, spec, eval_fun, logger=None):

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
        self.logger = logger
        

class StateMachine:
    
    def __init__(self):
        self.ops = {}
        self.start_state = None
        self.end_states = []
        self.state = None

    def add_state(self, name, op, state_type=-1):
        name = name.upper()
        self.ops[name] = op
        if state_type == 0:
            self.start_state = name
        if state_type == 1:
            self.end_states.append(name)

    def run(self, args):
        if self.start_state is None:
            raise RuntimeError("StateMachine.run: start state is not set")
        if not self.end_states:
            raise  RuntimeError("StateMachine.run: at least one state must be an end_state")
        if self.state[0] in self.end_states:
            return None
        if self.state is None:
            self.state = (self.start_state, None)
        try:
            op = self.ops[self.state[0]]
        except:
            raise RuntimeError("StateMachine.run: error in operation for state {self.state}")
    
        (new_state, data) = op(*args)
        new_state = new_state.upper()
        self.state = (new_state, data)
        return self.state

    def reset(self):
        if self.start_state is None:
            raise RuntimeError("StateMachine.reset: start state is not set")
        self.state = (self.start_state, None)
    

    
