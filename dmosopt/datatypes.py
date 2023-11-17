import os, sys, logging, pprint
from functools import partial
from collections import namedtuple
import numpy as np
from enum import IntEnum


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
        return f"Struct({self.__dict__})"

    def __str__(self):
        return f"<Struct>"


class StrategyState(IntEnum):
    EnqueuedRequests = 1
    WaitingRequests = 2
    CompletedEpoch = 3
    CompletedGeneration = 4


ParamSpec = namedtuple(
    "ParamSpec",
    [
        "bound1",
        "bound2",
        "is_integer",
    ],
)


EvalEntry = namedtuple(
    "EvalEntry",
    [
        "epoch",
        "parameters",
        "objectives",
        "features",
        "constraints",
        "prediction",
    ],
)

EvalRequest = namedtuple(
    "EvalRequest",
    [
        "parameters",
        "prediction",
        "epoch",
    ],
)

OptHistory = namedtuple(
    "OptHistory",
    [
        "n_gen",
        "n_eval",
        "x",
        "y",
        "c",
    ],
)

EpochResults = namedtuple(
    "EpochResults",
    [
        "best_x",
        "best_y",
        "gen_index",
        "x",
        "y",
        "optimizer",
    ],
)

GenerationResults = namedtuple(
    "GenerationResults",
    [
        "best_x",
        "best_y",
        "gen_index",
        "x",
        "y",
        "optimizer_params",
    ],
)


class OptProblem(object):
    __slots__ = (
        "dim",
        "lb",
        "ub",
        "int_var",
        "eval_fun",
        "param_names",
        "objective_names",
        "feature_dtypes",
        "constraint_names",
        "n_objectives",
        "n_features",
        "n_constraints",
        "logger",
    )

    def __init__(
        self,
        param_names,
        objective_names,
        feature_dtypes,
        constraint_names,
        spec,
        eval_fun,
        logger=None,
    ):
        self.dim = len(spec.bound1)
        assert self.dim > 0
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
        self.n_constraints = (
            len(constraint_names) if constraint_names is not None else None
        )
        self.logger = logger
