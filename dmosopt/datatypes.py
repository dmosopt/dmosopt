import os, sys, logging, pprint
from functools import partial
from collections import namedtuple
import numpy as np
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Dict, List, Union, Tuple, Optional


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


@dataclass
class ParameterValue:
    """Defines a single parameter value."""

    value: float
    is_integer: bool = False
    name: Optional[str] = None


@dataclass
class ParameterDefn:
    """Defines the range and type for a parameter."""

    lower: float
    upper: float
    is_integer: bool
    name: Optional[str] = None  # Original parameter name for reconstruction

    def __post_init__(self):
        if self.lower > self.upper:
            self.lower, self.upper = self.upper, self.lower


@dataclass
class ParameterSpace:
    """Handles nested parameter spaces and their conversions."""

    ranges: Dict[str, Union[ParameterDefn, ParameterValue, "ParameterSpace"]] = field(
        default_factory=dict
    )
    _flat_ranges: List[Union[ParameterDefn, ParameterValue]] = field(
        default_factory=list, init=False
    )
    _param_paths: Dict[str, List[str]] = field(default_factory=dict, init=False)

    def __post_init__(self):
        self._flatten_ranges()

    def _flatten_ranges(self, prefix: str = "") -> None:
        """Build flat representation of nested parameter ranges."""
        self._flat_ranges = []
        self._param_paths = {}

        for name, item in self.ranges.items():
            current_path = f"{prefix}.{name}" if prefix else name

            if isinstance(item, (ParameterDefn, ParameterValue)):
                item.name = current_path
                self._flat_ranges.append(item)
                self._param_paths[current_path] = current_path.split(".")
            elif isinstance(item, ParameterSpace):
                item._flatten_ranges(current_path)
                self._flat_ranges.extend(item._flat_ranges)
                self._param_paths.update(item._param_paths)

    @classmethod
    def from_dict(cls, config: Dict, is_value_only: bool = False) -> "ParameterSpace":
        """
        Create a ParameterSpace from a nested dictionary specification.

        Args:
            config: Nested dictionary where leaf nodes are dictionaries containing
                   parameter range specifications
            is_value_only: If True, treat numeric leaves as values rather than ranges


        Example config:
        {
            'soma': {
                'gkabar_kap': [ 0.001, 0.1, False ], # lower bound, upper bound, is_integer
                'gkdrbar_kdr': [0.001, 0.1] # lower bound, upper bound
            },
            'axon': {
                'gbar_nax': [0.01, 0.2] # lower bound, upper bound
            }
        }

        Returns:
            ParameterSpace instance
        """

        def parse_level(
            x: Union[List, float, int, Dict]
        ) -> Union[ParameterDefn, "ParameterSpace"]:
            # Check if this level is a parameter specification
            if isinstance(x, list):
                return ParameterDefn(
                    lower=float(x[0]),
                    upper=float(x[1]),
                    is_integer=x[2] if len(x) > 2 else False,
                )
            elif isinstance(x, (int, float, np.float32)) and is_value_only:
                return ParameterValue(value=float(x), is_integer=isinstance(x, int))
            elif isinstance(x, dict):
                # Create new ranges dictionary with parsed values
                ranges = {key: parse_level(value) for key, value in x.items()}
                # Return new ParameterSpace instance with the parsed ranges
                return cls(ranges=ranges)
            else:
                raise ValueError(f"Unexpected value type: {type(x)}")

        return parse_level(config)

    @property
    def is_value_space(self) -> bool:
        """Returns True if this is a value-only parameter space."""
        return all(isinstance(r, ParameterValue) for r in self._flat_ranges)

    # Other properties remain the same, but add value-specific ones
    @property
    def parameter_values(self) -> np.ndarray:
        """Returns array of parameter values in flattened space."""
        if not self.is_value_space:
            raise ValueError("Not a value-only parameter space")
        return np.asarray([param.value for param in self._flat_ranges])

    @property
    def parameter_names(self) -> List[str]:
        """Returns list of flattened parameter names."""
        return list([param_defn.name for param_defn in self._flat_ranges])

    @property
    def parameter_paths(self) -> Dict[str, List[str]]:
        """Returns dictionay of parameter paths."""
        return dict(self._param_paths)

    @property
    def items(self) -> List[Union[ParameterDefn, ParameterValue]]:
        return self._flat_ranges

    @property
    def n_parameters(self) -> int:
        """Number of parameters in flattened space."""
        return len(self._flat_ranges)

    @property
    def bound1(self) -> np.ndarray:
        """Returns array of lower parameter bounds in flattened space."""
        if self.is_value_space:
            raise ValueError("Cannot get bounds from value-only parameter space")
        return np.asarray([param_spec.lower for param_spec in self._flat_ranges])

    @property
    def bound2(self) -> np.ndarray:
        """Returns array of lower parameter bounds in flattened space."""
        if self.is_value_space:
            raise ValueError("Cannot get bounds from value-only parameter space")
        return np.asarray([param_spec.upper for param_spec in self._flat_ranges])

    @property
    def is_integer(self) -> np.ndarray:
        """Returns array of integer flags in flattened space."""
        return np.asarray([param_spec.is_integer for param_spec in self._flat_ranges])

    def flatten(self, params: Dict) -> np.ndarray:
        """
        Converts nested parameter dictionary to flat array.

        Args:
            params: Nested dictionary of parameter values

        Returns:
            Array of parameter values
        """
        flat_params = np.zeros(self.n_parameters)

        for i, param_range in enumerate(self._flat_ranges):
            # Navigate nested dictionary using parameter path
            current = params
            path = self._param_paths[param_range.name]

            for key in path[:-1]:
                current = current[key]
            value = current[path[-1]]

            flat_params[i] = value

        return flat_params

    def unflatten(self, flat_params: Optional[np.ndarray] = None) -> Dict:
        """
        Converts flat array to nested parameter dictionary.

        Args:
            flat_params: Array of values for each parameter

        Returns:
            Nested dictionary of parameter values in original space
        """
        params = {}

        if (flat_params is None) and not self.is_value_space:
            raise ValueError("Not a value-only parameter space")

        if flat_params is None:
            return self.unflatten(self.parameter_values)

        for i, param_range in enumerate(self._flat_ranges):

            value = flat_params[i]

            # Navigate/create nested dictionary using parameter path
            current = params
            path = self._param_paths[param_range.name]

            for key in path[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]

            current[path[-1]] = value

        return params


class StrategyState(IntEnum):
    EnqueuedRequests = 1
    WaitingRequests = 2
    CompletedEpoch = 3
    CompletedGeneration = 4


EvalEntry = namedtuple(
    "EvalEntry",
    [
        "epoch",
        "parameters",
        "objectives",
        "features",
        "constraints",
        "prediction",
        "time",
    ],
    defaults=[None, None, None, None, None, None, -1.0],
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
        "feature_constructor",
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
        feature_constructor,
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
        self.feature_constructor = feature_constructor
        self.constraint_names = constraint_names
        self.n_objectives = len(objective_names)
        self.n_features = len(feature_dtypes) if feature_dtypes is not None else None
        self.n_constraints = (
            len(constraint_names) if constraint_names is not None else None
        )
        self.logger = logger


def update_nested_dict(base: Dict, update: Dict) -> Dict:
    """
    Recursively update a nested dictionary with another nested dictionary.

    Args:
        base: Base nested dictionary
        update: Nested dictionary with values to update

    Returns:
        Updated nested dictionary
    """
    result = base.copy()
    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively update nested dictionaries
            result[key] = update_nested_dict(result[key], value)
        else:
            # Direct update for values or new keys
            result[key] = value
    return result
