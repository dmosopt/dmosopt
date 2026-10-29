"""Regression tests for evolutionary optimizers with very small populations.

NSGA-II and AGE-MOEA draw two distinct parents from a tournament mating
pool for every crossover. These tests check that the pool always holds at
least two individuals, that population sizes too small for crossover are
rejected when the optimizer is created, and that a population which holds
fewer individuals than the nominal population size still produces offspring.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from dmosopt.AGEMOEA import AGEMOEA
from dmosopt.MOEA import mating_pool_size
from dmosopt.NSGA2 import NSGA2

N_INPUT = 3
N_OUTPUT = 2
BOUNDS = np.array([[0.0, 1.0]] * N_INPUT)


def make_optimizer(cls, popsize, **kwargs):
    model = SimpleNamespace(feasibility=None)
    return cls(popsize=popsize, nInput=N_INPUT, nOutput=N_OUTPUT, model=model, **kwargs)


def initial_data(n, rng):
    x = rng.random((n, N_INPUT))
    y = np.column_stack([x[:, 0], 1.0 - x[:, 0] + x[:, 1]])
    return x, y


@pytest.mark.parametrize("popsize, expected", [(2, 2), (3, 2), (4, 2), (10, 5)])
def test_mating_pool_size(popsize, expected):
    assert mating_pool_size(popsize) == expected


@pytest.mark.parametrize("cls", [NSGA2, AGEMOEA])
def test_popsize_two_generates_offspring(cls):
    rng = np.random.default_rng(0)
    opt = make_optimizer(cls, popsize=2)
    x, y = initial_data(2, rng)
    opt.initialize_strategy(x, y, BOUNDS, local_random=rng)
    for _ in range(50):
        x_gen, _ = opt.generate()
        assert x_gen.shape[1] == N_INPUT
        assert np.all(x_gen >= BOUNDS[:, 0]) and np.all(x_gen <= BOUNDS[:, 1])


@pytest.mark.parametrize("cls", [NSGA2, AGEMOEA])
@pytest.mark.parametrize("popsize", [0, 1])
def test_popsize_too_small_is_rejected(cls, popsize):
    with pytest.raises(ValueError, match="popsize must be >= 2"):
        make_optimizer(cls, popsize=popsize)


@pytest.mark.parametrize("cls", [NSGA2, AGEMOEA])
def test_min_population_size_too_small_is_rejected(cls):
    with pytest.raises(ValueError, match="min_population_size must be >= 2"):
        make_optimizer(
            cls, popsize=10, adaptive_population_size=True, min_population_size=1
        )


@pytest.mark.parametrize("cls", [NSGA2, AGEMOEA])
@pytest.mark.parametrize("n_individuals", [1, 2, 3])
def test_population_smaller_than_pool(cls, n_individuals):
    rng = np.random.default_rng(1)
    opt = make_optimizer(cls, popsize=10)
    x, y = initial_data(n_individuals, rng)
    opt.initialize_strategy(x, y, BOUNDS, local_random=rng)
    assert opt.state.population_parm.shape[0] < opt.opt_params.poolsize
    for _ in range(20):
        x_gen, _ = opt.generate()
        assert x_gen.shape[1] == N_INPUT
