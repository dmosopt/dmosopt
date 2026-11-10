"""Unit tests for AGE-MOEA state initialisation and environmental selection."""

import numpy as np

from dmosopt.AGEMOEA import AGEMOEA, environmental_selection
from dmosopt.MOEA import Struct
from dmosopt.pareto_rank import pareto_rank


def make_optimizer(popsize, n_input, n_output):
    model = Struct(feasibility=None)
    return AGEMOEA(popsize=popsize, nInput=n_input, nOutput=n_output, model=model)


def test_initialize_state_rows_are_aligned():
    """Each state row holds the parameters, objectives, rank and crowding
    distance of the same individual."""
    rng = np.random.default_rng(1)
    n_input, n_output, popsize = 3, 2, 20
    x = rng.random((3 * popsize, n_input))
    y = rng.random((3 * popsize, n_output))
    bounds = np.column_stack([np.zeros(n_input), np.ones(n_input)])

    opt = make_optimizer(popsize, n_input, n_output)
    state = opt.initialize_state(x, y, bounds, local_random=rng)

    assert state.population_parm.shape == (popsize, n_input)
    assert state.population_obj.shape == (popsize, n_output)
    assert state.rank.shape == (popsize,)
    assert state.crowd_dist.shape == (popsize,)

    # Every selected parameter row maps back to its own objective row.
    for xi, yi in zip(state.population_parm, state.population_obj):
        idx = np.flatnonzero(np.all(x == xi, axis=1))
        assert idx.size == 1
        np.testing.assert_array_equal(y[idx[0]], yi)

    # The stored rank agrees with the rank of the full input set.
    full_rank = pareto_rank(y)
    for xi, ri in zip(state.population_parm, state.rank):
        idx = np.flatnonzero(np.all(x == xi, axis=1))[0]
        assert full_rank[idx] == ri


def test_truncated_front_is_scored_by_distance_to_ideal_point():
    """The front that is cut keeps its members closest to the ideal point."""
    y = np.array(
        [
            [0.0, 1.0],
            [0.5, 0.5],
            [1.0, 0.0],  # first front
            [0.2, 3.0],
            [3.0, 0.2],  # second front, far from the ideal point
            [0.6, 0.65],
            [0.65, 0.6],  # second front, close to the ideal point
        ]
    )
    x = np.arange(y.shape[0], dtype=float)[:, None]
    xs, _, rank, crowd_dist = environmental_selection(
        np.random.default_rng(0), x, y, 5, 1, 2
    )
    assert sorted(xs.ravel().tolist()) == [0.0, 1.0, 2.0, 5.0, 6.0]
    assert np.all(crowd_dist[rank == 1] > 0)
