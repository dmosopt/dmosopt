"""Unit tests for the non-dominated sort used to rank and truncate populations."""

import numpy as np

from dmosopt.MOEA import (
    orderMO,
    per_front_crowding_distance_metric,
    remove_worst,
    sortMO,
)
from dmosopt.indicators import crowding_distance_metric
from dmosopt.pareto_rank import pareto_rank


def reference_order(y):
    """Order by rank, then by crowding distance within each front."""
    rank = pareto_rank(y)
    dist = np.zeros(y.shape[0])
    for r in np.unique(rank):
        m = rank == r
        dist[m] = crowding_distance_metric(y[m])
    return np.lexsort((-dist, rank)), dist


def test_crowding_is_computed_within_each_front():
    rng = np.random.default_rng(0)
    y = rng.random((40, 2))
    rank = pareto_rank(y)
    dist = per_front_crowding_distance_metric(y, rank)
    for r in np.unique(rank):
        m = rank == r
        np.testing.assert_allclose(dist[m], crowding_distance_metric(y[m]))


def test_order_matches_per_front_reference():
    rng = np.random.default_rng(1)
    for n_obj in (2, 3):
        y = rng.random((40, n_obj))
        x = np.arange(40, dtype=float)[:, None]
        perm, rank, (dist,) = orderMO(x, y, y_distance_metrics=["crowding"])
        ref_perm, ref_dist = reference_order(y)
        np.testing.assert_array_equal(perm, ref_perm)
        np.testing.assert_allclose(dist, ref_dist[perm])
        np.testing.assert_array_equal(rank, pareto_rank(y)[perm])


def test_points_on_other_fronts_do_not_change_crowding():
    """Moving a dominated point must not change the crowding order of the
    first front."""
    front = np.array([[0.0, 1.0], [0.2, 0.6], [0.5, 0.5], [0.6, 0.2], [1.0, 0.0]])
    dominated_a = np.array([[0.25, 0.65]])
    dominated_b = np.array([[0.9, 0.9]])
    x = np.arange(6, dtype=float)[:, None]
    _, _, rank_a, (dist_a,), perm_a = sortMO(
        x,
        np.vstack([front, dominated_a]),
        y_distance_metrics=["crowding"],
        return_perm=True,
    )
    _, _, rank_b, (dist_b,), perm_b = sortMO(
        x,
        np.vstack([front, dominated_b]),
        y_distance_metrics=["crowding"],
        return_perm=True,
    )
    np.testing.assert_array_equal(perm_a[:5], perm_b[:5])
    np.testing.assert_allclose(dist_a[rank_a == 0], dist_b[rank_b == 0])


def test_sortmo_rows_stay_aligned():
    rng = np.random.default_rng(2)
    x = rng.random((30, 3))
    y = np.column_stack([x[:, 0], 1.0 - x[:, 0] + x[:, 1]])
    xs, ys, rank, _, perm = sortMO(
        x, y, y_distance_metrics=["crowding"], return_perm=True
    )
    np.testing.assert_array_equal(xs, x[perm])
    np.testing.assert_array_equal(ys, y[perm])
    np.testing.assert_allclose(ys[:, 0], xs[:, 0])


def test_remove_worst_keeps_most_crowded_free_members_of_cut_front():
    rng = np.random.default_rng(3)
    y = rng.random((40, 2))
    x = np.arange(40, dtype=float)[:, None]
    xs, _, _ = remove_worst(x, y, 20, y_distance_metrics=["crowding"])
    ref_perm, _ = reference_order(y)
    assert set(xs.ravel().astype(int)) == set(ref_perm[:20])
