"""Reference implementations of DDA used as ground-truth in correctness tests."""

import numpy as np


# ---------------------------------------------------------------------------
# Reference (original) implementations
# ---------------------------------------------------------------------------


def _ref_comparison_matrix(y, output=None):
    (n,) = y.shape
    si = np.argsort(y)

    if output is None:
        output = np.zeros((n, n), dtype=np.intp)
    else:
        output.fill(0)

    output[si[0], range(n)] = 1
    for i in range(1, n):
        if y[si[i]] == y[si[i - 1]]:
            output[si[i], range(n)] = output[si[i - 1], range(n)]
        else:
            output[si[i], si[range(i, n)]] = 1

    return output


def _ref_dominance_degree_matrix(Y):
    n, d = Y.shape

    D = np.zeros((n, n), dtype=np.intp)
    Cy = np.zeros((n, n), dtype=np.intp)

    for i in range(d):
        _ref_comparison_matrix(Y[:, i], output=Cy)
        D = D + Cy

    return D


def _ref_dda_ns(Y, return_dom=False):
    n, d = Y.shape

    D = _ref_dominance_degree_matrix(Y)
    DM = None
    if return_dom:
        DM = np.copy(D)

    for i in range(n):
        for j in range(i, n):
            if (D[i, j] == d) and (D[j, i] == d):
                D[i, j] = 0
                D[j, i] = 0

    count = 0
    k = 0
    rank = np.zeros((n,), dtype=np.intp)
    while True:
        Q = []
        maxD = np.max(D, axis=0)
        for i in range(n):
            if maxD[i] < d and maxD[i] >= 0:
                Q.append(i)
                count += 1
        for i in Q:
            D[i, :] = -1
            D[:, i] = -1

        rank[np.asarray(Q, dtype=np.intp)] = k
        k += 1
        if count == n:
            break

    if return_dom:
        return rank, DM
    else:
        return rank


def _ref_dda_ens(Y, return_dom=False):
    n, d = Y.shape

    D = _ref_dominance_degree_matrix(Y)
    DM = None
    if return_dom:
        DM = np.copy(D)

    for i in range(n):
        for j in range(i, n):
            if (D[i, j] == d) and (D[j, i] == d):
                D[i, j] = 0
                D[j, i] = 0

    n_fronts = 0
    fronts = []
    rank = np.zeros((n,), dtype=np.intp)

    y_order = np.argsort(Y[:, 0])
    for s in y_order:
        n_fronts = _ref_dda_insert(s, fronts, n_fronts, Y, D, d)

    for i, front in enumerate(fronts):
        for s in front:
            rank[s] = i

    if return_dom:
        return rank, DM
    else:
        return rank


def _ref_dda_insert(s, fronts, n_fronts, Y, D, d):
    is_inserted = False
    for k in range(0, n_fronts):
        is_dominated = False
        for s1 in fronts[k]:
            if D[s1][s] == d:
                is_dominated = True
                break
        if is_dominated is False:
            fronts[k].append(s)
            is_inserted = True
            break
    if is_inserted is False:
        n_fronts = n_fronts + 1
        fronts.append([s])
    return n_fronts


# ---------------------------------------------------------------------------
# Tests the reference implementations on the paper examples
# ---------------------------------------------------------------------------


def test_reference_output():
    y = np.asarray([0.9218, 0.7382, 0.1763, 0.4057, 0.9355, 0.9218])
    C = _ref_comparison_matrix(y)
    assert C.shape == (6, 6)

    Y = np.asarray(
        [
            [0.9501, 0.2311, 0.6068, 0.2311, 0.8913, 0.9501],
            [0.4565, 0.0185, 0.8214, 0.0185, 0.6154, 0.4565],
            [0.9218, 0.7382, 0.1763, 0.4057, 0.9355, 0.9218],
        ]
    ).T
    D = _ref_dominance_degree_matrix(Y)
    assert D.shape == (6, 6)

    Y1 = np.asarray(
        [
            [0.2031, 0.7894, 0.5678, 0.4940, 0.1343, 0.2031],
            [0.4031, 0.8041, 0.4940, 0.4954, 0.4131, 0.4031],
            [0.3946, 0.9640, 0.4947, 0.5494, 0.4113, 0.3946],
        ]
    ).T
    r_ns = _ref_dda_ns(Y1)
    r_ens = _ref_dda_ens(Y1)
    assert r_ns.shape == (6,)
    assert r_ens.shape == (6,)
    np.testing.assert_array_equal(r_ns, r_ens)
