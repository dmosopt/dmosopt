#
# An implementation of the Dominance Degree Matrix ranking algorithm:
#
# Y. Zhou, Z. Chen and J. Zhang,
# "Ranking Vectors by Means of the Dominance Degree Matrix,"
# IEEE Transactions on Evolutionary Computation, vol. 21, no. 1, pp. 34-51, Feb. 2017
# doi: 10.1109/TEVC.2016.2567648.
#

import numpy as np


def comparison_matrix(y, output=None):
    """Construct comparison matrix for input vector y
    y: input vector (N,)
    output: optional output matrix argument of dimension (N, N)
    """
    (n,) = y.shape
    si = np.argsort(y)
    y_sorted = y[si]

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


def dominance_degree_matrix(Y):
    n, d = Y.shape

    D = np.zeros((n, n), dtype=np.intp)
    Cy = np.zeros((n, n), dtype=np.intp)

    for i in range(d):
        comparison_matrix(Y[:, i], output=Cy)
        D = D + Cy

    return D


def dda_non_dominated_sort(Y, return_dom=False):
    """Rank objectives by Dominance Degree Matrix.
    y: input matrix (N, D)
    """
    n, d = Y.shape

    # 1. Construct the dominance degree matrix of set Y
    D = dominance_degree_matrix(Y)
    DM = None
    if return_dom:
        DM = np.copy(D)

    # 2. For the solutions with identical objective vectors, set the
    # corresponding elements of D to zero
    for i in range(n):
        for j in range(i, n):
            if (D[i, j] == d) and (D[j, i] == d):
                D[i, j] = 0
                D[j, i] = 0

    # 3. Assign the solutions Yi to a number of fronts
    count = 0
    k = 0  # the first front
    rank = np.zeros((n,), dtype=np.intp)
    while True:
        Q = []
        maxD = np.max(D, axis=0)
        for i in range(n):
            if maxD[i] < d and maxD[i] >= 0:
                # solution Yi belongs to current front
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


def dda_ens(Y, return_dom=False):
    """Rank objectives by Dominance Degree Matrix.
    y: input matrix (N, D)
    """
    n, d = Y.shape

    # 1. Construct the dominance degree matrix of set Y
    D = dominance_degree_matrix(Y)
    DM = None
    if return_dom:
        DM = np.copy(D)

    # 2. For the solutions with identical objective vectors, set the
    # corresponding elements of D to zero
    for i in range(n):
        for j in range(i, n):
            if (D[i, j] == d) and (D[j, i] == d):
                D[i, j] = 0
                D[j, i] = 0

    # 3. Assign the solutions Yi to a number of fronts
    n_fronts = 0  # number of fronts obtained
    fronts = []
    rank = np.zeros((n,), dtype=np.intp)

    y_order = np.argsort(Y[:, 0])
    for s in y_order:
        n_fronts = dda_insert(s, fronts, n_fronts, Y, D, d)

    for i, front in enumerate(fronts):
        for s in front:
            rank[s] = i

    if return_dom:
        return rank, DM
    else:
        return rank


def dda_insert(s, fronts, n_fronts, Y, D, d):
    """Update set of fronts with solution y."""
    is_inserted = False
    for k in range(0, n_fronts):
        is_dominated = False  # solution s is not dominated by fronts[k]
        for s1 in fronts[k]:
            if D[s1][s] == d:
                is_dominated = True  # solutions s is dominated by s1
                break
        if is_dominated is False:  # solution s is not dominated by fronts[k]
            fronts[k].append(s)
            is_inserted = True
            break
    if is_inserted is False:
        n_fronts = n_fronts + 1
        fronts.append([s])
    return n_fronts
