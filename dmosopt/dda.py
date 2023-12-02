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
    """Constructs comparison matrix for input vector y
    y: input vector (N,)
    output: optional output matrix argument of dimension (N, N)
    """
    (n,) = y.shape
    # Sort y in ascending order
    si = np.argsort(y)
    if output is None:
        output = np.zeros((n, n), dtype=np.intp)
    else:
        output.fill(0)
    for j in range(n):
        output[si[0], j] = 1
    for i in range(1, n):
        if y[si[i]] == y[si[i - 1]]:
            for j in range(n):
                output[si[i], j] = output[si[i - 1], j]
        else:
            for j in range(i, n):
                output[si[i], si[j]] = 1

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
