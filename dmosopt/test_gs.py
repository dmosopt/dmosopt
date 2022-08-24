import numpy as np
import pandas as pd


def rmtrend(x, y):
    """remove the trend between x and y from y"""
    xm = x - x.mean()
    ym = y - y.mean()
    b = (xm * ym).sum() / (xm**2.0).sum()  # b = (X'X)^(-1)X'y
    z = y - b * xm
    return z


def rand2rank(r):
    """transfer random number in [0,1] to integer number"""
    n = len(r)
    x = np.ndarray(n)
    x[r.argsort()] = np.array(range(n))
    return x


def decorr(x, n, s):
    """Ranked Gram-Schmidt (RGS) de-correlation iteration"""
    # Forward ranked Gram-Schmidt step:
    for j in range(1, s):
        for k in range(j):
            z = rmtrend(x[:, j], x[:, k])
            x[:, k] = (rand2rank(z) + 0.5) / n
    # Backward ranked Gram-Schmidt step:
    for j in range(s - 2, -1, -1):
        for k in range(s - 1, j, -1):
            z = rmtrend(x[:, j], x[:, k])
            x[:, k] = (rand2rank(z) + 0.5) / n
    return x


def Cov(X, Y):
    return ((X - X.mean()) * (Y - Y.mean())).mean()


def proj(X, Y, F=Cov):
    return X * (F(X, Y) / F(X, X))


def to_unit(X, F=Cov):
    return X / np.sqrt(F(X.T, X.T))


def gram_schmidt(A, F=Cov):
    B = A.copy()
    n = A.shape[1]
    for i in range(n):
        for j in range(i):
            B[:, i] = B[:, i] - proj(B[:, j], B[:, i], F=F)
        B[:, i] = to_unit(B[:, i], F=F)
    return B


if __name__ == "__main__":
    A = np.array([[1.0, 1.0, 0.0], [1.0, 3.0, 1.0], [2.0, -1.0, 1.0]])
    print(A)
    df = pd.DataFrame(A)
    print(df.corr())
    d1 = gram_schmidt(A)
    print(d1)
    print(pd.DataFrame(d1).corr())
    d2 = decorr(A, A.shape[0], A.shape[1])
    print(d2)
    print(pd.DataFrame(d2).corr())
