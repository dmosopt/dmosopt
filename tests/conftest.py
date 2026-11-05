import numpy as np
import pytest


@pytest.fixture
def synthetic_data():
    """80 samples, 3 parameters in [0, 1], 2 objectives, 3 constraints

    (X, Y, C, xlb, xub)"""
    rng = np.random.default_rng(0)
    X = rng.uniform(0.0, 1.0, (80, 3)).astype(np.float32)
    Y = np.sum(X**2, axis=1, keepdims=True) * np.ones((1, 2))
    C = X - 0.5  # feasible when > 0
    xlb, xub = np.zeros(3, np.float32), np.ones(3, np.float32)
    return X, Y.astype(np.float32), C.astype(np.float32), xlb, xub
