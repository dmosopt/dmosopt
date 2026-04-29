"""Non-dominated (Pareto) ranking with an optional moocore fast path.

Public API
----------
pareto_rank(Y) -> np.ndarray
    Assign each row of Y a 0-based front index (0 = Pareto-optimal).

When the ``moocore`` package is installed the ranking is delegated to its
compiled C library (``moocore.pareto_rank``, which already returns 0-indexed
ranks).  Otherwise the pure-Python/NumPy Dominance Degree Matrix
implementation in ``dmosopt.dda`` is used as a fallback.
"""

import numpy as np

try:
    import moocore as _moocore

    _MOOCORE_AVAILABLE = True
except ImportError:
    _MOOCORE_AVAILABLE = False


def pareto_rank(Y: np.ndarray) -> np.ndarray:
    """Rank rows of Y into Pareto fronts.

    Parameters
    ----------
    Y : array-like, shape (N, D)
        Objective matrix (minimisation assumed).

    Returns
    -------
    rank : np.ndarray, shape (N,), dtype int
        0-based front indices.  rank[i] == 0 means solution i is
        non-dominated; higher values indicate deeper fronts.
    """
    Y = np.asarray(Y, dtype=float)
    if _MOOCORE_AVAILABLE:
        return _moocore.pareto_rank(Y).astype(np.intp)
    from dmosopt.dda import dda_ens

    return dda_ens(Y)
