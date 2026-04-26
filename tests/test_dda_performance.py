"""Correctness and performance tests for dda_ens / dda_non_dominated_sort.

Correctness: the vectorised implementation must return the same rank arrays
as the reference (original) implementation in test_dda.py on random data
and on data with duplicate rows.

When moocore is installed:
  - The moocore fast path result must match moocore.pareto_rank() - 1.
  - The fast path wall-clock time must stay within RATIO_LIMIT of moocore.
"""

import time

import numpy as np
import pytest

from dmosopt.dda import _MOOCORE_AVAILABLE, dda_ens, dda_non_dominated_sort
from tests.test_dda import _ref_dda_ens, _ref_dda_ns

if _MOOCORE_AVAILABLE:
    import moocore as _moocore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RATIO_LIMIT = 20.0
N_REPEATS = 3


def _time(fn, x, n_repeats=N_REPEATS):
    fn(x)  # warmup
    t0 = time.perf_counter()
    for _ in range(n_repeats):
        fn(x)
    return (time.perf_counter() - t0) / n_repeats


def _random_unique(n, d, seed):
    rng = np.random.default_rng(seed)
    x = rng.random((n, d))
    while len(np.unique(x, axis=0)) < len(x):
        x = rng.random((n, d))
    return x


def _same_fronts(rank_a, rank_b):
    """True if a and b assign solutions to the same front structure.

    Both rank arrays are 0-indexed.  Two solutions must be on the same front
    iff they have identical rank values, but the specific integers used may
    differ between implementations (e.g. different tie-breaking can swap the
    label of two equally-sized fronts).  We therefore compare the induced
    partition, not raw integers.
    """

    # Build a canonical label by sorting unique rank values
    def _canonicalise(r):
        unique_sorted = np.unique(r)
        mapping = {v: i for i, v in enumerate(unique_sorted)}
        return np.array([mapping[v] for v in r], dtype=np.intp)

    return np.array_equal(_canonicalise(rank_a), _canonicalise(rank_b))


# ---------------------------------------------------------------------------
# Correctness: parametrised cases
# ---------------------------------------------------------------------------

CASES = [
    pytest.param(2, 50, id="2d-n50"),
    pytest.param(3, 100, id="3d-n100"),
    pytest.param(3, 300, id="3d-n300"),
    pytest.param(5, 100, id="5d-n100"),
    pytest.param(9, 100, id="9d-n100"),
    pytest.param(10, 100, id="10d-n100"),
]


@pytest.mark.parametrize("d,n", CASES)
def test_dda_correctness_vs_reference(d, n):
    """Vectorised dda_ens must agree with the original reference implementation."""
    x = _random_unique(n, d, seed=d * 1000 + n)
    rank_new = dda_ens(x)
    rank_ref = _ref_dda_ens(x)
    assert _same_fronts(rank_new, rank_ref), (
        f"d={d} n={n}: rank mismatch\nnew={rank_new}\nref={rank_ref}"
    )


@pytest.mark.parametrize("d,n", CASES)
def test_dda_non_dominated_sort_correctness_vs_reference(d, n):
    """Vectorised dda_non_dominated_sort must agree with the reference."""
    x = _random_unique(n, d, seed=d * 2000 + n)
    rank_new = dda_non_dominated_sort(x)
    rank_ref = _ref_dda_ns(x)
    assert _same_fronts(rank_new, rank_ref), (
        f"d={d} n={n}: rank mismatch\nnew={rank_new}\nref={rank_ref}"
    )


@pytest.mark.parametrize("d,n", CASES)
def test_dda_correctness_with_duplicates(d, n):
    """Identical objective vectors must be handled correctly."""
    rng = np.random.default_rng(d * 3000 + n)
    x = _random_unique(n, d, seed=d * 3000 + n)
    # Replace ~10% of rows with duplicates of random existing rows
    n_dups = max(1, n // 10)
    dup_src = rng.integers(0, n, size=n_dups)
    dup_dst = rng.integers(0, n, size=n_dups)
    x[dup_dst] = x[dup_src]

    rank_new = dda_ens(x)
    rank_ref = _ref_dda_ens(x)
    assert _same_fronts(rank_new, rank_ref), (
        f"d={d} n={n} with duplicates: rank mismatch"
    )


@pytest.mark.parametrize("d,n", CASES)
def test_dda_return_dom(d, n):
    """dda_ens(return_dom=True) must return (rank, DM) with correct shapes."""
    x = _random_unique(n, d, seed=d * 4000 + n)
    result = dda_ens(x, return_dom=True)
    assert isinstance(result, tuple) and len(result) == 2, (
        "return_dom=True must return a 2-tuple"
    )
    rank, DM = result
    assert rank.shape == (n,)
    assert DM.shape == (n, n)
    # ranks must be non-negative
    assert rank.min() >= 0
    # DM values must be in [0, d]
    assert DM.min() >= 0 and DM.max() <= d


@pytest.mark.skipif(not _MOOCORE_AVAILABLE, reason="moocore not installed")
@pytest.mark.parametrize("d,n", CASES)
def test_dda_correctness_vs_moocore(d, n):
    """When moocore is installed the fast path must match moocore.pareto_rank() - 1."""
    x = _random_unique(n, d, seed=d * 5000 + n)
    rank_dmosopt = dda_ens(x)
    rank_moocore = _moocore.pareto_rank(x).astype(np.intp) - 1
    assert _same_fronts(rank_dmosopt, rank_moocore), (
        f"d={d} n={n}: fast-path result differs from moocore.pareto_rank()"
    )


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------

PERF_CASES = [
    pytest.param(3, 500, id="3d-n500"),
    pytest.param(5, 300, id="5d-n300"),
    pytest.param(9, 200, id="9d-n200"),
]


@pytest.mark.skipif(not _MOOCORE_AVAILABLE, reason="moocore not installed")
@pytest.mark.parametrize("d,n", PERF_CASES)
def test_dda_performance_vs_moocore(d, n):
    """dda_ens (moocore fast path) must stay within RATIO_LIMIT of moocore."""
    x = _random_unique(n, d, seed=d * 6000 + n)

    t_moocore = _time(lambda z: _moocore.pareto_rank(z), x)
    t_dmosopt = _time(lambda z: dda_ens(z), x)

    ratio = t_dmosopt / t_moocore
    print(
        f"\nd={d} n={n}: moocore={t_moocore * 1e3:.3f}ms "
        f"dmosopt={t_dmosopt * 1e3:.3f}ms ratio={ratio:.2f}x"
    )
    assert ratio <= RATIO_LIMIT, (
        f"d={d} n={n}: dda_ens is {ratio:.1f}x slower than moocore "
        f"(limit {RATIO_LIMIT}x)."
    )
