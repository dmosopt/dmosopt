"""
Comprehensive Multi-Objective Optimization Benchmark Suite

Includes problems with varying characteristics to test adaptive termination:
- DTLZ suite (scalable objectives, different PF shapes)
- WFG suite (deceptive, biased, multi-modal)
- MaF suite (many-objective specific challenges)
"""

import numpy as np
from typing import Optional


# ============================================================================
# DTLZ Suite (Deb, Thiele, Laumanns, Zitzler, 2002)
# ============================================================================
# Scalable to arbitrary number of objectives
# Different Pareto front geometries: linear, concave, convex


def dtlz1(x: np.ndarray, n_obj: int = 3) -> np.ndarray:
    """
    DTLZ1: Linear Pareto front with 11^(k-1) local fronts

    Characteristics:
    - Multi-modal (many local Pareto fronts)
    - Linear true PF: sum(f_i) = 0.5
    - Tests algorithm's ability to find global front
    - Standard: k = n_var - n_obj + 1 = 5

    Challenges for termination:
    - May converge to local front (false convergence)
    - HV can stagnate at sub-optimal front
    - Requires multi-scale detection

    Test config:
    - n_var = n_obj + 4 (e.g., 30 vars for 3 obj, 10 vars for 10 obj)
    - Bounds: [0, 1] for all variables
    """
    n_var = len(x)
    k = n_var - n_obj + 1

    # g function (multi-modal)
    xm = x[-k:]
    g = 100 * (k + np.sum((xm - 0.5) ** 2 - np.cos(20 * np.pi * (xm - 0.5))))

    # Objective functions
    f = np.zeros(n_obj)
    for i in range(n_obj):
        f[i] = 0.5 * (1 + g)
        for j in range(n_obj - i - 1):
            f[i] *= x[j]
        if i > 0:
            f[i] *= 1 - x[n_obj - i - 1]

    return f


def dtlz2(x: np.ndarray, n_obj: int = 3) -> np.ndarray:
    """
    DTLZ2: Spherical (concave) Pareto front

    Characteristics:
    - Concave PF: sum(f_i^2) = 1
    - Uni-modal, relatively easy
    - Standard benchmark for many-objective optimization
    - Standard: k = n_var - n_obj + 1 = 10

    Challenges for termination:
    - Spherical front → high overlap ratio → should trigger MCM2RV
    - Tests hybrid algorithm's geometric pre-screening
    - Good convergence properties

    Test config:
    - n_var = n_obj + 9 (e.g., 30 vars for 3 obj, 19 vars for 10 obj)
    - Bounds: [0, 1] for all variables
    """
    n_var = len(x)
    k = n_var - n_obj + 1

    # g function (uni-modal)
    xm = x[-k:]
    g = np.sum((xm - 0.5) ** 2)

    # Objective functions
    f = np.zeros(n_obj)
    for i in range(n_obj):
        f[i] = 1 + g
        for j in range(n_obj - i - 1):
            f[i] *= np.cos(x[j] * np.pi / 2)
        if i > 0:
            f[i] *= np.sin(x[n_obj - i - 1] * np.pi / 2)

    return f


def dtlz3(x: np.ndarray, n_obj: int = 3) -> np.ndarray:
    """
    DTLZ3: Multi-modal version of DTLZ2

    Characteristics:
    - Concave PF (same as DTLZ2) but highly multi-modal
    - 3^k local Pareto fronts
    - Very difficult to converge to global front

    Challenges for termination:
    - Extreme multi-modality
    - Can get stuck in local fronts
    - HV may appear converged but at wrong front
    - Ultimate test for multi-scale stagnation detection

    Test config:
    - n_var = n_obj + 9
    - Bounds: [0, 1] for all variables
    - Use with caution: very difficult even for state-of-art algorithms
    """
    n_var = len(x)
    k = n_var - n_obj + 1

    # g function (highly multi-modal)
    xm = x[-k:]
    g = 100 * (k + np.sum((xm - 0.5) ** 2 - np.cos(20 * np.pi * (xm - 0.5))))

    # Objective functions (same as DTLZ2 but with different g)
    f = np.zeros(n_obj)
    for i in range(n_obj):
        f[i] = 1 + g
        for j in range(n_obj - i - 1):
            f[i] *= np.cos(x[j] * np.pi / 2)
        if i > 0:
            f[i] *= np.sin(x[n_obj - i - 1] * np.pi / 2)

    return f


def dtlz4(x: np.ndarray, n_obj: int = 3, alpha: float = 100.0) -> np.ndarray:
    """
    DTLZ4: Biased spherical Pareto front

    Characteristics:
    - Same shape as DTLZ2 but with biased density
    - Tests diversity maintenance
    - Solutions cluster near boundaries

    Challenges for termination:
    - Non-uniform density affects HV computation
    - Crowding distance may mislead
    - Per-objective convergence should detect bias

    Test config:
    - n_var = n_obj + 9
    - alpha = 100.0 (controls bias strength)
    - Bounds: [0, 1] for all variables
    """
    n_var = len(x)
    k = n_var - n_obj + 1

    # g function
    xm = x[-k:]
    g = np.sum((xm - 0.5) ** 2)

    # Objective functions with power bias
    f = np.zeros(n_obj)
    for i in range(n_obj):
        f[i] = 1 + g
        for j in range(n_obj - i - 1):
            f[i] *= np.cos(x[j] ** alpha * np.pi / 2)
        if i > 0:
            f[i] *= np.sin(x[n_obj - i - 1] ** alpha * np.pi / 2)

    return f


def dtlz5(x: np.ndarray, n_obj: int = 3) -> np.ndarray:
    """
    DTLZ5: Degenerate Pareto front (curve in M-D space)

    Characteristics:
    - True PF is a curve, not a (M-1)-D manifold
    - Degenerate problem: fewer degrees of freedom than objectives
    - Tests algorithm's behavior on degenerate fronts

    Challenges for termination:
    - Degenerate front has near-zero hypervolume improvement potential
    - May trigger early termination
    - Tests robustness of HV-based criteria

    Test config:
    - n_var = n_obj + 9
    - Bounds: [0, 1] for all variables
    - Expected HV much smaller than DTLZ2
    """
    n_var = len(x)
    k = n_var - n_obj + 1

    # g function
    xm = x[-k:]
    g = np.sum((xm - 0.5) ** 2)

    # Theta values (degenerate mapping)
    theta = np.zeros(n_obj - 1)
    theta[0] = x[0] * np.pi / 2
    for i in range(1, n_obj - 1):
        theta[i] = (1 + 2 * g * x[i]) / (2 * (1 + g)) * np.pi / 2

    # Objective functions
    f = np.zeros(n_obj)
    for i in range(n_obj):
        f[i] = 1 + g
        for j in range(n_obj - i - 1):
            f[i] *= np.cos(theta[j])
        if i > 0:
            f[i] *= np.sin(theta[n_obj - i - 1])

    return f


def dtlz7(x: np.ndarray, n_obj: int = 3) -> np.ndarray:
    """
    DTLZ7: Disconnected Pareto front

    Characteristics:
    - Multiple disconnected regions (2^(M-1) regions)
    - Mixed geometry (flat + curved regions)
    - Highly challenging for diversity maintenance

    Challenges for termination:
    - Disconnected regions may be discovered at different rates
    - Per-objective convergence should detect heterogeneity
    - HV can have sudden jumps when new regions discovered
    - Ultimate test for adaptive window termination

    Test config:
    - n_var = n_obj + 19
    - Bounds: [0, 1] for all variables
    - Expect 4 regions for 3 objectives, 8 for 4 objectives
    """
    n_var = len(x)
    k = n_var - n_obj + 1

    # g function
    xm = x[-k:]
    g = 1 + 9 * np.mean(xm)

    # First M-1 objectives
    f = np.zeros(n_obj)
    f[:-1] = x[: n_obj - 1]

    # Last objective
    h = n_obj - np.sum(f[:-1] / (1 + g) * (1 + np.sin(3 * np.pi * f[:-1])))
    f[-1] = (1 + g) * h

    return f


# ============================================================================
# WFG Suite (Walking Fish Group, Huband et al., 2006)
# ============================================================================
# More complex than DTLZ, with bias, deception, and multi-modality


def wfg_shape_linear(x: np.ndarray, m: int) -> np.ndarray:
    """Linear shape function for WFG."""
    f = np.zeros(m)
    for i in range(m):
        f[i] = 1.0
        for j in range(m - i - 1):
            f[i] *= x[j]
        if i > 0:
            f[i] *= 1 - x[m - i - 1]
    return f


def wfg_shape_convex(x: np.ndarray, m: int) -> np.ndarray:
    """Convex shape function for WFG."""
    f = np.zeros(m)
    for i in range(m):
        f[i] = 1.0
        for j in range(m - i - 1):
            f[i] *= 1 - np.cos(x[j] * np.pi / 2)
        if i > 0:
            f[i] *= 1 - np.sin(x[m - i - 1] * np.pi / 2)
    return f


def wfg1(x: np.ndarray, n_obj: int = 3, k: Optional[int] = None) -> np.ndarray:
    """
    WFG1: Mixed separable problem with bias and flat regions

    Characteristics:
    - Mixed separability
    - Flat regions in Pareto front
    - Biased parameters
    - Polynomial bias function

    Challenges for termination:
    - Flat regions lead to non-uniform improvement rates
    - Different objectives converge at different speeds
    - Strong test for per-objective convergence

    Test config:
    - n_var = k + l where k = n_obj - 1, l = 10
    - Bounds: [0, 2i] for x_i (i = 1..n_var)
    """
    n_var = len(x)
    if k is None:
        k = n_obj - 1
    ll = n_var - k

    # Normalization
    y = x / (2 * np.arange(1, n_var + 1))

    # Transformations
    t1 = np.copy(y)
    t1[k:] = y[k:] ** 0.02

    t2 = np.copy(t1)
    for i in range(k):
        t2[i] = t1[i]
    for i in range(k, n_var):
        t2[i] = 0.35 + 0.65 * t1[i]

    # Generate shape vector
    x_vec = np.zeros(n_obj)
    for i in range(n_obj - 1):
        x_vec[i] = np.max(t2[i * ll : (i + 1) * ll])
    x_vec[-1] = np.mean(t2[-ll:])

    # Shape function (convex mixed)
    f = wfg_shape_convex(x_vec, n_obj) * (1 + np.arange(1, n_obj + 1))

    return f


def wfg4(x: np.ndarray, n_obj: int = 3, k: Optional[int] = None) -> np.ndarray:
    """
    WFG4: Multi-modal problem

    Characteristics:
    - Multi-modal landscape
    - Concave Pareto front
    - Tests global search capability

    Challenges for termination:
    - Multiple local fronts
    - May appear converged at local front
    - Multi-scale stagnation should detect

    Test config:
    - n_var = k + l where k = n_obj - 1, l = 10
    - Bounds: [0, 2i] for x_i
    """
    n_var = len(x)
    if k is None:
        k = n_obj - 1
    ll = n_var - k

    # Normalization
    y = x / (2 * np.arange(1, n_var + 1))

    # Multi-modal transformation
    t1 = np.array(
        [y[i] + 0.35 - 0.15 * np.cos(10 * np.pi * y[i] - 5) for i in range(n_var)]
    )

    # Generate shape vector
    x_vec = np.zeros(n_obj)
    for i in range(n_obj - 1):
        x_vec[i] = np.mean(t1[i * ll : (i + 1) * ll])
    x_vec[-1] = np.mean(t1[-ll:])

    # Concave shape
    f = wfg_shape_convex(x_vec, n_obj) * (1 + np.arange(1, n_obj + 1))

    return f


# ============================================================================
# MaF Suite (Many-objective Test Suite, Cheng et al., 2017)
# ============================================================================
# Specifically designed for many-objective (M >= 4) problems


def maf1(x: np.ndarray, n_obj: int = 5) -> np.ndarray:
    """
    MaF1: Linear Pareto front with complicated PS

    Characteristics:
    - Linear PF (like DTLZ1)
    - Complex decision space structure
    - Scalable to very high dimensions (tested up to M=15)

    Challenges for termination:
    - Linear front → low overlap ratio → should use FPRAS
    - Tests algorithm selection at high dimensions
    - Good for testing d > 20 objective reduction

    Test config:
    - n_var = n_obj + 9
    - Bounds: [0, 1] for all variables
    - Test with n_obj = 10, 15, 20, 30 for dimension scaling
    """
    n_var = len(x)
    n_distance = n_var - n_obj + 1

    # Distance function (complex)
    xm = x[-n_distance:]
    g = np.sum((xm - 0.5) ** 2 - np.cos(20 * np.pi * (xm - 0.5)))

    # Objectives
    f = np.zeros(n_obj)
    for i in range(n_obj):
        f[i] = 1 + g
        for j in range(n_obj - i - 1):
            f[i] *= x[j]
        if i > 0:
            f[i] *= 1 - x[n_obj - i - 1]

    return f


def maf2(x: np.ndarray, n_obj: int = 5) -> np.ndarray:
    """
    MaF2: Concave Pareto front (spherical)

    Characteristics:
    - Concave PF (like DTLZ2)
    - Specifically tuned for many objectives
    - Well-behaved convergence

    Challenges for termination:
    - Spherical front at high dimensions
    - Tests MCM2RV performance
    - Good baseline for d = 5-15

    Test config:
    - n_var = n_obj + 9
    - Bounds: [0, 1] for all variables
    - Use as baseline for comparing with other MaF problems
    """
    n_var = len(x)
    n_distance = n_var - n_obj + 1

    # Distance function
    xm = x[-n_distance:]
    g = np.sum((xm - 0.5) ** 2)

    # Objectives (concave)
    f = np.zeros(n_obj)
    for i in range(n_obj):
        f[i] = 1 + g
        for j in range(n_obj - i - 1):
            f[i] *= np.cos(x[j] * np.pi / 2)
        if i > 0:
            f[i] *= np.sin(x[n_obj - i - 1] * np.pi / 2)

    return f


def maf4(x: np.ndarray, n_obj: int = 5) -> np.ndarray:
    """
    MaF4: Badly-scaled objectives

    Characteristics:
    - Concave PF with vastly different objective scales
    - Objectives have different ranges (10^i scale)
    - Tests normalization and scaling robustness

    Challenges for termination:
    - Badly-scaled objectives affect HV computation
    - Reference point selection critical
    - Per-objective convergence should handle scale differences

    Test config:
    - n_var = n_obj + 9
    - Bounds: [0, 1] for all variables
    - Critical test for reference point adaptation
    """
    n_var = len(x)
    n_distance = n_var - n_obj + 1

    # Distance function
    xm = x[-n_distance:]
    g = np.sum((xm - 0.5) ** 2)

    # Objectives with different scales
    f = np.zeros(n_obj)
    for i in range(n_obj):
        f[i] = 1 + g
        for j in range(n_obj - i - 1):
            f[i] *= np.cos(x[j] * np.pi / 2)
        if i > 0:
            f[i] *= np.sin(x[n_obj - i - 1] * np.pi / 2)
        # Apply scale factor
        f[i] *= 10 ** (i * 2)  # Scales: 1, 100, 10000, ...

    return f


# ============================================================================
# Utility Functions
# ============================================================================


def generate_problem_space(
    problem_name: str, n_obj: int, n_var: Optional[int] = None
) -> dict:
    """
    Generate parameter space for dmosopt based on problem name.

    Parameters
    ----------
    problem_name : str
        One of: 'dtlz1', 'dtlz2', 'dtlz3', 'dtlz4', 'dtlz5', 'dtlz7',
                'wfg1', 'wfg4', 'maf1', 'maf2', 'maf4'
    n_obj : int
        Number of objectives
    n_var : int, optional
        Number of variables (auto-determined if None)

    Returns
    -------
    dict
        Parameter space for dmosopt
    """
    # Determine n_var based on problem default config
    if n_var is None:
        if problem_name.startswith("dtlz"):
            if problem_name in ["dtlz1", "dtlz3"]:
                n_var = n_obj + 4  # k = 5 for DTLZ1/3
            elif problem_name == "dtlz7":
                n_var = n_obj + 19  # k = 20 for DTLZ7
            else:
                n_var = n_obj + 9  # k = 10 for others
        elif problem_name.startswith("wfg"):
            n_var = n_obj - 1 + 10  # k = M-1, l = 10
        elif problem_name.startswith("maf"):
            n_var = n_obj + 9
        else:
            n_var = n_obj + 10  # default

    # Create space
    space = {}

    if problem_name.startswith("wfg"):
        # WFG problems have non-uniform bounds
        for i in range(n_var):
            space[f"x{i + 1}"] = [0.0, 2.0 * (i + 1)]
    else:
        # Most problems use [0, 1]
        for i in range(n_var):
            space[f"x{i + 1}"] = [0.0, 1.0]

    return space


def get_problem_metadata(problem_name: str, n_obj: int) -> dict:
    """
    Get metadata about a problem for testing purposes.

    Returns
    -------
    dict with keys:
        - difficulty: 'easy', 'medium', 'hard', 'very_hard'
        - pf_shape: 'linear', 'concave', 'convex', 'mixed', 'disconnected', 'degenerate'
        - multi_modal: bool
        - expected_overlap_ratio: 'low', 'medium', 'high'
        - tests_features: list of features tested
        - standard_n_obj_range: tuple (min, max)
    """
    metadata = {
        "dtlz1": {
            "difficulty": "medium",
            "pf_shape": "linear",
            "multi_modal": True,
            "expected_overlap_ratio": "low",
            "tests_features": [
                "multi_modality",
                "fpras_algorithm",
                "false_convergence",
            ],
            "standard_n_obj_range": (3, 15),
        },
        "dtlz2": {
            "difficulty": "easy",
            "pf_shape": "concave",
            "multi_modal": False,
            "expected_overlap_ratio": "high",
            "tests_features": [
                "mcm2rv_algorithm",
                "spherical_front",
                "clean_convergence",
            ],
            "standard_n_obj_range": (3, 30),
        },
        "dtlz3": {
            "difficulty": "very_hard",
            "pf_shape": "concave",
            "multi_modal": True,
            "expected_overlap_ratio": "high",
            "tests_features": [
                "extreme_multimodality",
                "multi_scale_stagnation",
                "local_fronts",
            ],
            "standard_n_obj_range": (3, 10),
        },
        "dtlz4": {
            "difficulty": "medium",
            "pf_shape": "concave",
            "multi_modal": False,
            "expected_overlap_ratio": "high",
            "tests_features": [
                "biased_density",
                "per_objective_convergence",
                "diversity",
            ],
            "standard_n_obj_range": (3, 20),
        },
        "dtlz5": {
            "difficulty": "medium",
            "pf_shape": "degenerate",
            "multi_modal": False,
            "expected_overlap_ratio": "very_low",
            "tests_features": [
                "degenerate_front",
                "hv_robustness",
                "early_termination",
            ],
            "standard_n_obj_range": (3, 15),
        },
        "dtlz7": {
            "difficulty": "hard",
            "pf_shape": "disconnected",
            "multi_modal": False,
            "expected_overlap_ratio": "medium",
            "tests_features": [
                "disconnected_regions",
                "adaptive_windows",
                "heterogeneous_convergence",
            ],
            "standard_n_obj_range": (3, 10),
        },
        "wfg1": {
            "difficulty": "hard",
            "pf_shape": "mixed",
            "multi_modal": False,
            "expected_overlap_ratio": "medium",
            "tests_features": [
                "flat_regions",
                "biased_parameters",
                "mixed_separability",
            ],
            "standard_n_obj_range": (3, 15),
        },
        "wfg4": {
            "difficulty": "hard",
            "pf_shape": "concave",
            "multi_modal": True,
            "expected_overlap_ratio": "high",
            "tests_features": ["multi_modality", "global_search", "local_fronts"],
            "standard_n_obj_range": (3, 15),
        },
        "maf1": {
            "difficulty": "medium",
            "pf_shape": "linear",
            "multi_modal": False,
            "expected_overlap_ratio": "low",
            "tests_features": [
                "many_objectives",
                "dimension_reduction",
                "fpras_algorithm",
            ],
            "standard_n_obj_range": (5, 30),
        },
        "maf2": {
            "difficulty": "easy",
            "pf_shape": "concave",
            "multi_modal": False,
            "expected_overlap_ratio": "high",
            "tests_features": ["many_objectives", "mcm2rv_algorithm", "baseline"],
            "standard_n_obj_range": (5, 20),
        },
        "maf4": {
            "difficulty": "medium",
            "pf_shape": "concave",
            "multi_modal": False,
            "expected_overlap_ratio": "high",
            "tests_features": [
                "badly_scaled",
                "reference_point_adaptation",
                "normalization",
            ],
            "standard_n_obj_range": (5, 15),
        },
    }

    return metadata.get(problem_name, {})


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Test each problem
    problems = [
        ("dtlz1", 3),
        ("dtlz2", 3),
        ("dtlz3", 3),
        ("dtlz4", 3),
        ("dtlz5", 3),
        ("dtlz7", 3),
        ("wfg1", 3),
        ("wfg4", 3),
        ("maf1", 5),
        ("maf2", 5),
        ("maf4", 5),
    ]

    print("Testing benchmark problems...")
    print("=" * 80)

    for prob_name, n_obj in problems:
        # Generate space
        space = generate_problem_space(prob_name, n_obj)
        n_var = len(space)

        # Get metadata
        meta = get_problem_metadata(prob_name, n_obj)

        # Test evaluation
        x_test = np.random.rand(n_var)
        if prob_name.startswith("wfg"):
            # Scale for WFG bounds
            x_test = x_test * 2 * np.arange(1, n_var + 1)

        # Evaluate
        func = globals()[prob_name]
        f = func(x_test, n_obj=n_obj)

        print(f"\n{prob_name.upper()} (M={n_obj}, D={n_var})")
        print(f"  Shape: {meta.get('pf_shape', 'unknown')}")
        print(f"  Difficulty: {meta.get('difficulty', 'unknown')}")
        print(f"  Overlap: {meta.get('expected_overlap_ratio', 'unknown')}")
        print(f"  Test objective: f = {f}")
        print(f"  Tests: {', '.join(meta.get('tests_features', []))}")

    print("\n" + "=" * 80)
    print("All problems evaluated successfully!")
