"""
Constrained multi-objective optimisation benchmarks for dmosopt.

Exercises constraint handling and serves as a focused testbed for the
JointFTTransformer surrogate (dmosopt.model_transformer) without
requiring a full simulation run such as the motoneuron example.

Problem inventory
-----------------
CONSTR  Deb 2001:              2 vars | 2 obj | 2 constraints
SRN     Srinivas & Deb 1994:   2 vars | 2 obj | 2 constraints
TNK     Tanaka 1995:           2 vars | 2 obj | 2 constraints
OSY     Osyczka & Kundu 1995:  6 vars | 2 obj | 6 constraints

CONSTR and SRN have analytical Pareto fronts that can be used for
solution-quality assertions. TNK has a disconnected feasible region that
exercises non-trivial constraint topology. OSY, with its 6 parameters and
6 constraints, is the primary target for transformer surrogate testing
because it gives the attention mechanism meaningful structure to exploit.

Constraint sign convention
--------------------------
c_i > 0   <=>  constraint i is satisfied (positive feasibility margin)
c_i <= 0  <=>  constraint i is violated

This matches the convention used by MOASMO.py and
model_transformer.joint().
"""

import sys
import logging
import numpy as np
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json

from dmosopt import dmosopt
from dmosopt.hv_adaptive import AdaptiveHyperVolume

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Problem definitions
# Each problem exposes:
#   - <name>_objectives(x) -> np.ndarray   # values to minimise
#   - <name>_constraints(x) -> np.ndarray  # positive = feasible
#   - <name>_space()        -> dict         # dmosopt parameter space
#   - <name>_pareto(n)      -> np.ndarray   # (n, n_obj) analytical front, if known
# =============================================================================

# ---------------------------------------------------------------------------
# CONSTR  (Deb 2001)
# ---------------------------------------------------------------------------


def constr_objectives(x: np.ndarray) -> np.ndarray:
    """f1 = x1,  f2 = (1 + x2) / x1"""
    return np.array([x[0], (1.0 + x[1]) / x[0]], dtype=np.float32)


def constr_constraints(x: np.ndarray) -> np.ndarray:
    """
    g1: x2 + 9*x1 >= 6      c1 = x2 + 9*x1 - 6
    g2: -x2 + 9*x1 >= 1     c2 = -x2 + 9*x1 - 1
    """
    c1 = x[1] + 9.0 * x[0] - 6.0
    c2 = -x[1] + 9.0 * x[0] - 1.0
    return np.array([c1, c2], dtype=np.float32)


def constr_space() -> dict:
    return {"x1": [0.1, 1.0], "x2": [0.0, 5.0]}


def constr_pareto(n_points: int = 400) -> np.ndarray:
    """
    Analytical Pareto front for CONSTR, comprising two segments:
      Segment 1 (g1 active): x1 in [7/18, 2/3], x2 = 6 - 9*x1
                              f2 = 7/x1 - 9
      Segment 2 (x2 = 0):    x1 in [2/3, 1]
                              f2 = 1/x1
    """
    half = n_points // 2
    # Segment 1
    x1_s1 = np.linspace(7.0 / 18.0, 2.0 / 3.0, half)
    f1_s1 = x1_s1
    f2_s1 = 7.0 / x1_s1 - 9.0
    # Segment 2
    x1_s2 = np.linspace(2.0 / 3.0, 1.0, n_points - half)
    f1_s2 = x1_s2
    f2_s2 = 1.0 / x1_s2
    return np.column_stack(
        [np.concatenate([f1_s1, f1_s2]), np.concatenate([f2_s1, f2_s2])]
    )


def constr_obj_fun(pp: dict) -> Tuple[np.ndarray, np.ndarray]:
    x = np.array([pp[k] for k in sorted(pp)], dtype=np.float64)
    obj = constr_objectives(x)
    con = constr_constraints(x)
    logger.info("constr: x=%s  f=%s  c=%s", x, obj, con)
    return obj, con


# ---------------------------------------------------------------------------
# SRN  (Srinivas & Deb 1994)
# ---------------------------------------------------------------------------


def srn_objectives(x: np.ndarray) -> np.ndarray:
    """
    f1 = (x1 − 2)^2 + (x2 − 1)^2 + 2
    f2 = 9*x1 − (x2 − 1)^2
    """
    f1 = (x[0] - 2.0) ** 2 + (x[1] - 1.0) ** 2 + 2.0
    f2 = 9.0 * x[0] - (x[1] - 1.0) ** 2
    return np.array([f1, f2], dtype=np.float32)


def srn_constraints(x: np.ndarray) -> np.ndarray:
    """
    g1: x1^2 + x2^2 <= 225      c1 = 225 - x1^2 - x2^2
    g2: x1 - 3*x2 + 10 <= 0    c2 = 3*x2 - x1 - 10
    """
    c1 = 225.0 - x[0] ** 2 - x[1] ** 2
    c2 = 3.0 * x[1] - x[0] - 10.0
    return np.array([c1, c2], dtype=np.float32)


def srn_space() -> dict:
    return {"x1": [-20.0, 20.0], "x2": [-20.0, 20.0]}


def srn_obj_fun(pp: dict) -> Tuple[np.ndarray, np.ndarray]:
    x = np.array([pp[k] for k in sorted(pp)], dtype=np.float64)
    obj = srn_objectives(x)
    con = srn_constraints(x)
    logger.info("srn: x=%s  f=%s  c=%s", x, obj, con)
    return obj, con


# ---------------------------------------------------------------------------
# TNK  (Tanaka 1995)
# ---------------------------------------------------------------------------


def tnk_objectives(x: np.ndarray) -> np.ndarray:
    """f1 = x1,  f2 = x2"""
    return np.array([x[0], x[1]], dtype=np.float32)


def tnk_constraints(x: np.ndarray) -> np.ndarray:
    """
    g1: x1^2 + x2^2 - 1 - 0.1*cos(16*atan2(x1, x2)) >= 0
        c1 = x1^2 + x2^2 - 1 - 0.1*cos(16*atan2(x1, x2))
    g2: (x1 - 0.5)^2 + (x2 - 0.5)^2 <= 0.5
        c2 = 0.5 - (x1 - 0.5)^2 - (x2 - 0.5)^2
    """
    angle = np.arctan2(x[0], x[1]) if x[1] != 0.0 else np.pi / 2.0
    c1 = x[0] ** 2 + x[1] ** 2 - 1.0 - 0.1 * np.cos(16.0 * angle)
    c2 = 0.5 - (x[0] - 0.5) ** 2 - (x[1] - 0.5) ** 2
    return np.array([c1, c2], dtype=np.float32)


def tnk_space() -> dict:
    eps = 1e-6  # avoid x=0 singularity in arctan2
    return {"x1": [eps, np.pi], "x2": [eps, np.pi]}


def tnk_obj_fun(pp: dict) -> Tuple[np.ndarray, np.ndarray]:
    x = np.array([pp[k] for k in sorted(pp)], dtype=np.float64)
    obj = tnk_objectives(x)
    con = tnk_constraints(x)
    logger.info("tnk: x=%s  f=%s  c=%s", x, obj, con)
    return obj, con


# ---------------------------------------------------------------------------
# OSY  (Osyczka & Kundu 1995)
# Primary transformer target: 6 parameters, 6 constraints.
# ---------------------------------------------------------------------------


def osy_objectives(x: np.ndarray) -> np.ndarray:
    """
    f1 = −[25*(x1-2)^2 + (x2-2)^2 + (x3-1)^2 + (x4-4)^2 + (x5-1)^2 + (x6-4)^2]
         (negated sum-of-squares; both objectives minimised => maximise sum)
    f2 = x1^2 + x2^2
    """
    f1 = -(
        25.0 * (x[0] - 2.0) ** 2
        + (x[1] - 2.0) ** 2
        + (x[2] - 1.0) ** 2
        + (x[3] - 4.0) ** 2
        + (x[4] - 1.0) ** 2
        + (x[5] - 4.0) ** 2
    )
    f2 = x[0] ** 2 + x[1] ** 2
    return np.array([f1, f2], dtype=np.float32)


def osy_constraints(x: np.ndarray) -> np.ndarray:
    """
    g1: x1 + x2 >= 2         →  c1 = x1 + x2 - 2
    g2: x1 + x2 <= 6         →  c2 = 6 - x1 - x2
    g3: x2 - x1 <= 2         →  c3 = 2 - x2 + x1
    g4: x1 - 3*x2 <= 2       →  c4 = 2 - x1 + 3*x2
    g5: (x3-3)^2 + x4 >= 4    →  c5 = (x3-3)^2 + x4 - 4
    g6: (x5-3)^2 + x6 >= 4    →  c6 = (x5-3)^2 + x6 - 4
    """
    c1 = x[0] + x[1] - 2.0
    c2 = 6.0 - x[0] - x[1]
    c3 = 2.0 - x[1] + x[0]
    c4 = 2.0 - x[0] + 3.0 * x[1]
    c5 = (x[2] - 3.0) ** 2 + x[3] - 4.0
    c6 = (x[4] - 3.0) ** 2 + x[5] - 4.0
    return np.array([c1, c2, c3, c4, c5, c6], dtype=np.float32)


def osy_space() -> dict:
    return {
        "x1": [0.0, 10.0],
        "x2": [0.0, 10.0],
        "x3": [1.0, 5.0],
        "x4": [0.0, 6.0],
        "x5": [1.0, 5.0],
        "x6": [0.0, 10.0],
    }


def osy_obj_fun(pp: dict) -> Tuple[np.ndarray, np.ndarray]:
    x = np.array([pp[k] for k in sorted(pp)], dtype=np.float64)
    obj = osy_objectives(x)
    con = osy_constraints(x)
    logger.info("osy: x=%s  f=%s  c=%s", x, obj, con)
    return obj, con


def osy_diagnose_solution_set(
    y_mat: np.ndarray,
    c_mat: Optional[np.ndarray],
    label: str = "",
) -> dict:
    """
    Detect and log pathological conditions in the OSY solution set.

    OSY is the primary transformer benchmark because its first objective
    ``f1 = -[25*(x1-2)^2 + ...]`` is always non-positive, which makes naive
    reference-point formulas such as ``ref = max * 1.1 + epsilon`` produce a point
    that does *not* dominate the least-negative solutions, thus leading to negative
    hyperrectangle volumes, ``W <= 0``, and a NaN/crash in the HV estimator.

    Conditions checked
    ------------------
    1. Empty solution set.
    2. Objective range collapse: ``ptp < 1e-8`` in any dimension causes all
       hyperrectangle volumes to be zero (``W = 0``).
    3. Naive reference-point dominance failure: whether the broken
       ``max * 1.1 + epsilon`` formula would fail to dominate any solution (always
       triggered for OSY because ``f1 ≤ 0``).
    4. Negative hyperrectangle volumes under the safe reference point
       (``max + max(0.1 * range, 1e-6)``).
    5. Non-positive total volume ``W``
    6. All-infeasible best set.
    7. Per-constraint violation rates: identifies which of the six OSY
       constraints the surrogate struggles to satisfy.
    8. Objective value clustering: coefficient of variation (CV) per
       objective; a very low CV indicates solutions are tightly bunched.

    Parameters
    ----------
    y_mat : (n, 2) array of objective values (all best solutions, before
        feasibility filtering).
    c_mat : (n, 6) array of constraint values (positive = satisfied), or None.
    label : optional tag included in log messages (e.g. surrogate name).

    Returns
    -------
    dict with keys:
        ``issues``             - list of human-readable warning strings
        ``obj_ranges``         - peak-to-peak range per objective
        ``obj_cv``             - coefficient of variation per objective
        ``n_negative_volumes`` - count of negative hyperrectangle volumes
        ``W``                  - total volume (should be > 0)
        ``n_not_dominated``    - solutions not dominated by naive ref point
        ``n_feasible``         - count of feasible solutions (if c_mat given)
        ``constraint_viol_rate`` - per-constraint violation rate (if c_mat given)
    """
    prefix = f"OSY [{label}] " if label else "OSY "
    issues: List[str] = []
    stats: dict = {}

    # ------------------------------------------------------------------ #
    # 1. Empty solution set
    # ------------------------------------------------------------------ #
    if y_mat.shape[0] == 0:
        msg = "empty solution set"
        logger.warning("%s%s", prefix, msg)
        return {"issues": [msg]}

    n_sol = y_mat.shape[0]

    # ------------------------------------------------------------------ #
    # 2. Objective range collapse
    # ------------------------------------------------------------------ #
    y_max = np.max(y_mat, axis=0)
    y_min = np.min(y_mat, axis=0)
    y_range = y_max - y_min
    stats["obj_ranges"] = y_range.tolist()

    for i, r in enumerate(y_range):
        if r < 1e-8:
            issues.append(
                f"f{i + 1} range collapsed to {r:.2e}: all {n_sol} best "
                f"solutions share nearly identical values (W -> 0)"
            )

    # ------------------------------------------------------------------ #
    # 3. Naive reference-point dominance failure
    #    (the original bug: max * 1.1 moves negative maxima further negative)
    # ------------------------------------------------------------------ #
    ref_naive = y_max * 1.1 + 1e-6
    not_dominated = ~np.all(ref_naive > y_mat, axis=1)
    n_not_dominated = int(np.sum(not_dominated))
    stats["n_not_dominated"] = n_not_dominated

    if n_not_dominated > 0:
        bad_objs = [
            f"f{i + 1}(max={y_max[i]:.4g}, naive_ref={ref_naive[i]:.4g})"
            for i in range(y_mat.shape[1])
            if y_max[i] < 0
        ]
        issues.append(
            f"{n_not_dominated}/{n_sol} solutions not dominated by naive "
            f"ref_point (max*1.1+epsilon < max for negative objectives: "
            + ", ".join(bad_objs)
            + ")"
        )

    # ------------------------------------------------------------------ #
    # 4. Negative volumes under the safe reference point
    # ------------------------------------------------------------------ #
    ref_safe = y_max + np.maximum(0.1 * y_range, 1e-6)
    diffs = ref_safe - y_mat  # shape (n_sol, n_obj)
    volumes = np.prod(diffs, axis=1)
    n_neg = int(np.sum(volumes < 0))
    stats["n_negative_volumes"] = n_neg

    if n_neg > 0:
        issues.append(
            f"{n_neg}/{n_sol} negative hyperrectangle volumes even under safe "
            f"reference point — reference_point does not dominate all solutions"
        )

    # ------------------------------------------------------------------ #
    # 5. Total volume W
    # ------------------------------------------------------------------ #
    W = float(np.sum(volumes))
    stats["W"] = W

    if W <= 0.0:
        issues.append(
            f"W = {W:.4e} <= 0: hypervolume estimator will crash "
            f"(probabilities = volumes / W produces NaN or negatives)"
        )

    # ------------------------------------------------------------------ #
    # 6. Feasibility
    # ------------------------------------------------------------------ #
    if c_mat is not None:
        feasible_mask = np.all(c_mat > 0, axis=1)
        n_feasible = int(np.sum(feasible_mask))
        stats["n_feasible"] = n_feasible
        stats["feasibility_rate"] = float(n_feasible / n_sol)

        if n_feasible == 0:
            issues.append(
                f"no feasible solutions in best set ({n_sol} solutions, "
                f"all constraint-violating)"
            )

        # ---------------------------------------------------------------- #
        # 7. Per-constraint violation rates
        # ---------------------------------------------------------------- #
        viol_rates = np.mean(c_mat <= 0, axis=0)  # shape (n_con,)
        stats["constraint_viol_rate"] = viol_rates.tolist()
        osy_con_labels = [
            "x1+x2<=2",
            "x1+x2<=6",
            "x2-x1<=2",
            "x1-3x2<=2",
            "(x3-3)^2+x4>=4",
            "(x5-3)^2+x6>=4",
        ]
        for j, (rate, lbl) in enumerate(zip(viol_rates, osy_con_labels)):
            if rate > 0.5:
                issues.append(
                    f"c{j + 1} ({lbl}) violated by {rate * 100:.0f}% of best solutions"
                )

    # ------------------------------------------------------------------ #
    # 8. Objective clustering (coefficient of variation)
    # ------------------------------------------------------------------ #
    y_std = np.std(y_mat, axis=0)
    y_mean_abs = np.abs(np.mean(y_mat, axis=0))
    # avoid division by zero for zero-mean objectives
    cv = np.where(y_mean_abs > 1e-12, y_std / y_mean_abs, np.inf)
    stats["obj_cv"] = cv.tolist()

    for i, c in enumerate(cv):
        if c < 1e-3:
            issues.append(
                f"f{i + 1} CV = {c:.2e}: solutions are tightly clustered "
                f"(std={y_std[i]:.4g}, |mean|={y_mean_abs[i]:.4g}); "
                f"may indicate surrogate convergence to a single point"
            )

    # ------------------------------------------------------------------ #
    # Report
    # ------------------------------------------------------------------ #
    if issues:
        logger.warning("%spathological conditions detected:", prefix)
        for issue in issues:
            logger.warning("%s  - %s", prefix, issue)
    else:
        logger.info("%sno pathological conditions detected", prefix)

    logger.debug(
        "%sstats: ranges=%s  W=%.4e  n_neg_vol=%d  n_not_dom=%d",
        prefix,
        [f"{r:.4g}" for r in y_range],
        W,
        n_neg,
        n_not_dominated,
    )

    stats["issues"] = issues
    return stats


# =============================================================================
# Result dataclass
# =============================================================================


@dataclass
class ConstrainedBenchmarkResult:
    """Diagnostics from a single constrained benchmark run."""

    problem_name: str
    surrogate: str  # "gpr" or "transformer"
    n_variables: int
    n_objectives: int
    n_constraints: int
    n_evaluations: int
    feasibility_rate: float  # fraction of best solutions that are feasible
    final_hv: float
    computation_time_seconds: float
    constraint_violation_mean: float  # mean(max(0, -c)) over best solutions
    extra: Dict = field(default_factory=dict)


# =============================================================================
# Runner
# =============================================================================

_PROBLEM_REGISTRY = {
    "constr": {
        "obj_fun_name": "test_constrained_benchmarks.constr_obj_fun",
        "space": constr_space(),
        "objective_names": ["f1", "f2"],
        "constraint_names": ["c1", "c2"],
        "n_initial": 12,
    },
    "srn": {
        "obj_fun_name": "test_constrained_benchmarks.srn_obj_fun",
        "space": srn_space(),
        "objective_names": ["f1", "f2"],
        "constraint_names": ["c1", "c2"],
        "n_initial": 12,
    },
    "tnk": {
        "obj_fun_name": "test_constrained_benchmarks.tnk_obj_fun",
        "space": tnk_space(),
        "objective_names": ["f1", "f2"],
        "constraint_names": ["c1", "c2"],
        "n_initial": 20,
    },
    "osy": {
        "obj_fun_name": "test_constrained_benchmarks.osy_obj_fun",
        "space": osy_space(),
        "objective_names": ["f1", "f2"],
        "constraint_names": ["c1", "c2", "c3", "c4", "c5", "c6"],
        "n_initial": 20,
    },
}


class ConstrainedBenchmarkRunner:
    """Runs constrained MOO benchmark problems and collects diagnostics."""

    def __init__(self, output_dir: str = "constrained_benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results: List[ConstrainedBenchmarkResult] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        problem_name: str,
        use_transformer: bool = False,
        population_size: int = 100,
        num_generations: int = 100,
        n_epochs: int = 4,
        verbose: bool = False,
        save: bool = False,
    ) -> ConstrainedBenchmarkResult:
        """
        Run one benchmark problem.

        Parameters
        ----------
        problem_name : str
            One of "constr", "srn", "tnk", "osy".
        use_transformer : bool
            When True, uses the JointFTTransformer surrogate via
            ``surrogate_custom_training``.  When False, uses the default
            GPR surrogate.
        population_size, num_generations, n_epochs : int
            Optimisation budget controls.
        verbose : bool
            Forward verbosity flag to dmosopt.run.
        save : bool
            Persist HDF5 checkpoint to output_dir.
        """
        if problem_name not in _PROBLEM_REGISTRY:
            raise ValueError(
                f"Unknown problem '{problem_name}'. "
                f"Choose from: {list(_PROBLEM_REGISTRY)}"
            )

        cfg = _PROBLEM_REGISTRY[problem_name]
        surrogate_label = "transformer" if use_transformer else "gpr"
        opt_id = f"{problem_name}_{surrogate_label}"

        logger.info("=" * 70)
        logger.info(
            "Running %s  surrogate=%s  pop=%d  gen=%d",
            problem_name.upper(),
            surrogate_label,
            population_size,
            num_generations,
        )
        logger.info("=" * 70)

        # Make module-level obj_fun importable by string
        sys.modules["test_constrained_benchmarks"] = sys.modules[__name__]

        dmosopt_params = {
            "opt_id": opt_id,
            "obj_fun_name": cfg["obj_fun_name"],
            "problem_parameters": {},
            "space": cfg["space"],
            "objective_names": cfg["objective_names"],
            "constraint_names": cfg["constraint_names"],
            "optimizer_name": "age",
            "population_size": population_size,
            "num_generations": num_generations,
            "n_initial": cfg["n_initial"],
            "n_epochs": n_epochs,
            "surrogate_method_name": "gpr",
            "surrogate_options": {"lengthscale_bounds": (1e-4, 100.0)},
            "termination_conditions": True,
            "save": save,
            "file_path": str(self.output_dir / f"{opt_id}.h5") if save else None,
            "save_surrogate_evals": False,
        }

        if use_transformer:
            dmosopt_params.update(
                {
                    "surrogate_custom_training": "dmosopt.model_transformer.joint",
                    "surrogate_custom_training_kwargs": {
                        "mode": "c+o",
                        "epochs": "auto",
                        "objectives": True,
                        "constraints": True,
                        "sensitivity": False,
                    },
                }
            )

        start = time.time()
        best = dmosopt.run(dmosopt_params, verbose=verbose, return_constraints=True)
        elapsed = time.time() - start

        result = self._collect_result(
            problem_name=problem_name,
            surrogate_label=surrogate_label,
            cfg=cfg,
            best=best,
            elapsed=elapsed,
        )
        self.results.append(result)
        self._save_result(result)
        self._log_result(result)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_result(
        self,
        problem_name: str,
        surrogate_label: str,
        cfg: dict,
        best,
        elapsed: float,
    ) -> ConstrainedBenchmarkResult:
        n_var = len(cfg["space"])
        n_obj = len(cfg["objective_names"])
        n_con = len(cfg["constraint_names"])

        if best is None:
            return ConstrainedBenchmarkResult(
                problem_name=problem_name,
                surrogate=surrogate_label,
                n_variables=n_var,
                n_objectives=n_obj,
                n_constraints=n_con,
                n_evaluations=0,
                feasibility_rate=0.0,
                final_hv=0.0,
                computation_time_seconds=elapsed,
                constraint_violation_mean=np.nan,
            )

        # best = (prms, lres [, lconstr])
        # Each element is a list of (name, array) tuples; convert to dicts.
        if len(best) == 3:
            bestx, besty, bestc = (
                dict(part) if part is not None else None for part in best
            )
        else:
            bestx, besty = (dict(part) for part in best)
            bestc = None

        obj_names = cfg["objective_names"]
        con_names = cfg["constraint_names"]

        # Collect objective matrix: shape (n_solutions, n_obj)
        y_mat = np.column_stack([np.asarray(besty[name]) for name in obj_names])

        # Collect constraint matrix: shape (n_solutions, n_con)
        feasibility_rate = 1.0
        constraint_violation_mean = 0.0
        c_mat = None
        if bestc is not None:
            c_mat = np.column_stack([np.asarray(bestc[name]) for name in con_names])
            # Feasible rows: all constraint values > 0
            feasible_mask = np.all(c_mat > 0, axis=1)
            feasibility_rate = float(np.mean(feasible_mask))
            violation = np.maximum(0.0, -c_mat)
            constraint_violation_mean = float(np.mean(violation))

        # OSY-specific diagnostics: run on the full (pre-filter) solution set
        # so that degenerate conditions in the surrogate output are visible
        # before feasibility masking discards information.
        extra: dict = {}
        if problem_name == "osy":
            extra["diagnostics"] = osy_diagnose_solution_set(
                y_mat, c_mat, label=surrogate_label
            )

        # Use only feasible solutions for HV
        if c_mat is not None:
            if feasible_mask.any():
                y_mat = y_mat[feasible_mask]

        final_hv = self._hypervolume(y_mat, n_obj)

        opt = dmosopt.dopt_dict[f"{problem_name}_{surrogate_label}"]
        x_all, y_all = opt.optimizer_dict[0].get_evals()
        n_evals = len(y_all)

        return ConstrainedBenchmarkResult(
            problem_name=problem_name,
            surrogate=surrogate_label,
            n_variables=n_var,
            n_objectives=n_obj,
            n_constraints=n_con,
            n_evaluations=n_evals,
            feasibility_rate=feasibility_rate,
            final_hv=final_hv,
            computation_time_seconds=elapsed,
            constraint_violation_mean=constraint_violation_mean,
            extra=extra,
        )

    def _hypervolume(self, y_mat: np.ndarray, n_obj: int) -> float:
        if y_mat.shape[0] == 0:
            return 0.0
        y_max = np.max(y_mat, axis=0)
        y_range = np.ptp(y_mat, axis=0)
        ref_point = y_max + np.maximum(y_range * 0.1, 1e-6)
        hv = AdaptiveHyperVolume(reference_point=ref_point)
        return float(hv.compute_hypervolume(y_mat, algorithm="hybrid"))

    def _log_result(self, r: ConstrainedBenchmarkResult) -> None:
        logger.info(
            "%s [%s]: evals=%d  feasible=%.0f%%  HV=%.4f  "
            "mean_violation=%.4f  time=%.1fs",
            r.problem_name.upper(),
            r.surrogate,
            r.n_evaluations,
            r.feasibility_rate * 100,
            r.final_hv,
            r.constraint_violation_mean,
            r.computation_time_seconds,
        )

    def _save_result(self, r: ConstrainedBenchmarkResult) -> None:
        path = self.output_dir / f"{r.problem_name}_{r.surrogate}_result.json"
        with open(path, "w") as f:
            json.dump(asdict(r), f, indent=2)

    def generate_report(self) -> None:
        if not self.results:
            logger.warning("No results to report.")
            return

        report_path = self.output_dir / "constrained_benchmark_report.txt"
        header = (
            f"{'Problem':<10} {'Surrogate':<14} {'Evals':>6} "
            f"{'Feasible%':>10} {'HV':>10} {'MeanViol':>10} {'Time(s)':>8}"
        )
        sep = "-" * len(header)

        with open(report_path, "w") as f:
            f.write("Constrained Benchmark Report\n")
            f.write("=" * len(header) + "\n")
            f.write(header + "\n")
            f.write(sep + "\n")
            for r in self.results:
                f.write(
                    f"{r.problem_name:<10} {r.surrogate:<14} {r.n_evaluations:>6} "
                    f"{r.feasibility_rate * 100:>9.1f}% {r.final_hv:>10.4f} "
                    f"{r.constraint_violation_mean:>10.4f} "
                    f"{r.computation_time_seconds:>7.1f}s\n"
                )

        logger.info("Report written to %s", report_path)
        with open(report_path) as f:
            print(f.read())


# =============================================================================
# Solution quality helpers (for problems with analytical fronts)
# =============================================================================


def constr_solution_quality(
    objectives: np.ndarray,
    epsilon: float = 0.15,
    n_pareto_samples: int = 1000,
) -> dict:
    """
    Measure proximity of solutions to the analytical CONSTR Pareto front.

    Parameters
    ----------
    objectives : (n, 2) array of [f1, f2] values.
    epsilon : tolerance radius for "on-front" classification.
    """
    pareto = constr_pareto(n_pareto_samples)
    distances = np.array(
        [np.min(np.linalg.norm(pareto - pt, axis=1)) for pt in objectives]
    )
    return {
        "mean_dist_to_front": float(np.mean(distances)),
        "max_dist_to_front": float(np.max(distances)),
        "n_on_front": int(np.sum(distances <= epsilon)),
        "pct_on_front": float(np.mean(distances <= epsilon) * 100),
    }


# =============================================================================
# Pytest entry points
# =============================================================================


def test_constr_gpr(tmp_path):
    """
    CONSTR with the default GPR surrogate.
    Baseline: verifies the constrained optimisation machinery works end-to-end
    and that the best solutions are predominantly feasible.
    """
    runner = ConstrainedBenchmarkRunner(output_dir=str(tmp_path))
    result = runner.run(
        "constr",
        use_transformer=False,
        population_size=60,
        num_generations=50,
        n_epochs=3,
    )

    assert result.feasibility_rate > 0.0, (
        "No feasible solutions found; constraint handling may be broken."
    )
    assert result.final_hv > 0.0, (
        "Hypervolume should be positive for feasible solutions."
    )
    logger.info(
        "CONSTR/GPR: feasibility=%.0f%%  HV=%.4f",
        result.feasibility_rate * 100,
        result.final_hv,
    )


def test_constr_transformer(tmp_path):
    """
    CONSTR with the JointFTTransformer surrogate.
    Verifies that the transformer can be plugged in via
    ``surrogate_custom_training`` and still produces feasible solutions.
    """
    runner = ConstrainedBenchmarkRunner(output_dir=str(tmp_path))
    result = runner.run(
        "constr",
        use_transformer=True,
        population_size=100,
        num_generations=100,
        n_epochs=3,
    )

    assert result.feasibility_rate > 0.0, (
        "Transformer surrogate: no feasible solutions found."
    )
    assert result.final_hv > 0.0
    logger.info(
        "CONSTR/transformer: feasibility=%.0f%%  HV=%.4f",
        result.feasibility_rate * 100,
        result.final_hv,
    )


def test_tnk_transformer(tmp_path):
    """
    TNK with the JointFTTransformer surrogate.
    TNK has a disconnected feasible region, making accurate constraint
    learning particularly important: the surrogate must correctly identify
    the two distinct feasible patches rather than interpolating across the
    infeasible gap.
    """
    runner = ConstrainedBenchmarkRunner(output_dir=str(tmp_path))
    result = runner.run(
        "tnk",
        use_transformer=True,
        population_size=100,
        num_generations=400,
        n_epochs=3,
    )

    assert result.feasibility_rate > 0.0, (
        "TNK/transformer: no feasible solutions found."
    )
    logger.info(
        "TNK/transformer: feasibility=%.0f%%  HV=%.4f",
        result.feasibility_rate * 100,
        result.final_hv,
    )


def test_osy_transformer(tmp_path):
    """
    OSY with the JointFTTransformer surrogate.

    This is the primary transformer benchmark: 6 parameters and 6 constraints
    give the attention mechanism non-trivial structure to learn.  The test
    checks that the transformer surrogate:
      1. Completes without error (integration check).
      2. Discovers at least some feasible solutions.
      3. Achieves a positive hypervolume over the feasible front.

    A full-quality convergence check is intentionally omitted here — OSY
    requires many evaluations to converge and a brief run is used instead.
    """
    runner = ConstrainedBenchmarkRunner(output_dir=str(tmp_path))
    result = runner.run(
        "osy",
        use_transformer=True,
        population_size=100,
        num_generations=400,
        n_epochs=3,
    )

    assert result.feasibility_rate > 0.0, (
        "OSY/transformer: no feasible solutions; surrogate may not be learning "
        "the constraint boundaries correctly."
    )
    assert result.final_hv > 0.0
    assert result.n_evaluations > 0
    logger.info(
        "OSY/transformer: evals=%d  feasibility=%.0f%%  HV=%.4f",
        result.n_evaluations,
        result.feasibility_rate * 100,
        result.final_hv,
    )


# =============================================================================
# Standalone benchmark runner
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run constrained MOO benchmarks (with optional transformer surrogate)"
    )
    parser.add_argument(
        "--problems",
        nargs="+",
        default=["constr", "srn", "tnk", "osy"],
        choices=list(_PROBLEM_REGISTRY),
        help="Problems to run (default: all)",
    )
    parser.add_argument(
        "--surrogate",
        choices=["gpr", "transformer", "both"],
        default="both",
        help="Surrogate to use (default: both)",
    )
    parser.add_argument("--pop", type=int, default=100, help="Population size")
    parser.add_argument("--gen", type=int, default=400, help="Max generations")
    parser.add_argument(
        "--epochs", type=int, default=4, help="Surrogate epochs (n_epochs)"
    )
    parser.add_argument("--output-dir", default="constrained_benchmark_results")
    parser.add_argument("--save", action="store_true", help="Save HDF5 checkpoints")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    runner = ConstrainedBenchmarkRunner(output_dir=args.output_dir)
    use_transformer_flags: List[bool] = []
    if args.surrogate in ("gpr", "both"):
        use_transformer_flags.append(False)
    if args.surrogate in ("transformer", "both"):
        use_transformer_flags.append(True)

    for problem in args.problems:
        for use_transformer in use_transformer_flags:
            try:
                runner.run(
                    problem,
                    use_transformer=use_transformer,
                    population_size=args.pop,
                    num_generations=args.gen,
                    n_epochs=args.epochs,
                    verbose=args.verbose,
                    save=args.save,
                )
            except Exception as exc:
                logger.error(
                    "Failed: %s (transformer=%s): %s", problem, use_transformer, exc
                )

    runner.generate_report()
