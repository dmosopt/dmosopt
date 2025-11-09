"""
Adaptive Hypervolume-Based Termination Criteria

Implements efficient hypervolume computation optimized for termination detection
in multi-objective optimization. Uses progressive precision, multi-fidelity tracking,
and dimension-dependent algorithm routing for reliable convergence detection.

Key Features:
- Progressive precision: Adapts accuracy to optimization stage
- Multi-fidelity tracking: Maintains HV estimates at multiple precisions
- Dimension-dependent routing: Selects optimal algorithm for problem size
- Multi-stage verification: Reduces false convergence signals
- Functional design: Pure functions with lazy evaluation

Based on:
- Deng & Zhang (2020) adaptive Monte Carlo methods
- Yang et al. (2019) box decomposition algorithms
- While et al. (2012) exact hypervolume computation
"""

import copy
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
import time
from scipy.stats import linregress

from dmosopt.termination import SlidingWindowTermination
from dmosopt.hv import AdaptiveHyperVolume

# ============================================================================
# Data Classes for Results and State
# ============================================================================


@dataclass
class HVResult:
    """Result of hypervolume computation."""

    hypervolume: float
    algorithm_used: str
    n_objectives: int
    n_points: int
    epsilon_target: float
    epsilon_actual: float
    computation_time_ms: float
    method: str  # 'exact' or 'approximate'


@dataclass
class HVEstimate:
    """Single HV estimate with metadata."""

    value: float
    generation: int
    epsilon: float
    computation_time_ms: float
    algorithm: str
    front_hash: int


@dataclass
class MultiFidelityHVState:
    """State for multi-fidelity tracking."""

    hv_coarse: Optional[HVEstimate] = None
    hv_medium: Optional[HVEstimate] = None
    hv_fine: Optional[HVEstimate] = None
    history_coarse: List[float] = field(default_factory=list)
    history_medium: List[float] = field(default_factory=list)
    history_fine: List[float] = field(default_factory=list)


@dataclass
class ConvergenceResult:
    """Result of convergence check."""

    converged: bool
    confidence: float
    primary_reason: str
    supporting_evidence: Dict[str, Any]
    recommendation: str


# ============================================================================
# Progressive Precision Scheduler
# ============================================================================


class ProgressivePrecisionScheduler:
    """
    Manages adaptive error bounds based on optimization progress.

    Adapts precision from coarse (early) to fine (late) optimization stages.
    """

    def __init__(
        self,
        early_threshold: int = 20,
        mid_threshold: int = 50,
        early_epsilon: float = 0.05,
        mid_epsilon: float = 0.02,
        late_epsilon: float = 0.01,
        early_check_freq: int = 1,
        mid_check_freq: int = 5,
        late_check_freq: int = 10,
    ):
        """
        Initialize precision schedule.

        Parameters
        ----------
        early_threshold : int
            Generation to switch from early to mid stage
        mid_threshold : int
            Generation to switch from mid to late stage
        early_epsilon : float
            Approximation error for early stage (5%)
        mid_epsilon : float
            Approximation error for mid stage (2%)
        late_epsilon : float
            Approximation error for late stage (1%)
        early_check_freq : int
            Check HV every N generations in early stage
        mid_check_freq : int
            Check HV every N generations in mid stage
        late_check_freq : int
            Check HV every N generations in late stage
        """
        self.early_threshold = early_threshold
        self.mid_threshold = mid_threshold
        self.early_epsilon = early_epsilon
        self.mid_epsilon = mid_epsilon
        self.late_epsilon = late_epsilon
        self.early_check_freq = early_check_freq
        self.mid_check_freq = mid_check_freq
        self.late_check_freq = late_check_freq

    def get_precision_config(self, generation: int) -> Dict[str, Any]:
        """
        Get precision configuration for current generation.

        Pure function - no side effects.

        Parameters
        ----------
        generation : int
            Current generation number

        Returns
        -------
        dict
            Configuration with epsilon, check_freq, stage, and should_check
        """
        if generation < self.early_threshold:
            return {
                "epsilon": self.early_epsilon,
                "check_freq": self.early_check_freq,
                "stage": "early",
                "should_check": generation % self.early_check_freq == 0,
            }
        elif generation < self.mid_threshold:
            return {
                "epsilon": self.mid_epsilon,
                "check_freq": self.mid_check_freq,
                "stage": "mid",
                "should_check": generation % self.mid_check_freq == 0,
            }
        else:
            return {
                "epsilon": self.late_epsilon,
                "check_freq": self.late_check_freq,
                "stage": "late",
                "should_check": generation % self.late_check_freq == 0,
            }

    def adapt_to_progress(
        self, hv_history: List[float], current_generation: int
    ) -> Dict[str, Any]:
        """
        Adaptively adjust precision based on HV progress.

        If HV changing rapidly, can use coarser precision.
        If HV stagnating, should use finer precision.

        Parameters
        ----------
        hv_history : list of float
            Recent HV values
        current_generation : int
            Current generation

        Returns
        -------
        dict
            Adjusted precision configuration
        """
        base_config = self.get_precision_config(current_generation)

        if len(hv_history) < 5:
            return base_config

        # Compute relative change in recent HV
        recent_hv = np.array(hv_history[-5:])
        relative_changes = np.abs(np.diff(recent_hv)) / (recent_hv[:-1] + 1e-10)
        mean_change = np.mean(relative_changes)

        # If HV changing rapidly (>5%), can use coarser epsilon
        if mean_change > 0.05:
            base_config["epsilon"] = min(base_config["epsilon"] * 1.5, 0.1)
            base_config["adaptive_reason"] = "rapid_change"
        # If HV stagnating (<0.5%), should use finer epsilon
        elif mean_change < 0.005:
            base_config["epsilon"] = max(base_config["epsilon"] * 0.5, 0.005)
            base_config["adaptive_reason"] = "stagnating"

        return base_config


# ============================================================================
# Dimension-Dependent Algorithm Router
# ============================================================================


class HVAlgorithmRouter:
    """
    Routes to appropriate HV algorithm based on problem characteristics.

    Selects optimal algorithm based on dimensionality and front size.
    """

    def __init__(self):
        """Initialize router with algorithm registry."""
        self.algorithm_registry = {
            "box_decomposition": self._compute_box_decomposition,
            "adaptive_mc": self._compute_adaptive_mc,
            "reduced_mc": self._compute_reduced_mc,
        }

    def select_algorithm(
        self, n_objectives: int, n_points: int, epsilon: float
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Select optimal algorithm and configuration.

        Pure function based on problem characteristics.

        Parameters
        ----------
        n_objectives : int
            Number of objectives
        n_points : int
            Number of Pareto points
        epsilon : float
            Target approximation error

        Returns
        -------
        tuple
            (algorithm_name, algorithm_config)
        """
        d = n_objectives

        # Decision tree for algorithm selection
        if d < 10:
            return "box_decomposition", {"method": "exact"}

        elif d <= 20:
            # Use adaptive MC with appropriate sample budget
            n_samples = self._estimate_samples_needed(d, epsilon)
            return "adaptive_mc", {
                "method": "approximate",
                "epsilon": epsilon,
                "delta": 0.25,
                "n_samples": n_samples,
            }

        else:  # d > 20
            # Objective reduction + MC
            target_dims = min(15, d // 2 + 5)
            return "reduced_mc", {
                "method": "approximate",
                "epsilon": epsilon,
                "delta": 0.25,
                "target_dims": target_dims,
                "reduction_method": "correlation",
            }

    def compute_hypervolume(
        self,
        pareto_front: np.ndarray,
        reference_point: np.ndarray,
        epsilon: float = 0.01,
        minimize: bool = True,
        verbose: bool = False,
    ) -> HVResult:
        """
        Compute hypervolume using optimal algorithm.

        Main interface for HV computation.

        Parameters
        ----------
        pareto_front : np.ndarray
            Pareto front points (n_points, n_objectives)
        reference_point : np.ndarray
            Reference point
        epsilon : float
            Target approximation error
        minimize : bool
            If True, minimize objectives
        verbose : bool
            Print algorithm selection info

        Returns
        -------
        HVResult
            Hypervolume result with metadata
        """
        n_points, n_objectives = pareto_front.shape

        # Select algorithm
        algo_name, algo_config = self.select_algorithm(n_objectives, n_points, epsilon)

        if verbose:
            print(f"Selected algorithm: {algo_name}")
            print(f"Configuration: {algo_config}")

        # Route to appropriate implementation
        algorithm_fn = self.algorithm_registry[algo_name]

        # Execute
        start = time.time()
        hv_value = algorithm_fn(pareto_front, reference_point, minimize, algo_config)
        elapsed_ms = (time.time() - start) * 1000

        return HVResult(
            hypervolume=hv_value,
            algorithm_used=algo_name,
            n_objectives=n_objectives,
            n_points=n_points,
            epsilon_target=epsilon,
            epsilon_actual=algo_config.get("epsilon", 0.0),
            computation_time_ms=elapsed_ms,
            method=algo_config["method"],
        )

    def _compute_box_decomposition(
        self, front: np.ndarray, ref: np.ndarray, minimize: bool, config: Dict
    ) -> float:
        """Box decomposition for  d < 10."""
        hv = AdaptiveHyperVolume(ref_point=ref)
        return hv.compute_hypervolume(front, algorithm="box")

    def _compute_adaptive_mc(
        self, front: np.ndarray, ref: np.ndarray, minimize: bool, config: Dict
    ) -> float:
        """Adaptive MC for 10 < d <= 20."""
        hv = AdaptiveHyperVolume(
            ref_point=ref,
            mc_epsilon=config.get("epsilon", 0.02),
            mc_delta=config.get("delta", 0.25),
        )

        return hv.compute_hypervolume(front, algorithm="hybrid")

    def _compute_reduced_mc(
        self, front: np.ndarray, ref: np.ndarray, minimize: bool, config: Dict
    ) -> float:
        """
        Objective reduction + MC for d > 20.

        Reduces objectives first, then computes HV in reduced space.
        """
        # Simple reduction: select highest variance objectives
        target_dims = config["target_dims"]
        variances = np.var(front, axis=0)
        selected_indices = np.argsort(variances)[-target_dims:]
        selected_indices = np.sort(selected_indices)

        reduced_front = front[:, selected_indices]
        reduced_ref = ref[selected_indices]

        hv = AdaptiveHyperVolume(
            ref_point=reduced_ref,
            use_adaptive_mc=True,
            mc_epsilon=config.get("epsilon", 0.05),
            mc_delta=config.get("delta", 0.25),
        )

        return hv.compute_hypervolume(reduced_front, algorithm="hybrid")

    @staticmethod
    def _estimate_samples_needed(n_objectives: int, epsilon: float) -> int:
        """
        Estimate number of MC samples needed for target accuracy.

        Based on Deng & Zhang (2020) complexity analysis.
        """
        # Rough heuristic: O(1/epsilon^2) for MC convergence
        base_samples = int(1.0 / (epsilon**2))

        # Scale with dimensionality
        dim_factor = 1 + (n_objectives - 10) * 0.1

        return int(base_samples * dim_factor)

    @staticmethod
    def _compute_basic_hv(front: np.ndarray, ref: np.ndarray, minimize: bool) -> float:
        """Basic hypervolume computation (dominated volume)."""
        # Simple box decomposition for small fronts
        if len(front) == 0:
            return 0.0

        # For minimization, compute dominated volume
        total_volume = 0.0
        for point in front:
            # Volume of box from point to reference
            if np.all(point <= ref):
                box_volume = np.prod(ref - point)
                total_volume += box_volume

        return total_volume

    @staticmethod
    def _compute_mc_hv(
        front: np.ndarray, ref: np.ndarray, minimize: bool, n_samples: int = 10000
    ) -> float:
        """Monte Carlo hypervolume estimation."""
        if len(front) == 0:
            return 0.0

        hv = AdaptiveHyperVolume(
            ref_point=ref, monte_carlo_samples=n_samples, use_adaptive_mc=False
        )

        hv_estimate = hv.compute_hypervolume("monte_carlo")
        return hv_estimate


# ============================================================================
# Multi-Fidelity HV Tracker
# ============================================================================


class MultiFidelityHVTracker:
    """
    Tracks HV at multiple fidelity levels with intelligent caching.

    Maintains HV estimates at coarse, medium, and fine precisions.
    """

    def __init__(
        self,
        reference_point: np.ndarray,
        coarse_epsilon: float = 0.05,
        medium_epsilon: float = 0.02,
        fine_epsilon: float = 0.01,
        coarse_freq: int = 1,
        medium_freq: int = 5,
        fine_freq: int = 10,
    ):
        """
        Initialize multi-fidelity tracker.

        Parameters
        ----------
        reference_point : np.ndarray
            Reference point for HV
        coarse_epsilon : float
            Error bound for coarse estimates (5%)
        medium_epsilon : float
            Error bound for medium estimates (2%)
        fine_epsilon : float
            Error bound for fine estimates (1%)
        coarse_freq : int
            Compute coarse HV every N generations
        medium_freq : int
            Compute medium HV every N generations
        fine_freq : int
            Compute fine HV every N generations
        """
        self.reference_point = reference_point.copy()
        self.coarse_epsilon = coarse_epsilon
        self.medium_epsilon = medium_epsilon
        self.fine_epsilon = fine_epsilon
        self.coarse_freq = coarse_freq
        self.medium_freq = medium_freq
        self.fine_freq = fine_freq

        # Algorithm router
        self.router = HVAlgorithmRouter()

        # Current state (updated via state transitions)
        self.state = MultiFidelityHVState()

    def should_compute_fidelity(self, generation: int, fidelity: str) -> bool:
        """
        Determine if fidelity level should be computed this generation.

        Pure function.

        Parameters
        ----------
        generation : int
            Current generation
        fidelity : str
            'coarse', 'medium', or 'fine'

        Returns
        -------
        bool
            True if should compute
        """
        freq_map = {
            "coarse": self.coarse_freq,
            "medium": self.medium_freq,
            "fine": self.fine_freq,
        }
        return generation % freq_map[fidelity] == 0

    def compute_and_update(
        self,
        pareto_front: np.ndarray,
        generation: int,
        force_fidelities: Optional[List[str]] = None,
        minimize: bool = True,
        verbose: bool = False,
    ) -> MultiFidelityHVState:
        """
        Compute HV at appropriate fidelity levels and update state.

        Returns new state.

        Parameters
        ----------
        pareto_front : np.ndarray
            Current Pareto front
        generation : int
            Current generation number
        force_fidelities : list of str, optional
            Force computation of specific fidelities
        minimize : bool
            If True, minimize objectives
        verbose : bool
            Print computation details

        Returns
        -------
        MultiFidelityHVState
            New state with updated estimates
        """
        new_state = copy.deepcopy(self.state)

        # Compute front hash for cache validation
        front_hash = hash(pareto_front.tobytes())

        # Determine which fidelities to compute
        if force_fidelities is None:
            compute_coarse = self.should_compute_fidelity(generation, "coarse")
            compute_medium = self.should_compute_fidelity(generation, "medium")
            compute_fine = self.should_compute_fidelity(generation, "fine")
        else:
            compute_coarse = "coarse" in force_fidelities
            compute_medium = "medium" in force_fidelities
            compute_fine = "fine" in force_fidelities

        # Compute coarse HV
        if compute_coarse:
            result = self.router.compute_hypervolume(
                pareto_front,
                self.reference_point,
                epsilon=self.coarse_epsilon,
                minimize=minimize,
                verbose=verbose,
            )

            new_state.hv_coarse = HVEstimate(
                value=result.hypervolume,
                generation=generation,
                epsilon=self.coarse_epsilon,
                computation_time_ms=result.computation_time_ms,
                algorithm=result.algorithm_used,
                front_hash=front_hash,
            )
            new_state.history_coarse.append(result.hypervolume)

        # Compute medium HV
        if compute_medium:
            result = self.router.compute_hypervolume(
                pareto_front,
                self.reference_point,
                epsilon=self.medium_epsilon,
                minimize=minimize,
                verbose=verbose,
            )

            new_state.hv_medium = HVEstimate(
                value=result.hypervolume,
                generation=generation,
                epsilon=self.medium_epsilon,
                computation_time_ms=result.computation_time_ms,
                algorithm=result.algorithm_used,
                front_hash=front_hash,
            )
            new_state.history_medium.append(result.hypervolume)

        # Compute fine HV
        if compute_fine:
            result = self.router.compute_hypervolume(
                pareto_front,
                self.reference_point,
                epsilon=self.fine_epsilon,
                minimize=minimize,
                verbose=verbose,
            )

            new_state.hv_fine = HVEstimate(
                value=result.hypervolume,
                generation=generation,
                epsilon=self.fine_epsilon,
                computation_time_ms=result.computation_time_ms,
                algorithm=result.algorithm_used,
                front_hash=front_hash,
            )
            new_state.history_fine.append(result.hypervolume)

        # Update internal state
        self.state = new_state

        return new_state

    def get_best_estimate(
        self, generation: int, max_age: int = 10
    ) -> Optional[HVEstimate]:
        """
        Get best available HV estimate (finest with acceptable age).

        Parameters
        ----------
        generation : int
            Current generation
        max_age : int
            Maximum generation age for cached estimate

        Returns
        -------
        HVEstimate or None
            Best available estimate or None
        """
        candidates = []

        # Check fine estimate
        if self.state.hv_fine is not None:
            age = generation - self.state.hv_fine.generation
            if age <= max_age:
                candidates.append((0, self.state.hv_fine))

        # Check medium estimate
        if self.state.hv_medium is not None:
            age = generation - self.state.hv_medium.generation
            if age <= max_age:
                candidates.append((1, self.state.hv_medium))

        # Check coarse estimate
        if self.state.hv_coarse is not None:
            age = generation - self.state.hv_coarse.generation
            if age <= max_age:
                candidates.append((2, self.state.hv_coarse))

        if not candidates:
            return None

        # Return finest (lowest priority number)
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]


# ============================================================================
# Convergence Detector
# ============================================================================


class ConvergenceDetector:
    """
    Detects optimization convergence using HV-based criteria.

    Pure function design with configurable thresholds. Uses multi-stage
    verification to reduce false positives.
    """

    def __init__(
        self,
        stagnation_threshold: float = 0.01,
        stagnation_window: int = 5,
        relative_threshold: float = 0.001,
        min_generations: int = 20,
        confidence_level: float = 0.95,
    ):
        """
        Initialize convergence detector.

        Parameters
        ----------
        stagnation_threshold : float
            Absolute HV change threshold
        stagnation_window : int
            Number of generations to check
        relative_threshold : float
            Relative HV change threshold
        min_generations : int
            Minimum generations before convergence
        confidence_level : float
            Statistical confidence level
        """
        self.stagnation_threshold = stagnation_threshold
        self.stagnation_window = stagnation_window
        self.relative_threshold = relative_threshold
        self.min_generations = min_generations
        self.confidence_level = confidence_level

    def check_convergence(
        self,
        mf_tracker: MultiFidelityHVTracker,
        current_generation: int,
        pareto_front: np.ndarray,
        verbose: bool = False,
    ) -> ConvergenceResult:
        """
        Check if optimization has converged.

        Multi-stage convergence detection:
        1. Quick check with coarse HV
        2. Verify with medium HV
        3. Confirm with fine HV

        Parameters
        ----------
        mf_tracker : MultiFidelityHVTracker
            Multi-fidelity HV tracker
        current_generation : int
            Current generation
        pareto_front : np.ndarray
            Current Pareto front
        verbose : bool
            Print diagnostic information

        Returns
        -------
        ConvergenceResult
            Decision with evidence and confidence
        """
        # Minimum generation requirement
        if current_generation < self.min_generations:
            return ConvergenceResult(
                converged=False,
                confidence=0.0,
                primary_reason="insufficient_generations",
                supporting_evidence={"generation": current_generation},
                recommendation="Continue optimization",
            )

        # Stage 1: Coarse HV check
        coarse_stagnant, coarse_evidence = self._check_stagnation(
            mf_tracker.state.history_coarse, "coarse"
        )

        if not coarse_stagnant:
            return ConvergenceResult(
                converged=False,
                confidence=0.0,
                primary_reason="hv_still_improving",
                supporting_evidence=coarse_evidence,
                recommendation="Continue optimization",
            )

        if verbose:
            print("Coarse HV shows stagnation, verifying with medium fidelity...")

        # Stage 2: Medium HV verification
        if not mf_tracker.should_compute_fidelity(current_generation, "medium"):
            mf_tracker.compute_and_update(
                pareto_front,
                current_generation,
                force_fidelities=["medium"],
                verbose=verbose,
            )

        medium_stagnant, medium_evidence = self._check_stagnation(
            mf_tracker.state.history_medium, "medium"
        )

        if not medium_stagnant:
            return ConvergenceResult(
                converged=False,
                confidence=0.3,
                primary_reason="false_convergence_medium",
                supporting_evidence={
                    "coarse": coarse_evidence,
                    "medium": medium_evidence,
                },
                recommendation="Continue - coarse signal was noise",
            )

        if verbose:
            print("Medium HV confirms stagnation, final check with fine fidelity...")

        # Stage 3: Fine HV confirmation
        mf_tracker.compute_and_update(
            pareto_front, current_generation, force_fidelities=["fine"], verbose=verbose
        )

        fine_stagnant, fine_evidence = self._check_stagnation(
            mf_tracker.state.history_fine, "fine"
        )

        if not fine_stagnant:
            return ConvergenceResult(
                converged=False,
                confidence=0.6,
                primary_reason="false_convergence_fine",
                supporting_evidence={
                    "coarse": coarse_evidence,
                    "medium": medium_evidence,
                    "fine": fine_evidence,
                },
                recommendation="Continue - medium signal was noise",
            )

        # All three fidelities confirm stagnation
        confidence = self._compute_confidence(
            coarse_evidence, medium_evidence, fine_evidence
        )

        return ConvergenceResult(
            converged=True,
            confidence=confidence,
            primary_reason="hv_stagnation_confirmed",
            supporting_evidence={
                "coarse": coarse_evidence,
                "medium": medium_evidence,
                "fine": fine_evidence,
            },
            recommendation="Optimization converged - terminate",
        )

    def _check_stagnation(
        self, hv_history: List[float], fidelity: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if HV history shows stagnation.

        Pure function.

        Parameters
        ----------
        hv_history : list of float
            Recent HV values
        fidelity : str
            Fidelity level name

        Returns
        -------
        tuple
            (is_stagnant, evidence_dict)
        """
        if len(hv_history) < self.stagnation_window:
            return False, {"reason": "insufficient_history", "length": len(hv_history)}

        recent_hv = np.array(hv_history[-self.stagnation_window :])

        # Absolute change criterion
        max_change = np.max(np.abs(np.diff(recent_hv)))
        absolute_stagnant = max_change < self.stagnation_threshold

        # Relative change criterion
        mean_hv = np.mean(recent_hv)
        relative_changes = np.abs(np.diff(recent_hv)) / (mean_hv + 1e-10)
        max_relative_change = np.max(relative_changes)
        relative_stagnant = max_relative_change < self.relative_threshold

        # Statistical test: trend is not significant
        try:
            x = np.arange(len(recent_hv))
            slope, intercept, r_value, p_value, std_err = linregress(x, recent_hv)
            trend_not_significant = p_value > 0.05
        except ImportError:
            # Fallback if scipy not available
            trend_not_significant = True
            p_value = 1.0
            slope = 0.0

        # Overall stagnation: all criteria met
        is_stagnant = absolute_stagnant and relative_stagnant and trend_not_significant

        evidence = {
            "fidelity": fidelity,
            "window_size": self.stagnation_window,
            "max_absolute_change": float(max_change),
            "max_relative_change": float(max_relative_change),
            "trend_slope": float(slope),
            "trend_p_value": float(p_value),
            "mean_hv": float(mean_hv),
            "absolute_stagnant": absolute_stagnant,
            "relative_stagnant": relative_stagnant,
            "trend_not_significant": trend_not_significant,
            "overall_stagnant": is_stagnant,
        }

        return is_stagnant, evidence

    @staticmethod
    def _compute_confidence(
        coarse_evidence: Dict, medium_evidence: Dict, fine_evidence: Dict
    ) -> float:
        """
        Compute confidence in convergence decision.

        Pure function based on evidence consistency.

        Parameters
        ----------
        coarse_evidence : dict
            Evidence from coarse HV
        medium_evidence : dict
            Evidence from medium HV
        fine_evidence : dict
            Evidence from fine HV

        Returns
        -------
        float
            Confidence score [0, 1]
        """
        # Base confidence from fine estimate
        confidence = 0.7

        # Bonus for consistency across fidelities
        if coarse_evidence["overall_stagnant"]:
            confidence += 0.1

        if medium_evidence["overall_stagnant"]:
            confidence += 0.1

        # Bonus for strong statistical evidence
        if fine_evidence["trend_p_value"] > 0.1:
            confidence += 0.05

        if fine_evidence["max_relative_change"] < 0.0005:
            confidence += 0.05

        return min(confidence, 1.0)


# ============================================================================
# Main class: Adaptive Hypervolume Progress Termination
# ============================================================================


class HypervolumeProgressTermination(SlidingWindowTermination):
    """
    Adaptive hypervolume-based termination with multi-fidelity tracking:

    - Uses progressive precision (coarse - medium - fine)
    - Routes to optimal HV algorithm based on dimensionality
    - Performs multi-stage verification to avoid false convergence
    - Scales efficiently to high dimensions (d > 20)

    Based on Deng & Zhang (2020), Yang et al. (2019), While et al. (2012).
    """

    def __init__(
        self,
        problem,
        ref_point: Optional[np.ndarray] = None,
        hv_tol: float = 1e-5,
        n_last: int = 15,
        nth_gen: int = 5,
        n_max_gen: Optional[int] = None,
        adaptive_ref_point: bool = True,
        min_generations: int = 20,
        verbose: bool = False,
        **kwargs,
    ):
        """
        Initialize adaptive hypervolume progress termination.

        Parameters
        ----------
        problem : optimization problem
            Problem instance with n_objectives and logger
        ref_point : np.ndarray, optional
            Reference point for hypervolume (None = auto)
        hv_tol : float
            Tolerance for hypervolume stagnation
        n_last : int
            Window size for tracking convergence
        nth_gen : int
            Check convergence every n-th generation
        n_max_gen : int, optional
            Maximum generations
        adaptive_ref_point : bool
            Whether to adapt reference point during optimization
        min_generations : int
            Minimum generations before convergence possible
        verbose : bool
            Print detailed progress information
        """
        super().__init__(
            problem,
            metric_window_size=n_last,
            data_window_size=2,
            min_data_for_metric=2,
            nth_gen=nth_gen,
            n_max_gen=n_max_gen,
            **kwargs,
        )

        self.ref_point = ref_point.copy() if ref_point is not None else None
        self.hv_tol = hv_tol
        self.adaptive_ref_point = adaptive_ref_point
        self.verbose = verbose

        # Initialize components (lazy - only when needed)
        self._precision_scheduler = None
        self._mf_tracker = None
        self._convergence_detector = None

        self.convergence_detector_config = {
            "stagnation_threshold": hv_tol,
            "stagnation_window": min(n_last, 5),
            "relative_threshold": hv_tol / 10,
            "min_generations": min_generations,
        }

    def _initialize_components(self, F: np.ndarray):
        """Lazy initialization of components."""
        if self._mf_tracker is not None:
            return  # Already initialized

        # Initialize or update reference point
        if self.ref_point is None or self.adaptive_ref_point:
            margin = 0.1
            worst = F.max(axis=0)
            best = F.min(axis=0)
            range_vals = worst - best
            self.ref_point = worst + margin * np.abs(range_vals)

        # Initialize precision scheduler
        self._precision_scheduler = ProgressivePrecisionScheduler(
            early_threshold=20,
            mid_threshold=50,
            early_epsilon=0.05,
            mid_epsilon=0.02,
            late_epsilon=0.01,
        )

        # Initialize multi-fidelity tracker
        self._mf_tracker = MultiFidelityHVTracker(
            reference_point=self.ref_point,
            coarse_epsilon=0.05,
            medium_epsilon=0.02,
            fine_epsilon=0.01,
            coarse_freq=1,
            medium_freq=5,
            fine_freq=10,
        )

        # Initialize convergence detector
        self._convergence_detector = ConvergenceDetector(
            **self.convergence_detector_config
        )

    def _store(self, opt):
        """Store objective values and update reference point."""
        F = opt.y

        # Lazy initialization
        self._initialize_components(F)

        # Update reference point if adaptive
        if self.adaptive_ref_point:
            margin = 0.1
            worst = F.max(axis=0)
            best = F.min(axis=0)
            range_vals = worst - best
            self.ref_point = worst + margin * np.abs(range_vals)
            if self._mf_tracker is not None:
                self._mf_tracker.reference_point = self.ref_point

        return {"F": F, "ref_point": self.ref_point.copy()}

    def _metric(self, data):
        """Calculate hypervolume and check for convergence."""
        _, current = data[-2], data[-1]
        F_current = current["F"]

        # Get current generation
        generation = (
            len(self._mf_tracker.state.history_coarse) if self._mf_tracker else 0
        )

        # Update multi-fidelity HV estimates
        self._mf_tracker.compute_and_update(
            F_current, generation, minimize=True, verbose=self.verbose
        )

        # Get best available estimate
        best_estimate = self._mf_tracker.get_best_estimate(generation, max_age=10)

        hv_current = best_estimate.value if best_estimate else 0.0

        # Calculate improvement
        history = self._mf_tracker.state.history_coarse
        if len(history) >= 2:
            hv_improvement = history[-1] - history[-2]
            relative_improvement = hv_improvement / (history[-2] + 1e-10)
        else:
            hv_improvement = 0.0
            relative_improvement = 0.0

        # Check convergence
        convergence_result = self._convergence_detector.check_convergence(
            self._mf_tracker, generation, F_current, verbose=self.verbose
        )

        return {
            "hv": hv_current,
            "hv_improvement": hv_improvement,
            "relative_improvement": relative_improvement,
            "converged": convergence_result.converged,
            "confidence": convergence_result.confidence,
            "reason": convergence_result.primary_reason,
        }

    def _decide(self, metrics):
        """Terminate based on convergence detection."""
        if len(metrics) < 3:
            return True

        latest = metrics[-1]

        # Check if convergence detected
        if latest["converged"]:
            self.problem.logger.info(
                f"Hypervolume convergence detected\n"
                f"  Final HV: {latest['hv']:.6f}\n"
                f"  Confidence: {latest['confidence']:.2%}\n"
                f"  Reason: {latest['reason']}"
            )
            return False

        # Continue optimization
        self.problem.logger.info(
            f"HV Progress - Current: {latest['hv']:.6f}, "
            f"Improvement: {latest['relative_improvement']:.2e}, "
            f"Confidence: {latest['confidence']:.2%}"
        )

        return True
