"""
Termination Criteria for High-Dimensional Multi-Objective Optimization

This module provides termination criteria for complex multi-objective
optimization problems with many objectives.

Key features:
- Per-objective convergence tracking
- Hypervolume-based progress monitoring (more robust in high dimensions)
- Multi-scale stagnation detection
- Adaptive window sizing based on problem characteristics
- Landscape complexity indicators

"""

import numpy as np
from typing import List, Optional
from dataclasses import dataclass
from collections import deque

from dmosopt.termination import (
    Termination,
    SlidingWindowTermination,
    MaximumGenerationTermination,
    TerminationCollection,
)
from dmosopt.indicators import crowding_distance_metric
from dmosopt.hv_termination import HypervolumeProgressTermination


@dataclass
class ConvergenceState:
    """Tracks convergence state for a single objective or metric."""

    values: deque
    converged: bool = False
    stagnation_count: int = 0
    improvement_rate: float = 0.0

    def __post_init__(self):
        if not isinstance(self.values, deque):
            self.values = deque(
                self.values,
                maxlen=self.values.maxlen if hasattr(self.values, "maxlen") else None,
            )


class PerObjectiveConvergence(SlidingWindowTermination):
    """
    Tracks convergence of each objective independently, allowing for
    heterogeneous convergence rates across objectives.

    Terminates when a specified fraction of objectives have converged.
    More suitable for high-dimensional objective spaces where different
    objectives may converge at vastly different rates.
    """

    def __init__(
        self,
        problem,
        obj_tol: float = 1e-4,
        min_converged_fraction: float = 0.8,
        n_last: int = 20,
        nth_gen: int = 5,
        n_max_gen: Optional[int] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        problem : optimization problem instance
        obj_tol : tolerance for considering an objective converged
        min_converged_fraction : minimum fraction of objectives that must converge
        n_last : window size for tracking convergence
        nth_gen : check convergence every n-th generation
        n_max_gen : maximum generations
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
        self.n_objectives = problem.n_objectives
        self.obj_tol = obj_tol
        self.min_converged_fraction = min_converged_fraction

        # Track per-objective states
        self.objective_states = [
            ConvergenceState(values=deque(maxlen=n_last))
            for _ in range(self.n_objectives)
        ]

    def _store(self, opt):
        """Store ideal and nadir points for each objective."""
        F = opt.y
        return {"ideal": F.min(axis=0), "nadir": F.max(axis=0), "F": F}

    def _metric(self, data):
        """Calculate per-objective improvement rates."""
        last, current = data[-2], data[-1]

        # Normalize range
        norm = current["nadir"] - current["ideal"]
        norm[norm < 1e-32] = 1.0

        # Per-objective change in ideal point
        delta_ideal = np.abs(current["ideal"] - last["ideal"]) / norm

        # Update objective states
        for i, delta in enumerate(delta_ideal):
            self.objective_states[i].values.append(delta)

            # Check if objective has converged
            if len(self.objective_states[i].values) >= self.metric_window_size:
                mean_change = np.mean(self.objective_states[i].values)
                self.objective_states[i].improvement_rate = mean_change

                if mean_change < self.obj_tol:
                    self.objective_states[i].stagnation_count += 1
                    if self.objective_states[i].stagnation_count >= 3:
                        self.objective_states[i].converged = True
                else:
                    self.objective_states[i].stagnation_count = 0
                    self.objective_states[i].converged = False

        return {
            "delta_ideal": delta_ideal,
            "converged_objectives": sum(s.converged for s in self.objective_states),
            "mean_improvement": np.mean(
                [s.improvement_rate for s in self.objective_states]
            ),
        }

    def _decide(self, metrics):
        """Terminate when sufficient objectives have converged."""
        latest = metrics[-1]
        n_converged = latest["converged_objectives"]
        converged_fraction = n_converged / self.n_objectives

        if converged_fraction >= self.min_converged_fraction:
            self.problem.logger.info(
                f"Optimization terminated: {n_converged}/{self.n_objectives} objectives "
                f"({converged_fraction:.1%}) have converged (threshold: {self.min_converged_fraction:.1%})"
            )
            return False

        self.problem.logger.info(
            f"Convergence progress: {n_converged}/{self.n_objectives} objectives converged "
            f"({converged_fraction:.1%}), mean improvement rate: {latest['mean_improvement']:.2e}"
        )
        return True


class MultiScaleStagnationTermination(SlidingWindowTermination):
    """
    Detects stagnation at multiple timescales simultaneously.

    This is critical for complex optimization landscapes where progress
    may slow at different rates for different timescales. Inspired by
    multifractal analysis concepts from the neuroscience domain.
    """

    def __init__(
        self,
        problem,
        timescales: List[int] = [5, 10, 20, 40],
        stagnation_tol: float = 1e-4,
        min_scales_stagnant: int = 3,
        n_max_gen: Optional[int] = None,
        nth_gen: int = 1,
        **kwargs,
    ):
        """
        Parameters
        ----------
        problem : optimization problem instance
        timescales : list of window sizes representing different timescales
        stagnation_tol : tolerance for considering a scale stagnant
        min_scales_stagnant : minimum number of scales that must be stagnant
        n_max_gen : maximum generations
        nth_gen : check every n-th generation
        """
        max_scale = max(timescales)
        super().__init__(
            problem,
            metric_window_size=max_scale,
            data_window_size=max_scale,
            min_data_for_metric=max(timescales),
            nth_gen=nth_gen,
            n_max_gen=n_max_gen,
            **kwargs,
        )
        self.timescales = sorted(timescales)
        self.stagnation_tol = stagnation_tol
        self.min_scales_stagnant = min_scales_stagnant

        # Track metrics at each scale
        self.scale_metrics = {scale: deque(maxlen=scale) for scale in timescales}

    def _store(self, opt):
        """Store both objective and parameter information."""
        F = opt.y
        X = opt.x

        # Compute summary statistics
        ideal = F.min(axis=0)
        nadir = F.max(axis=0)

        # Diversity metric
        cd = crowding_distance_metric(F)
        diversity = np.mean(cd)

        return {"ideal": ideal, "nadir": nadir, "diversity": diversity, "F": F, "X": X}

    def _metric(self, data):
        """Calculate improvement at multiple timescales."""
        if len(data) < 2:
            return None

        current = data[-1]
        scale_improvements = {}

        for scale in self.timescales:
            if len(data) >= scale + 1:
                past = data[-(scale + 1)]

                # Calculate normalized change in ideal point
                norm = current["nadir"] - current["ideal"]
                norm[norm < 1e-32] = 1.0

                delta_ideal = np.abs(current["ideal"] - past["ideal"]) / norm
                mean_delta = np.mean(delta_ideal)

                # Calculate diversity change
                delta_diversity = abs(current["diversity"] - past["diversity"])

                scale_improvements[scale] = {
                    "ideal_change": mean_delta,
                    "diversity_change": delta_diversity,
                    "stagnant": mean_delta < self.stagnation_tol,
                }

        return scale_improvements

    def _decide(self, metrics):
        """Terminate when multiple scales show stagnation."""
        latest = metrics[-1]

        if latest is None:
            return True

        # Count stagnant scales
        stagnant_scales = [scale for scale, info in latest.items() if info["stagnant"]]

        n_stagnant = len(stagnant_scales)

        if n_stagnant >= self.min_scales_stagnant:
            self.problem.logger.info(
                f"Optimization terminated: {n_stagnant}/{len(self.timescales)} "
                f"timescales show stagnation (threshold: {self.min_scales_stagnant}). "
                f"Stagnant scales: {stagnant_scales}"
            )
            return False

        # Report progress
        improvements = {scale: info["ideal_change"] for scale, info in latest.items()}
        self.problem.logger.info(
            f"Multi-scale progress - stagnant: {n_stagnant}/{len(self.timescales)}, "
            f"improvements by scale: {improvements}"
        )
        return True


class AdaptiveWindowTermination(SlidingWindowTermination):
    """
    Automatically adjusts window size based on problem complexity and
    optimization progress. Starts with small windows for fast initial
    convergence detection, expands windows as optimization progresses.
    """

    def __init__(
        self,
        problem,
        initial_window: int = 10,
        max_window: int = 50,
        expansion_rate: float = 1.2,
        tol: float = 1e-4,
        n_max_gen: Optional[int] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        problem : optimization problem instance
        initial_window : starting window size
        max_window : maximum window size
        expansion_rate : factor to expand window when progress detected
        tol : tolerance for convergence
        n_max_gen : maximum generations
        """
        super().__init__(
            problem,
            metric_window_size=initial_window,
            data_window_size=2,
            min_data_for_metric=2,
            nth_gen=1,
            n_max_gen=n_max_gen,
            **kwargs,
        )
        self.initial_window = initial_window
        self.max_window = max_window
        self.expansion_rate = expansion_rate
        self.tol = tol
        self.current_window_size = initial_window

    def _store(self, opt):
        F = opt.y
        return {"ideal": F.min(axis=0), "nadir": F.max(axis=0)}

    def _metric(self, data):
        last, current = data[-2], data[-1]

        norm = current["nadir"] - current["ideal"]
        norm[norm < 1e-32] = 1.0

        delta = np.mean(np.abs(current["ideal"] - last["ideal"]) / norm)

        return {"delta": delta, "window_size": self.current_window_size}

    def _decide(self, metrics):
        if len(metrics) < self.current_window_size:
            return True

        recent_deltas = [m["delta"] for m in metrics[-self.current_window_size :]]
        mean_delta = np.mean(recent_deltas)

        # If making progress, expand window (be more patient)
        if mean_delta > self.tol * 10:
            new_window = min(
                int(self.current_window_size * self.expansion_rate), self.max_window
            )
            if new_window > self.current_window_size:
                self.current_window_size = new_window
                self.metric_window_size = new_window
                self.problem.logger.info(
                    f"Expanding patience window to {self.current_window_size} "
                    f"(progress detected: {mean_delta:.2e})"
                )

        # Terminate if stagnant
        if mean_delta < self.tol:
            self.problem.logger.info(
                f"Optimization terminated: mean change {mean_delta:.2e} "
                f"below tolerance {self.tol:.2e} over {self.current_window_size} generations"
            )
            return False

        return True


class CompositeAdaptiveTermination(TerminationCollection):
    """
    Combines multiple termination criteria specifically designed for
    high-dimensional multi-objective problems.
    """

    def __init__(
        self,
        problem,
        n_max_gen: int = 2000,
        # Per-objective convergence params
        obj_tol: float = 1e-4,
        min_converged_fraction: float = 0.8,
        # Hypervolume params
        hv_tol: float = 1e-5,
        ref_point: Optional[np.ndarray] = None,
        # Multi-scale params
        timescales: Optional[List[int]] = None,
        stagnation_tol: float = 1e-4,
        # Control flags
        use_per_objective: bool = True,
        use_hypervolume: bool = True,
        use_multiscale: bool = True,
        **kwargs,
    ):
        """
        Parameters
        ----------
        problem : optimization problem instance
        n_max_gen : maximum generations
        obj_tol : tolerance for per-objective convergence
        min_converged_fraction : fraction of objectives that must converge
        hv_tol : tolerance for hypervolume improvement
        ref_point : reference point for hypervolume (None = auto)
        timescales : timescales for multi-scale analysis (None = auto)
        stagnation_tol : tolerance for stagnation detection
        use_per_objective : enable per-objective termination
        use_hypervolume : enable hypervolume termination
        use_multiscale : enable multi-scale termination
        """

        terminations = [MaximumGenerationTermination(problem, n_max_gen=n_max_gen)]

        if use_per_objective:
            terminations.append(
                PerObjectiveConvergence(
                    problem=problem,
                    obj_tol=obj_tol,
                    min_converged_fraction=min_converged_fraction,
                    n_last=20,
                    nth_gen=5,
                    **kwargs,
                )
            )

        if use_hypervolume:
            terminations.append(
                HypervolumeProgressTermination(
                    problem=problem,
                    ref_point=ref_point,
                    hv_tol=hv_tol,
                    n_last=15,
                    nth_gen=5,
                    **kwargs,
                )
            )

        if use_multiscale:
            if timescales is None:
                # Auto-determine timescales based on problem size
                base_scale = max(5, problem.n_objectives // 5)
                timescales = [base_scale * (2**i) for i in range(4)]

            terminations.append(
                MultiScaleStagnationTermination(
                    problem=problem,
                    timescales=timescales,
                    stagnation_tol=stagnation_tol,
                    min_scales_stagnant=3,
                    nth_gen=2,
                    **kwargs,
                )
            )

        super().__init__(problem, *terminations)

        self.problem.logger.info(
            f"Initialized CompositeHighDimTermination with {len(terminations)} criteria:\n"
            f"  - Max generations: {n_max_gen}\n"
            f"  - Per-objective convergence: {use_per_objective}\n"
            f"  - Hypervolume progress: {use_hypervolume}\n"
            f"  - Multi-scale stagnation: {use_multiscale}"
        )


class ResourceAwareTermination(Termination):
    """
    Terminates optimization based on computational resource consumption
    rather than convergence metrics. Useful for time-constrained scenarios.
    """

    def __init__(
        self,
        problem,
        max_time_seconds: Optional[float] = None,
        max_function_evals: Optional[int] = None,
        target_quality_threshold: Optional[float] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        problem : optimization problem instance
        max_time_seconds : maximum wall-clock time
        max_function_evals : maximum number of function evaluations
        target_quality_threshold : stop if quality metric exceeds threshold
        """
        super().__init__(problem)
        self.max_time_seconds = max_time_seconds
        self.max_function_evals = max_function_evals
        self.target_quality_threshold = target_quality_threshold
        self.start_time = None

    def _do_continue(self, opt):
        import time

        if self.start_time is None:
            self.start_time = time.time()

        # Check time limit
        if self.max_time_seconds is not None:
            elapsed = time.time() - self.start_time
            if elapsed > self.max_time_seconds:
                self.problem.logger.info(
                    f"Optimization terminated: time limit reached "
                    f"({elapsed:.1f}s > {self.max_time_seconds:.1f}s)"
                )
                return False

        # Check evaluation limit
        if self.max_function_evals is not None:
            n_evals = getattr(
                opt, "n_evals", getattr(opt, "n_gen", 0) * getattr(opt, "pop_size", 1)
            )
            if n_evals > self.max_function_evals:
                self.problem.logger.info(
                    f"Optimization terminated: evaluation limit reached "
                    f"({n_evals} > {self.max_function_evals})"
                )
                return False

        # Check quality threshold
        if self.target_quality_threshold is not None:
            # This requires a quality metric to be available
            quality = getattr(opt, "quality_metric", None)
            if quality is not None and quality > self.target_quality_threshold:
                self.problem.logger.info(
                    f"Optimization terminated: quality threshold reached "
                    f"({quality:.6f} > {self.target_quality_threshold:.6f})"
                )
                return False

        return True


# Convenience function
def create_adaptive_termination(
    problem, n_max_gen: int = 2000, strategy: str = "comprehensive", **kwargs
) -> Termination:
    """
    Factory function to create appropriate adaptive termination
    criteria for high-dimensional problems.

    Parameters
    ----------
    problem : optimization problem instance
    n_max_gen : maximum generations
    strategy : termination strategy
        - 'comprehensive': all criteria (default)
        - 'fast': hypervolume + multiscale only
        - 'conservative': per-objective + multiscale only
        - 'simple': hypervolume only
    **kwargs : additional arguments passed to termination criteria

    Returns
    -------
    termination : Termination instance

    Examples
    --------
    >>> # For a problem with 35 objectives
    >>> termination = create_adaptive_termination(
    ...     problem=my_problem,
    ...     n_max_gen=1500,
    ...     strategy='comprehensive'
    ... )

    >>> # For faster termination with limited compute
    >>> termination = create_adaptive_termination(
    ...     problem=my_problem,
    ...     n_max_gen=1000,
    ...     strategy='fast',
    ...     hv_tol=1e-4  # Less strict
    ... )

    """

    if strategy == "comprehensive":
        return CompositeAdaptiveTermination(
            problem=problem,
            n_max_gen=n_max_gen,
            use_per_objective=True,
            use_hypervolume=True,
            use_multiscale=True,
            hv_tol=1e-6,
            **kwargs,
        )

    elif strategy == "fast":
        return CompositeAdaptiveTermination(
            problem=problem,
            n_max_gen=n_max_gen,
            use_per_objective=False,
            use_hypervolume=True,
            use_multiscale=True,
            **kwargs,
        )

    elif strategy == "conservative":
        return CompositeAdaptiveTermination(
            problem=problem,
            n_max_gen=n_max_gen,
            use_per_objective=True,
            use_hypervolume=False,
            use_multiscale=True,
            **kwargs,
        )

    elif strategy == "simple":
        return HypervolumeProgressTermination(
            problem=problem, n_last=20, nth_gen=5, n_max_gen=n_max_gen, **kwargs
        )

    else:
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            f"Choose from: 'comprehensive', 'fast', 'conservative', 'simple'"
        )
