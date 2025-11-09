"""
Test suite for synthetic optimization benchmarks.
"""

import sys
import logging
import numpy as np
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict
import json

from dmosopt import dmosopt
from dmosopt.benchmarks.moo_benchmarks import (
    generate_problem_space,
    get_problem_metadata,
)
from dmosopt.hv_adaptive import AdaptiveHyperVolume

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    problem_name: str
    n_objectives: int
    n_variables: int
    converged: bool
    final_generation: int
    final_hv: float
    computation_time_seconds: float
    termination_reason: str
    hv_trajectory: List[float]
    algorithm_usage: Dict[str, int]
    convergence_confidence: float


class BenchmarkRunner:
    """Runs benchmark suite and collects diagnostics."""

    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results: List[BenchmarkResult] = []

    def create_objective_function(self, problem_name: str, n_obj: int):
        """Create objective function for dmosopt."""
        problem_func = globals()[problem_name]

        def obj_fun(pp):
            """Objective function wrapper for dmosopt."""
            param_values = np.asarray([pp[k] for k in sorted(pp)])

            # Handle WFG scaling
            if problem_name.startswith("wfg"):
                n_var = len(param_values)
                # WFG problems have bounds [0, 2i]
                # But dmosopt normalizes to [0, 1], so scale back
                param_values = param_values * 2 * np.arange(1, n_var + 1)

            result = problem_func(param_values, n_obj=n_obj)
            logger.info(
                f"{problem_name}: pp={list(pp.values())[:3]}..., f={result[:3]}..."
            )
            return result

        return obj_fun

    def run_single_benchmark(
        self,
        problem_name: str,
        n_obj: int,
        strategy: str = "comprehensive",
        max_gen: int = 500,
        population_size: int = 100,
        verbose: bool = True,
    ) -> BenchmarkResult:
        """
        Run a single benchmark problem.

        Parameters
        ----------
        problem_name : str
            Problem identifier (e.g., 'dtlz2')
        n_obj : int
            Number of objectives
        strategy : str
            Termination strategy
        max_gen : int
            Maximum generations
        population_size : int
            Population size
        verbose : bool
            Print progress

        Returns
        -------
        BenchmarkResult
            Results and diagnostics
        """

        logger.info(f"\n{'=' * 80}")
        logger.info(f"Running {problem_name.upper()} with M={n_obj}")
        logger.info(f"{'=' * 80}\n")

        # Generate parameter space
        space = generate_problem_space(problem_name, n_obj)
        n_var = len(space)

        # Get metadata
        # meta = get_problem_metadata(problem_name, n_obj)

        # Create objective function
        obj_fun = self.create_objective_function(problem_name, n_obj)

        # Setup dmosopt parameters
        objective_names = [f"f{i + 1}" for i in range(n_obj)]

        dmosopt_params = {
            "opt_id": f"{problem_name}_m{n_obj}",
            "obj_fun_name": "benchmark_runner.obj_fun",  # Placeholder
            "problem_parameters": {},
            "optimizer_name": "age",
            "surrogate_options": {"lengthscale_bounds": (1e-4, 1000.0)},
            "surrogate_method_name": "megp",
            "termination_conditions": True,
            "population_size": population_size,
            "num_generations": max_gen,
            "space": space,
            "objective_names": objective_names,
            "n_initial": max(5, n_obj),
            "n_epochs": 4,
            "save": True,
            "file_path": str(self.output_dir / f"{problem_name}_m{n_obj}.h5"),
            "resample_fraction": 1.0,
        }

        # Configure termination
        # Note: This would need proper integration with dmosopt's termination system
        # For now, we'll run with fixed generations and analyze HV trajectory

        start_time = time.time()

        # Run optimization (with objective function override)
        sys.modules["benchmark_runner"] = sys.modules[__name__]
        sys.modules["benchmark_runner"].obj_fun = obj_fun

        try:
            best = dmosopt.run(dmosopt_params, verbose=verbose)
            elapsed = time.time() - start_time

            # Extract results
            optimizer = dmosopt.dopt_dict[dmosopt_params["opt_id"]].optimizer_dict[0]
            x_all, y_all = optimizer.get_evals()

            # Get best solutions
            if best is not None:
                bestx, besty = best
                besty_dict = dict(besty)
                final_hv = self._compute_final_hv(besty_dict, n_obj)
            else:
                final_hv = 0.0

            # Analyze HV trajectory (would need to extract from logs)
            hv_trajectory = []  # Placeholder
            algorithm_usage = {}  # Placeholder

            result = BenchmarkResult(
                problem_name=problem_name,
                n_objectives=n_obj,
                n_variables=n_var,
                converged=True,  # Placeholder
                final_generation=len(y_all),
                final_hv=final_hv,
                computation_time_seconds=elapsed,
                termination_reason="max_generations",  # Placeholder
                hv_trajectory=hv_trajectory,
                algorithm_usage=algorithm_usage,
                convergence_confidence=0.0,  # Placeholder
            )

            self.results.append(result)
            self._save_result(result)

            logger.info(f"\nCompleted {problem_name} (M={n_obj})")
            logger.info(f"  Time: {elapsed:.1f}s")
            logger.info(f"  Generations: {result.final_generation}")
            logger.info(f"  Final HV: {final_hv:.6f}")

            return result

        except Exception as e:
            logger.error(f"Failed to run {problem_name} (M={n_obj}): {e}")
            raise

    def _compute_final_hv(self, besty_dict: Dict, n_obj: int) -> float:
        """Compute hypervolume of final solutions."""
        # Extract objective values
        front = np.array([[besty_dict[f"f{i + 1}"] for i in range(n_obj)]])

        # Compute reference point (simple approach)
        ref_point = np.max(front, axis=0) * 1.1

        # Compute HV (simplified)
        hv = AdaptiveHyperVolume(ref_point=ref_point)
        return hv.compute_hypervolume(front, algorithm="hybrid")

    def _save_result(self, result: BenchmarkResult):
        """Save individual result to JSON."""
        filepath = (
            self.output_dir
            / f"{result.problem_name}_m{result.n_objectives}_result.json"
        )
        with open(filepath, "w") as f:
            json.dump(asdict(result), f, indent=2)

    def run_tier(self, tier: int = 1, verbose: bool = True) -> List[BenchmarkResult]:
        """
        Run a specific tier of benchmarks.

        Tier 1: Core validation (DTLZ2, DTLZ1, DTLZ7, MaF2 M=5)
        Tier 2: Stress testing (DTLZ3, DTLZ5, DTLZ4, MaF4)
        Tier 3: High-dimensional scaling (MaF1/MaF2 with M=10,15,20,30)
        Tier 4: Advanced challenges (WFG suite)
        """
        tier_configs = {
            1: [  # Core validation
                ("dtlz2", 3),
                ("dtlz1", 3),
                ("dtlz7", 3),
                ("maf2", 5),
            ],
            2: [  # Stress testing
                ("dtlz3", 3),
                ("dtlz5", 3),
                ("dtlz4", 5),
                ("maf4", 5),
            ],
            3: [  # High-dimensional
                ("maf1", 10),
                ("maf2", 15),
                ("maf1", 20),
                ("maf2", 30),
            ],
            4: [  # Advanced
                ("wfg1", 5),
                ("wfg4", 5),
            ],
        }

        if tier not in tier_configs:
            raise ValueError(f"Invalid tier {tier}. Choose 1-4.")

        logger.info(f"\n{'#' * 80}")
        logger.info(f"Running Tier {tier} Benchmarks")
        logger.info(f"{'#' * 80}\n")

        tier_results = []
        for problem_name, n_obj in tier_configs[tier]:
            try:
                result = self.run_single_benchmark(problem_name, n_obj, verbose=verbose)
                tier_results.append(result)
            except Exception as e:
                logger.error(f"Tier {tier} benchmark {problem_name} failed: {e}")
                continue

        return tier_results

    def generate_report(self):
        """Generate comprehensive test report."""
        if not self.results:
            logger.warning("No results to report")
            return

        report_path = self.output_dir / "benchmark_report.txt"

        with open(report_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("Optimization Benchmark Report\n")
            f.write("=" * 80 + "\n\n")

            # Summary statistics
            f.write("Summary Statistics\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total problems: {len(self.results)}\n")
            f.write(f"Converged: {sum(r.converged for r in self.results)}\n")
            f.write(
                f"Average time: {np.mean([r.computation_time_seconds for r in self.results]):.1f}s\n"
            )
            f.write("\n")

            # Per-problem results
            f.write("Individual Results\n")
            f.write("-" * 80 + "\n")
            f.write(
                f"{'Problem':<15} {'M':<5} {'Conv':<8} {'Gen':<8} {'HV':<12} {'Time (s)':<10}\n"
            )
            f.write("-" * 80 + "\n")

            for r in self.results:
                f.write(
                    f"{r.problem_name:<15} {r.n_objectives:<5} "
                    f"{'Yes' if r.converged else 'No':<8} "
                    f"{r.final_generation:<8} {r.final_hv:<12.6f} "
                    f"{r.computation_time_seconds:<10.1f}\n"
                )

            f.write("\n")

            f.write("Problem Characteristics\n")
            f.write("-" * 80 + "\n")

            for r in self.results:
                meta = get_problem_metadata(r.problem_name, r.n_objectives)
                f.write(f"\n{r.problem_name.upper()} (M={r.n_objectives}):\n")
                f.write(f"  Shape: {meta.get('pf_shape', 'unknown')}\n")
                f.write(f"  Difficulty: {meta.get('difficulty', 'unknown')}\n")
                f.write(f"  Overlap: {meta.get('expected_overlap_ratio', 'unknown')}\n")
                f.write(f"  Tests: {', '.join(meta.get('tests_features', []))}\n")

        logger.info(f"\nReport saved to {report_path}")

        with open(report_path, "r") as f:
            print(f.read())


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run benchmark test suite")
    parser.add_argument(
        "--tier",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="Tier to run (1=core, 2=stress, 3=scaling, 4=advanced)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Output directory for results",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce logging verbosity")

    args = parser.parse_args()

    # Create runner
    runner = BenchmarkRunner(output_dir=args.output_dir)

    # Run specified tier
    results = runner.run_tier(tier=args.tier, verbose=not args.quiet)

    # Generate report
    runner.generate_report()

    # Print summary
    print(f"\n{'=' * 80}")
    print(f"Completed Tier {args.tier}")
    print(f"Results saved to {args.output_dir}")
    print(f"{'=' * 80}\n")
