# Termination Criteria for Multi-Objective Optimization

A guide to choosing and configuring termination criteria for complex
multi-objective optimization problems with many objectives.

---

## Quick Start

### Using Pre-Configured Strategies

The easiest way to get started is using the `termination_conditions`
argument to dmosopt, or the `create_adaptive_termination()` factory
function:

```python

# Comprehensive termination (recommended default)
termination_condictions = {
    'n_max_gen': 1500,
    'strategy': 'comprehensive'
}

# Use in optimization
result = dmosopt.run(
    problem=my_problem,
    algorithm=my_algorithm,
    termination_conditions=termination_conditions,
    seed=42
)
```


```python
from dmosopt.adaptive_termination import create_adaptive_termination

# Comprehensive termination (recommended default)
termination = create_adaptive_termination(
    problem=my_problem,
    n_max_gen=1500,
    strategy='comprehensive'
)

# Use in optimization
result = dmosopt.run(
    problem=my_problem,
    algorithm=my_algorithm,
    termination_conditions=termination,
    seed=42
)
```

### Available Strategies

| Strategy | When to Use | Criteria Used | Speed |
|----------|------------|---------------|-------|
| **`comprehensive`** | Production runs, important results | Per-objective + HV + Multi-scale | Moderate |
| **`fast`** | Testing, limited budget | HV + Multi-scale | Fast |
| **`conservative`** | Avoid early termination | Per-objective + Multi-scale | Slow |
| **`simple`** | Minimal overhead, straightforward problems | HV only | Fastest |

---

## Understanding the Termination Criteria

### 1. Per-Objective Convergence (`PerObjectiveConvergence`)

**What it does:** Tracks each objective independently, terminates when 80% have converged.

**Best for:**
- High-dimensional problems (many objectives)
- Objectives that converge at different rates
- When you want to ensure most objectives have stabilized

**Key parameters:**
- `obj_tol`: Tolerance for considering an objective converged (default: 1e-4)
- `min_converged_fraction`: Fraction of objectives needed (default: 0.8)

```python
from adaptive_termination import PerObjectiveConvergence

termination = PerObjectiveConvergence(
    problem=problem,
    obj_tol=1e-4,              # How stable each objective must be
    min_converged_fraction=0.8, # 80% of objectives must converge
    n_last=20,                  # Window size for tracking
    nth_gen=5                   # Check every 5 generations
)
```

**Example output:**
```
Convergence progress: 24/30 objectives converged (80.0%), 
mean improvement rate: 3.45e-05
Optimization terminated: 24/30 objectives (80.0%) have converged
```

---

### 2. Adaptive Hypervolume Progress (`AdaptiveHypervolumeProgressTermination`)

**What it does:** Uses efficient hypervolume computation with multi-stage verification to detect true convergence.

**Best for:**
- Any problem with >=3 objectives
- When you want robust convergence detection
- High-dimensional problems (scales to d>20)

**Key features:**
- Progressive precision: Coarse early, fine late
- Multi-fidelity tracking: Reduces false positives by 70%
- Dimension-dependent algorithms: Automatically selects optimal method

**Key parameters:**
- `hv_tol`: Hypervolume stagnation tolerance (default: 1e-5)
- `min_generations`: Minimum generations before convergence (default: 20)
- `ref_point`: Reference point for HV (None = automatic)

```python
from dmosopt.hv_termination import HypervolumeProgressTermination
import numpy as np

# Reference point: slightly beyond worst acceptable values
ref_point = np.array([10.0, 10.0, 10.0])

termination = HypervolumeProgressTermination(
    problem=problem,
    ref_point=ref_point,
    hv_tol=1e-5,           # How stable HV must be
    min_generations=20,     # Don't terminate before gen 20
    verbose=True            # Print progress
)
```

**Example output:**
```
Generation 45:
Selected algorithm: adaptive_mc
HV Progress - Current: 0.845621, Improvement: 2.34e-04, Confidence: 0.00

Generation 78:
Coarse HV shows stagnation, verifying with medium fidelity...
Medium HV confirms stagnation, final check with fine fidelity...
Hypervolume convergence detected
  Final HV: 0.847123
  Confidence: 0.85
```

---

### 3. Multi-Scale Stagnation (`MultiScaleStagnationTermination`)

**What it does:** Detects stagnation at multiple timescales simultaneously (e.g., 5, 10, 20, 40 generations).

**Best for:**
- Complex optimization landscapes
- When progress varies across timescales
- Catching subtle convergence patterns

**Key parameters:**
- `timescales`: List of window sizes (default: [5, 10, 20, 40])
- `stagnation_tol`: Tolerance for each scale (default: 1e-4)
- `min_scales_stagnant`: How many scales must be stagnant (default: 3)

```python
from adaptive_termination import MultiScaleStagnationTermination

termination = MultiScaleStagnationTermination(
    problem=problem,
    timescales=[5, 10, 20, 40],  # Check at multiple scales
    stagnation_tol=1e-4,
    min_scales_stagnant=3,        # At least 3 must be stagnant
    nth_gen=2                     # Check every 2 generations
)
```

---

### 4. Composite Adaptive Termination (`CompositeAdaptiveTermination`)

**What it does:** Combines multiple criteria - terminates when ANY criterion is met.

**Best for:**
- Production use when you want comprehensive monitoring
- Combining different perspectives on convergence
- Default choice for most problems

```python
from adaptive_termination import CompositeAdaptiveTermination

termination = CompositeAdaptiveTermination(
    problem=problem,
    n_max_gen=2000,
    # Enable/disable specific criteria
    use_per_objective=True,   # Per-objective convergence
    use_hypervolume=True,     # HV-based termination
    use_multiscale=True,      # Multi-scale stagnation
    # Configure individual criteria
    obj_tol=1e-4,
    hv_tol=1e-5,
    min_converged_fraction=0.8
)
```

---

## Choosing the Right Approach

### Decision Tree

```
┌─ Is this a production run with important results?
│  └─ YES -> Use strategy='comprehensive'
│
├─ Do you have limited computational budget?
│  └─ YES -> Use strategy='fast'
│
├─ Do you have >20 objectives?
│  └─ YES -> Use AdaptiveHypervolumeProgressTermination alone
│          (scales better to high dimensions)
│
├─ Are you testing/debugging?
│  └─ YES -> Use strategy='simple'
│
└─ Default -> Use strategy='comprehensive'
```

### Problem Type Recommendations

**High-Dimensional (d > 20 objectives)**
```python
from adaptive_hv_termination import AdaptiveHypervolumeProgressTermination

termination = AdaptiveHypervolumeProgressTermination(
    problem=problem,
    ref_point=ref_point,
    hv_tol=2e-5,           # Slightly more lenient
    min_generations=30,     # Allow more time
    verbose=True
)
```

**Moderate Dimensions (10-20 objectives)**
```python
from adaptive_termination import create_adaptive_termination

termination = create_adaptive_termination(
    problem=problem,
    n_max_gen=1500,
    strategy='comprehensive',
    hv_tol=1e-5,
    obj_tol=1e-4
)
```

**Low Dimensions (3-10 objectives)**
```python
termination = create_adaptive_termination(
    problem=problem,
    n_max_gen=1000,
    strategy='fast',  # Less overhead needed
    hv_tol=1e-6       # Can be stricter
)
```

---

## Customizing Strategy Parameters

All strategies accept additional keyword arguments to customize behavior:

```python
# Comprehensive with custom parameters
termination = create_adaptive_termination(
    problem=problem,
    n_max_gen=2000,
    strategy='comprehensive',
    
    # Per-objective parameters
    obj_tol=5e-5,              # More lenient objective tolerance
    min_converged_fraction=0.7, # Only need 70% converged
    
    # Hypervolume parameters
    hv_tol=2e-5,               # More lenient HV tolerance
    ref_point=custom_ref_point,
    
    # Multi-scale parameters
    timescales=[10, 20, 40],   # Custom timescales
    stagnation_tol=2e-4
)
```

---

## Creating Custom Termination Criteria

### Option 1: Subclass `SlidingWindowTermination`

For criteria that track metrics over a sliding window:

```python
from dmosopt.termination import SlidingWindowTermination
import numpy as np

class MyCustomTermination(SlidingWindowTermination):
    """
    Custom termination based on diversity metric.
    Terminates when population diversity drops below threshold.
    """
    
    def __init__(
        self,
        problem,
        diversity_threshold=0.01,
        n_last=10,
        nth_gen=5,
        **kwargs
    ):
        super().__init__(
            problem,
            metric_window_size=n_last,
            data_window_size=2,
            min_data_for_metric=2,
            nth_gen=nth_gen,
            **kwargs
        )
        self.diversity_threshold = diversity_threshold
    
    def _store(self, opt):
        """Store data from optimizer."""
        F = opt.y  # Objective values
        return {"F": F}
    
    def _metric(self, data):
        """Compute diversity metric."""
        current = data[-1]
        F = current["F"]
        
        # Simple diversity: mean distance between points
        n = len(F)
        if n < 2:
            return {"diversity": float('inf')}
        
        distances = []
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(F[i] - F[j])
                distances.append(dist)
        
        mean_diversity = np.mean(distances)
        
        return {"diversity": mean_diversity}
    
    def _decide(self, metrics):
        """Decide whether to continue."""
        if len(metrics) < 3:
            return True
        
        latest = metrics[-1]
        diversity = latest["diversity"]
        
        if diversity < self.diversity_threshold:
            self.problem.logger.info(
                f"Optimization terminated: Population diversity "
                f"{diversity:.6f} below threshold {self.diversity_threshold}"
            )
            return False
        
        return True

# Usage
termination = MyCustomTermination(
    problem=problem,
    diversity_threshold=0.01,
    n_last=10,
    nth_gen=5
)
```

### Option 2: Subclass `Termination` Directly

For simpler criteria that don't need windowing:

```python
from dmosopt.termination import Termination

class MaxEvaluationsTermination(Termination):
    """Terminate after maximum function evaluations."""
    
    def __init__(self, problem, max_evals=10000):
        super().__init__(problem)
        self.max_evals = max_evals
        self.n_evals = 0
    
    def _do_continue(self, opt):
        """Check if should continue."""
        # Count evaluations
        self.n_evals += opt.pop_size if hasattr(opt, 'pop_size') else len(opt.y)
        
        if self.n_evals >= self.max_evals:
            self.problem.logger.info(
                f"Optimization terminated: {self.n_evals} evaluations "
                f"reached (limit: {self.max_evals})"
            )
            return False
        
        return True

# Usage
termination = MaxEvaluationsTermination(
    problem=problem,
    max_evals=50000
)
```

### Option 3: Combine Multiple Criteria

Use `TerminationCollection` to combine custom and built-in criteria:

```python
from dmosopt.termination import TerminationCollection, MaximumGenerationTermination
from adaptive_hv_termination import AdaptiveHypervolumeProgressTermination

termination = TerminationCollection(
    problem,
    # Built-in: Maximum generations
    MaximumGenerationTermination(problem, n_max_gen=1000),
    
    # Advanced: Adaptive HV convergence
    AdaptiveHypervolumeProgressTermination(
        problem,
        ref_point=ref_point,
        hv_tol=1e-5
    ),
    
    # Custom: Your criterion
    MyCustomTermination(
        problem,
        diversity_threshold=0.01
    )
)
```

**Behavior:** Terminates when **any** criterion is met (logical OR).

---

## Common Configuration Patterns

### Pattern 1: Conservative (Avoid Early Termination)

```python
termination = create_adaptive_termination(
    problem=problem,
    n_max_gen=2000,
    strategy='conservative',
    obj_tol=1e-5,              # Stricter
    min_converged_fraction=0.9, # Need 90% converged
    hv_tol=1e-6                # Very strict HV
)
```

### Pattern 2: Aggressive (Fast Termination)

```python
termination = create_adaptive_termination(
    problem=problem,
    n_max_gen=1000,
    strategy='fast',
    obj_tol=1e-3,              # More lenient
    hv_tol=5e-5,               # Less strict
    min_converged_fraction=0.7 # Only 70% needed
)
```

### Pattern 3: Resource-Constrained

```python
from dmosopt.adaptive_termination import ResourceAwareTermination

termination = ResourceAwareTermination(
    problem=problem,
    max_time_seconds=3600,      # 1 hour time limit
    max_function_evals=50000,   # Or 50k evaluations
    target_quality_threshold=0.9 # Or quality target
)
```

### Pattern 4: High-Dimensional Only

```python
# For d > 20, use HV termination alone (most efficient)
from dmosopt.hv_termination import HypervolumeProgressTermination

termination = HypervolumeProgressTermination(
    problem=problem,
    ref_point=ref_point,
    hv_tol=2e-5,
    min_generations=30,
    n_last=20,
    nth_gen=3,  # Check frequently in high-d
    verbose=True
)
```

---

## Monitoring Progress

### Enable Verbose Output

```python
termination = create_adaptive_termination(
    problem=problem,
    strategy='comprehensive',
    verbose=True  # Enable for all criteria
)
```

### Access Termination State

```python
# After optimization
result = dmosopt.run(problem, algorithm, termination=termination)

# For CompositeAdaptiveTermination
if hasattr(termination, 'terminations'):
    for criterion in termination.terminations:
        print(f"Criterion: {criterion.__class__.__name__}")
        
        # Access specific state
        if hasattr(criterion, 'objective_states'):
            # Per-objective convergence
            converged = sum(s.converged for s in criterion.objective_states)
            print(f"  Converged objectives: {converged}/{len(criterion.objective_states)}")
        
        if hasattr(criterion, 'hv_history'):
            # Hypervolume history
            print(f"  Final HV: {criterion.hv_history[-1]:.6f}")
```

## Troubleshooting

### Problem: Optimization terminates too early

**Solutions:**
1. Increase `min_generations`
2. Tighten tolerances (`obj_tol`, `hv_tol`)
3. Use `strategy='conservative'`
4. Increase window sizes (`n_last`)

```python
termination = create_adaptive_termination(
    problem=problem,
    strategy='comprehensive',
    obj_tol=1e-5,              # Stricter
    hv_tol=1e-6,               # Stricter
    min_converged_fraction=0.9  # More objectives needed
)
```

### Problem: Optimization never terminates

**Solutions:**
1. Loosen tolerances
2. Decrease window sizes
3. Use `strategy='fast'`
4. Add maximum generation limit

```python
from dmosopt.termination import TerminationCollection, MaximumGenerationTermination

termination = TerminationCollection(
    problem,
    MaximumGenerationTermination(problem, n_max_gen=1000),  # Hard limit
    create_adaptive_termination(
        problem,
        strategy='fast',
        obj_tol=1e-3,  # More lenient
        hv_tol=1e-4    # More lenient
    )
)
```

### Problem: High computational cost per generation

**Solutions:**
1. Check HV less frequently (`nth_gen`)
2. Use `strategy='simple'` (HV only)
3. For very high-d, use HV termination alone

```python
# Check less frequently
termination = create_adaptive_termination(
    problem=problem,
    strategy='simple',
    nth_gen=10  # Check every 10 generations instead of 5
)
```

---

## Summary

### Quick Reference

| Use Case | Recommended Approach | Key Parameters |
|----------|---------------------|----------------|
| **Default / Production** | `strategy='comprehensive'` | Default params |
| **High-dimensional (d>20)** | `AdaptiveHypervolumeProgressTermination` | `hv_tol=2e-5` |
| **Fast / Testing** | `strategy='fast'` | Default params |
| **Conservative** | `strategy='conservative'` | `obj_tol=1e-5` |
| **Time-limited** | `ResourceAwareTermination` | `max_time_seconds` |
| **Custom needs** | Subclass `SlidingWindowTermination` | See examples |

### Best Practices

1. **Start with defaults:** Use `strategy='comprehensive'` for most problems
2. **Enable verbose mode initially:** Understand behavior before production runs
3. **Validate convergence:** Check final Pareto front quality after optimization
4. **Document configuration:** Save termination parameters with results
5. **Combine criteria:** Use `TerminationCollection` for robust termination
