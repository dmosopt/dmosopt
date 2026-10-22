# Surrogate gradient descent

dmosopt' [joint surrogate model](/guide/surrogates#joint-model) is differentiable in the parameters, so its gradient can move samples towards predicted feasibility before the expensive evaluation (see [Gressmann+2026](https://arxiv.org/abs/2603.20984) and [full code release](https://github.com/GazzolaLab/MultiObjectiveSurrogateOptimization)).

This guide walks through the hippocampal CA1 network optimization implemented in the [MiV-Simulator](https://github.com/GazzolaLab/MiV-Simulator). A runnable, self-contained version of the setup can be found [here](https://github.com/GazzolaLab/MiV-Simulator-Cases/tree/main/7-optimization).

## 1. The loop

Consider a CA1 network model with 8,711 cells and 12 populations, 146 synaptic weights to fit to meet 3 rate objectives and 57 band constraints. 

The feasible set in this high dimensional space is a very thin manifold, so the chance that crossover and mutation produce a feasible sample is small.

To make the optimization more effective, we can leverage the [joint transformer model](/guide/surrogates#joint-model) that can fit all objectives and every constraint at once. Since its predictions are differentiable
with respect to the parameters, we can then use the model to inform the NSGA-II search direction.

The loop becomes:

1. Evaluate a batch, fit the joint model on everything evaluated so far.
2. Run NSGA-II on the surrogate, as usual.
3. Before the resampled population is sent to the simulator, take each candidate and run a few hundred steps of gradient descent on the surrogate's predictions to push the predicted constraint probabilities towards one, and the predicted objectives towards the front.
4. Simulate the moved candidates.

dmosopt's `joint` training function exposes the two hooks for it:

- `model_cls` replaces the surrogate class, so the model can be given a projection method that uses its own gradients;
- `optimizer_wrapper` wraps dmosopt' NSGA-II optimizer class to project the samples down the surrogate gradient.


## 2. The losses

`JointFTTransformer.make_feasible` moves a batch of parameters down the gradient of a loss on the model's own predictions. `heads(x)` is the model's forward pass, returning the predicted objectives and the predicted probability that each constraint is satisfied. The loss has up to two terms. The objective term is a smooth stand-in for hypervolume against the evaluated front, so objectives are not traded away against the constraints:

```python{4,5}
def hypervolume(x):
    o, _ = heads(x)
    nadir = ops.max(ops.concatenate([o, y_norm], axis=0), axis=0)
    front = o / (nadir + 1e-8)
    return -ops.sum(ops.prod(ops.maximum(1.1 - front, 0.0), axis=-1))
```

The constraint term is a focal cross-entropy towards 1 on the predicted probabilities, so a candidate the model already believes feasible contributes little and the gradient concentrates on the constraints it predicts violated. `constraint_weights` scales each constraint's share of that term:

```python{4,5}
def infeasibility(x):
    _, c = heads(x)
    p = ops.clip(c, 1e-7, 1.0 - 1e-7)
    focal = -((1.0 - p) ** 2) * ops.log(p)
    return ops.mean(focal * weights_t)
```

The update is plain Adam on the parameters with the step expressed as a fraction of each parameter's range. The candidates are clipped to the bounds at every step and stop when the loss has plateaued.

## 3. Wrapping the optimizer

The projection runs on the population NSGA-II has after its last generation. dmosopt's optimizer exposes that population as `population_objectives`:

```python{3,18,28}
# project non-elite half
move = np.zeros(len(parameters), dtype=bool)
move[len(parameters) // 2 :] = True
if len(self._seen):
    from scipy.spatial.distance import cdist

    move |= cdist(parameters, self._seen).min(axis=1) <= 1e-12

original = parameters.copy()
model = self._surrogate._wrapped
info = {}
if move.any():
    moved, info = model.make_feasible(
        parameters[move],
        targets=self._targets,
        max_iterations=self._iterations,
        min_iterations=200,
        constraint_weights=self._constraint_weights,
        max_move=self._max_move,
    )
    parameters[move] = moved

# revert moves that the surrogate itself rates worse on every objective
reverted = np.zeros(len(parameters), dtype=bool)
if move.any():
    before = np.asarray(self._surrogate.evaluate(original[move]))
    after = np.asarray(self._surrogate.evaluate(parameters[move]))
    worse = np.all(after >= before, axis=1) & np.any(after > before, axis=1)
    idx = np.flatnonzero(move)[worse]
    parameters[idx] = original[idx]
    reverted[idx] = True
```

Notably,

- Only the lower performing half is projected.
- `make_feasible` optimizes a loss, not the objectives directly, so if the moved point is predicted worse on every objective, it is put back.

## 4. Switching it on

```python
dmosopt_params = {
    # ... problem definition ...
    "surrogate_custom_training": "dmosopt.model_transformer.joint",
    "surrogate_custom_training_kwargs": {
        "mode": "c+o",
        "objectives": True,
        "sensitivity": True,
        "feasibility_solving": True,
        "feasibility_after": 10,          # project after the last NSGA-II generation
        "feasibility_iterations": 1000,
        "feasibility_max_move": 0.1,
        "feasibility_constraint_weights": {52: 4, 45: 4, 51: 4},   # by column index
                                          # = penalize silent populations more strongly
    },
    "num_generations": 10,
    "population_size": 100,
    "resample_fraction": 1.0,
}
```

## Results on the CA1 model

The configuration evaluates 438 random initial samples before handing over to 10 surrogate epochs. 

Below, we pick the solution that satisfies the most constraints, 49 of 57, with every population active and on target:

![Objectives and constraints](/examples/ca1-row846-targets.svg)

The raster for every simulated cell over 1.5 s after a 3 s warm-up, grouped by population:

![Raster and sampled membrane potentials](/examples/ca1-row846-raster.svg)

