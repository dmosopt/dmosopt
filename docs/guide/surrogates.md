# Surrogates

A surrogate model can make the optimization more efficient by building an approximate model of the problem that can be queried at a faster rate. The most promising points according to the surrogate model can then be evaluated at the actual problem. dmosopt implements various strategies listed below:

<ul>
    <li v-for="i in ['gpr', 'egp', 'megp', 'mdgp', 'mdspp','vgp', 'svgp', 'spv', 'siv', 'crv']">
        {{ i }} - <a href="https://github.com/iraikov/dmosopt/blob/main/dmosopt/model.py" target="_blank">
            {{ i.toUpperCase() }}_Matern
        </a>
    </li>
</ul>

You may also point to your custom implementations by specifying a Python import path.

## Joint model

The joint model is a Transformer-based surrogate that can predict objectives and constraints simultaneously using the `JointFTTransformer` architecture from `dmosopt.model_transformer`. Unlike Gaussian process surrogates which fit a separate model per objective, the joint model trains a single multi-task model over all outputs at once. This can be especially effective for problems with many objectives and constraints, or where objectives and constraints are correlated.

### Installation

The joint model requires [Keras 3](https://keras.io/) and a deep-learning backend of your choice (JAX, PyTorch, TensorFlow). Set the backend via the environment variable before running your script:

```bash
export KERAS_BACKEND=jax   # or torch, tensorflow
```

### Configuration

To use the joint model, set `surrogate_custom_training` to point to the built-in training function:

```python
dmosopt_params = {
    # ... problem definition ...
    "surrogate_custom_training": "dmosopt.model_transformer.joint",
    "surrogate_custom_training_kwargs": {
        "mode": "c+o",  
        "epochs": "auto", 
    },
}
```

The `mode` parameter controls which outputs are modelled jointly:

| Mode | Description |
|------|-------------|
| `"c+o"` | Train on both constraints and objectives (recommended) |
| `"o"` | Objectives only |
| `"c"` | Constraints only |

When `epochs` is set to `"auto"`, the model uses cross-validation to determine the optimal number of training epochs, which is recommended for most use cases.

### Sensitivity-guided optimization

The joint model additionally provides gradient-based parameter sensitivity analysis. When sensitivity is enabled (the default), the optimizer uses the estimated sensitivities to adapt crossover and mutation rates per parameter, focusing search effort on the most influential parameters.

### Example

See the [Motoneuron example](/examples/motoneuron) for a complete worked example using the joint model to optimize a neuron model with 5 objectives and 8 constraints.

### Gradient-based feasibility solving

Because the joint model is differentiable in the parameters, its predictions can also be used to inform the optimizer via `feasibility_solving`:

```python
dmosopt_params = {
    # ...
    "surrogate_custom_training": "dmosopt.model_transformer.joint",
    "surrogate_custom_training_kwargs": {
        "mode": "c+o",
        "objectives": True,
        "feasibility_solving": True,
        "feasibility_constraint_weights": {0: 4, 4: 4},
    },
}
```

| Option | Default | Description |
|---|---|---|
| `feasibility_solving` | `False` | Project the population before it is evaluated |
| `feasibility_targets` | `"objective constraint"` | Loss terms: a hypervolume stand-in on the predicted objectives, focal cross-entropy on the predicted constraint probabilities, or both |
| `feasibility_iterations` | `1000` | Maximum descent steps per projection |
| `feasibility_after` | `10` | NSGA-II generations before the projection runs |
| `feasibility_constraint_weights` | `None` | One weight per constraint, or `{column: weight}` with the rest at 1, scaling each constraint's share of the loss |
| `feasibility_max_move` | `0.1` | Adam trust region, as a fraction of each parameter's range |

See the how-to guide [Surrogate gradient descent](/examples/howto/surrogate-gradient-descent) for a full walk through of the idea, the code, and its results on a 146-parameter network model with 57 constraints.
