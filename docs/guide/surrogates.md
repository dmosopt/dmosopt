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

The joint model is a Transformer-based surrogate that can predict objectives and constraints simultaneously using the `JointFTTransformer` architecture from `dmosopt.custom_training`. Unlike Gaussian process surrogates which fit a separate model per objective, the joint model trains a single multi-task model over all outputs at once. This can be especially effective for problems with many objectives and constraints, or where objectives and constraints are correlated.

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
    "surrogate_custom_training": "dmosopt.custom_training.joint",
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

