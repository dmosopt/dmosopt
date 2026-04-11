# Motoneuron

This example demonstrates multi-objective optimization of a biophysical motoneuron model using the [joint surrogate model](/guide/surrogates#joint-model). The motoneuron model is derived from electrophysiological recordings of mouse embryonic stem cell-derived motoneurons ([Bhatt et al., J. Neurosci 2004](https://www.jneurosci.org/content/24/36/7848)).

The optimization targets five electrophysiological objectives (input resistance, membrane time constant, frequency-current relationship, spike amplitude, and ISI adaptation) subject to eight feasibility constraints. It uses the `JointFTTransformer` custom training surrogate which trains a single multi-task model over all objectives and constraints simultaneously.

::: info Requirements

- [NEURON](https://www.neuron.yale.edu/neuron/) simulator
- [Keras 3](https://keras.io/) with a backend of your choice (see [joint model docs](/guide/surrogates#joint-model))
- `click matplotlib mpi4py numpy pyyaml scipy`
- Compile the NMODL mechanisms: `cd examples/motoneuron/mechanisms && nrnivmodl`

:::

## Running the example

```bash
export KERAS_BACKEND=torch
mpirun -n 8 python example_dmosopt_motoneuron.py
```

## Source

<<< @/../examples/motoneuron/example_dmosopt_motoneuron.py
