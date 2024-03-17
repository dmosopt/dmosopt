# Sampling

dmosopt implements various sampling strategies listed below:

<ul>
    <li v-for="i in ['glp', 'slh', 'lh', 'mc', 'sobol']">
        {{ i }} - <a href="https://github.com/iraikov/dmosopt/blob/main/dmosopt/sampling.py" target="_blank">
            {{ i }}()
        </a>
    </li>
</ul>

You may also point to your custom implementations by specifying a Python import path.

## Dynamic sampling

By default, the number of samples is pre-determined via the `n_initial` parameter. However, dmosopt supports dynamic sampling strategies that generate samples until custom criteria are met. To implement a dynamic sampling strategy, you can point `dynamic_initial_sampling` to a callable object (with optional parameters `dynamic_initial_sampling_kwargs`). The callable receives the filepath, iteration count, the sampler options, the evaluated samples up to this point, as well as an unevaluated next batch of samples. It must return a new set of samples or `None` to end the dynamic sampling process. For example:

```python
from dmosopt.datatypes import EvalEntry

def dynamic_sampling(
    file_path: str,
    interation: int,
    evaluated_samples: list[EvalEntry], 
    next_samples: list, 
    sampler: dict[], # contains n_initial, maxiter, method, param_names, xlb, xub
    **kwargs
):
    done = ... # decide if sampling is complete based on `evaluated_samples`

    if done:
        # no more samples, sampling will be complete using `evaluated_samples`
        return

    # return the next set of samples 
    # (can be the unmodified `next_samples` or some custom set of samples)
    return next_samples
```