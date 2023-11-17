# Welcome!

*dmosopt* aims to simplify distributed multi-objective surrogate-assisted optimization while staying out of your way.


## Overview

At its core, dmosopt implements a unified, generic optimization API via the `dmosopt.run(dopt_params=...)` function. The `dopt_params` argument takes a dictionary that allows users to define arbitrary multi-objective optimization problems, including the objectives, sampling space, constraints and optimization strategies. At the most basic level, all dmosopt scripts will take the following form:

```python
from dmosopt import dmosopt

dmosopt.run({
    "opt_id": "example_optimization",
    ...  #  configuation options
})
```

The `opt_id` serves a unique namespace for the resulting output file that captures the best solutions as well as various meta data about the optimization progress. 

The output file format is [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format), supporting high-performance, parallel data access.

dmosopt relies on MPI and more specifically [distwq](https://github.com/iraikov/distwq) to support scaling and parallelization.


## Where to go from here

::: info Installation

To get started, run `pip install dmosopt`.

[Learn more](./installation)

:::

::: info [Continue with the Guide](./run)

To dive into available configuration options and features.

:::

::: info [Check out the examples](../examples/zdt1)

Explore examples that demonstrate dmosopt' optimization capabilities.

:::

::: info [Consult the Reference](../reference/index)

Describes available APIs in full detail.

:::
