# Optimizers

dmosopt currently implements four different multi-objective optimizers, briefly described below.

## CMAES

[Covariance-Matrix Adaptation Evolution Strategy](https://en.wikipedia.org/wiki/CMA-ES) where each generation is sampled from a multivariate normal distribution while progressively updating the covariance matrix to model variable dependencies. The first generation is sampled with default parameters, but each of the following uses data from the previous generation to determine its parameters. Specifically, the following generations will take, say, the top 25% of data points from the previous generation, and calculate the covariance matrix and mean of those data points. Then, it will generate the next set of data using that matrix and mean. Generally, the changing covariance matrix allows for a dynamic search space that will grow and shrink as needed.

## SMPSO

[Speed-constrained Multi-objective Particle Swarm Optimization](https://en.wikipedia.org/wiki/Particle_swarm_optimization) is inspired by the swarm behaviors seen in nature, particularly in flocks of birds. Each particle (data point) in a generation has its own velocity and inertia. Initially, these particles are spread randomly across the search space with a random speed and direction. From one generation to the next, the particles will move according to their velocity. In addition, the velocity of each particle will change from one generation to the next. Each particle’s velocity is updated differently since they are influenced by their own personal best as well as the overall best solution found by the group.

## NSGA2

[Non-dominated Sorting Genetic Algorithm 2](https://ieeexplore.ieee.org/document/996017) follows an "elitist principle". For each generation of data, it ranks the points into tiers. The best solutions are non-dominated, which means that if point X dominates point Y, there is no way to say X is worse than Y. In addition, at least one objective of X is better than that objective of Y. After creating these rankings, this algorithm uses crowd distance sorting. This essentially means that across the best solutions, they will pick out data that is more unique to keep. Lastly, the next generation of data will be made through crossover and mutation of the previous data, as well as keeping some of the best performers.

## AGE

[Adaptive Geometry Estimation](https://doi.org/10.1145/3321707.3321839) follows the same process as NSGA2. The main difference between the two algorithms is a different crowding distance formula. In other words, it has a different method of choosing which data to keep.

## Implement your own

Extending, modifying or implementing custom optimizers is straightforward. Define a class inheriting from the [MOEA](https://github.com/iraikov/dmosopt/blob/main/dmosopt/MOEA.py) base class and point to it in your optimizer configuration.