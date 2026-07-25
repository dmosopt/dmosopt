"""
MPI distributed-evaluation test.

Not collected by a plain `pytest` run (see the `--ignore=tests/mpi` flag in
CI): dmosopt/distwq pick their controller/worker role from `MPI.COMM_WORLD`
at import time, so this must be launched directly under `mpirun`/`mpiexec`
with more than one rank, e.g.:

    mpirun -n 4 python tests/mpi/test_mpi_distributed.py

Rank 0 becomes the distwq controller and asserts on the result; the
remaining ranks become workers that evaluate `obj_fun` and return None from
`dmosopt.run`.
"""

import logging
import numpy as np
from mpi4py import MPI
from dmosopt import dmosopt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def sphere_2obj(x):
    f = np.zeros(2)
    f[0] = np.sum(x**2)
    f[1] = np.sum((x - 1.0) ** 2)
    return f


def obj_fun(pp):
    x = np.asarray([pp[k] for k in sorted(pp)])
    return sphere_2obj(x)


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    assert size >= 2, (
        f"expected to be launched under mpirun with >1 rank, got size={size}"
    )

    space = {f"x{i}": [0.0, 1.0] for i in range(4)}
    dmosopt_params = {
        "opt_id": "mpi_smoke",
        "obj_fun_name": "test_mpi_distributed.obj_fun",
        "problem_parameters": {},
        "space": space,
        "objective_names": ["f1", "f2"],
        "population_size": 8,
        "num_generations": 2,
        "initial_maxiter": 2,
        "optimizer_name": "nsga2",
        "surrogate_method_name": "gpr",
        "termination_conditions": False,
        "n_initial": 4,
        "n_epochs": 1,
        "save": False,
    }

    best = dmosopt.run(dmosopt_params, verbose=True)

    # Only the controller rank (rank 0) gets a non-None result back.
    if best is not None:
        bestx, besty = best
        n_workers_used = comm.Get_size() - 1
        logger.info(f"MPI world size: {size} (workers: {n_workers_used})")
        logger.info(f"Best solutions: {besty}")
        assert n_workers_used >= 1, "no MPI worker ranks participated"
        assert len(bestx) > 0 and len(besty) > 0, "no evaluations recorded"
