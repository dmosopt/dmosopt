"""
MPI test of the time limit given to dmosopt.run.

Like test_mpi_distributed.py, this is not collected by a plain `pytest`
run and must be launched under `mpirun`/`mpiexec` with more than one
rank, e.g.:

    mpirun -n 4 python tests/mpi/test_mpi_time_limit.py

The run asks for far more epochs than can finish within the time limit,
and each evaluation takes a fixed time.  Once the limit has passed, the
optimizer submits no further evaluations, collects the running ones,
and returns the best solutions found so far.  Rank 0 checks that the run
returned soon after the limit, with some evaluations recorded.
"""

import logging
import time

import numpy as np
from mpi4py import MPI

from dmosopt import dmosopt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TIME_LIMIT = 3.0
EVAL_TIME = 0.1
N_EPOCHS = 1000


def obj_fun(pp):
    time.sleep(EVAL_TIME)
    x = np.asarray([pp[k] for k in sorted(pp)])
    return np.array([np.sum(x**2), np.sum((x - 1.0) ** 2)])


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    assert size >= 2, (
        f"expected to be launched under mpirun with >1 rank, got size={size}"
    )

    dmosopt_params = {
        "opt_id": "mpi_time_limit",
        "obj_fun_name": "test_mpi_time_limit.obj_fun",
        "problem_parameters": {},
        "space": {f"x{i}": [0.0, 1.0] for i in range(4)},
        "objective_names": ["f1", "f2"],
        "population_size": 8,
        "num_generations": 2,
        "initial_maxiter": 2,
        "optimizer_name": "nsga2",
        "surrogate_method_name": "gpr",
        "termination_conditions": False,
        "n_initial": 4,
        "n_epochs": N_EPOCHS,
        "save": False,
    }

    start = time.time()
    best = dmosopt.run(dmosopt_params, time_limit=TIME_LIMIT, verbose=True)
    elapsed = time.time() - start

    # Only the controller rank (rank 0) gets a non-None result back.
    if best is not None:
        bestx, besty = best
        logger.info(f"Time-limited run returned after {elapsed:.1f} s")
        assert len(bestx) > 0 and len(besty) > 0, "no evaluations recorded"
        # The run may finish the evaluations that were running when the
        # limit passed, plus one surrogate fit, but no further epochs.
        assert elapsed < TIME_LIMIT + 60.0, (
            f"run did not stop after the time limit: {elapsed:.1f} s"
        )
