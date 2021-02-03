
import os, sys, logging, datetime, gc, pprint
from functools import partial
import click
import numpy as np
from dmosopt.dmosopt import init_from_h5
from dmosopt.MOASMO import get_best, onestep

script_name = os.path.basename(__file__)

def list_find(f, lst):
    """

    :param f:
    :param lst:
    :return:
    """
    i = 0
    for x in lst:
        if f(x):
            return i
        else:
            i = i + 1
    return None

@click.command()
@click.option("--file-path", '-p', required=True, type=click.Path())
@click.option("--opt-id", required=True, type=str)
@click.option("--resample-fraction", required=True, type=float)
@click.option("--population-size", required=True, type=int)
@click.option("--num-generations", required=True, type=int)
@click.option("--verbose", '-v', is_flag=True)
def main(file_path, opt_id, resample_fraction, population_size, num_generations, verbose):

    old_evals, param_names, is_int, lo_bounds, hi_bounds, objective_names, problem_parameters, problem_ids = \
                  init_from_h5(file_path, None, opt_id, None)

    logger = logging.getLogger(opt_id)
    logger.setLevel(logging.INFO)

    if problem_ids is None:
        problem_ids = [0]
    for problem_id in problem_ids:
        old_eval_xs = [e[0] for e in old_evals[problem_id]]
        old_eval_ys = [e[1] for e in old_evals[problem_id]]
        x = np.vstack(old_eval_xs)
        y = np.vstack(old_eval_ys)

        print(f"Found {x.shape[0]} results for id {problem_id}")

        n_dim = len(lo_bounds)
        n_objectives = len(objective_names)
        
        x_resample = onestep(n_dim, n_objectives,
                             np.asarray(lo_bounds), np.asarray(hi_bounds), resample_fraction,
                             x, y, pop=population_size, gen=num_generations,
                             logger=logger)

        for i, x in enumerate(x_resample):
            print(f"resampled coordinates {i}: {pprint.pformat(x)}")
            
            


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda x: os.path.basename(x) == script_name, sys.argv)+1):])

