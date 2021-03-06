
import os, sys, logging, datetime, gc, pprint
from functools import partial
import click
import numpy as np
from dmosopt.dmosopt import init_from_h5
from dmosopt.MOASMO import get_best

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
@click.option("--sort-key", required=False, type=str)
@click.option("--filter-objectives", required=False, type=str)
@click.option("--verbose", '-v', is_flag=True)
def main(file_path, opt_id, sort_key, filter_objectives, verbose):

    old_evals, param_names, is_int, lo_bounds, hi_bounds, objective_names, feature_names, problem_parameters, problem_ids = \
                  init_from_h5(file_path, None, opt_id, None)

    if problem_ids is None:
        problem_ids = [0]
    for problem_id in problem_ids:
        old_eval_xs = [e[0] for e in old_evals[problem_id]]
        old_eval_ys = [e[1] for e in old_evals[problem_id]]
        x = np.vstack(old_eval_xs)
        y = np.vstack(old_eval_ys)

        if filter_objectives is not None:
            filtered_objective_set = set(filter_objectives.split(','))
            filtered_objective_index = []
            for i, objective_name in enumerate(objective_names):
                if objective_name in filtered_objective_set:
                    filtered_objective_index.append(i)
            filtered_objective_names = [objective_names[i] for i in filtered_objective_index]
            filtered_y = y[filtered_objective_index]
            objective_names = filtered_objective_names
            y = filtered_y

        print(f"Found {x.shape[0]} results for id {problem_id}")
        
        best_x, best_y, _ = get_best(x, y, None, len(param_names), len(objective_names))
        prms = list(zip(param_names, list(best_x.T)))
        res = list(zip(objective_names, list(best_y.T)))
        prms_dict = dict(prms)
        res_dict = dict(res)
        n_res = next(iter(res_dict.values())).shape[0]
        if sort_key is None:
            for i in range(n_res):
                res_i = { k: res_dict[k][i] for k in res_dict }
                prms_i = { k: prms_dict[k][i] for k in prms_dict }
                print(f"Best eval {i} for id {problem_id}: {pprint.pformat(res_i)}@{prms_i}")
        else:
            sort_array = res_dict[sort_key]
            sorted_index = np.argsort(sort_array, kind='stable')
            for i, n in enumerate(sorted_index):
                res_n = { k: res_dict[k][n] for k in res_dict }
                prms_n = { k: prms_dict[k][n] for k in prms_dict }
                print(f"Best eval {i} so far for id {problem_id} / {sort_key}: {pprint.pformat(res_n)}@{prms_n}")
            
            


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda x: os.path.basename(x) == script_name, sys.argv)+1):])

