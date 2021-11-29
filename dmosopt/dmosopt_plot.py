
import sys, os, time, copy, logging, datetime, gc, pprint
from functools import partial
import click
import numpy as np
import itertools as it
import h5py, yaml
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from dmosopt.dmosopt import init_from_h5
from dmosopt.MOASMO import get_best
from dmosopt.visualization import save_figure

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
@click.option('--constraints/--no-constraints', default=True)
@click.option("--file-path", '-p', required=True, type=click.Path())
@click.option("--opt-id", required=True, type=str)
@click.option("--start-iter", default=0, type=int)
@click.option("--filter-objectives", required=False, type=str)
@click.option("--normalize-objectives", '-n', is_flag=True)
@click.option("--best", '-b', is_flag=True)
@click.option("--font-size", type=float, default=14)
@click.option("--fig-size", type=(float,float), default=(15,8))
@click.option("--fig-format", type=str, default='png')
@click.option("--verbose", '-v', is_flag=True)
def main(constraints, file_path, opt_id, start_iter, filter_objectives, normalize_objectives, best, font_size, fig_size, fig_format, verbose):

    max_epoch, old_evals, param_names, is_int, lo_bounds, hi_bounds, objective_names, feature_names, constraint_names, problem_parameters, problem_ids = \
                  init_from_h5(file_path, None, opt_id, None)

    if problem_ids is None:
        problem_ids = [0]
    for problem_id in problem_ids:
        old_eval_xs = [e.parameters for e in old_evals[problem_id]]
        old_eval_ys = [e.objectives for e in old_evals[problem_id]]
        old_eval_fs = None
        f = None
        if feature_names is not None:
            old_eval_fs = [e.features for e in old_evals[problem_id]]
            f = np.concatenate(old_eval_fs, axis=None)
        old_eval_cs = None
        c = None
        if constraint_names is not None:
            old_eval_cs = [e.constraints for e in old_evals[problem_id]]
            c = np.vstack(old_eval_cs)
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

        print(f'Found {x.shape[0]} results for id {problem_id}')
        sys.stdout.flush()

        if best:
            best_x, best_y, best_f, best_c, best_epochs, _ = get_best(x, y, f, c, len(param_names), len(objective_names), 
                                                                      feasible=constraints)

            prms = list(zip(param_names, list(best_x.T)))
            res = list(zip(objective_names, list(best_y.T)))
            prms_dict = dict(prms)
            res_dict = dict(res)
            constr_dict = None
            ftrs_dict = None
            
            if best_f is not None:
                ftrs = list(zip(feature_names, list(best_f.T)))
                ftrs_dict = dict(ftrs)
            
            if constraints and best_c is not None:
                constr = list(zip(constraint_names, list(best_c.T)))
                constr_dict = dict(constr)
            n_res = best_y.shape[0]

        else:
            prms = list(zip(param_names, list(x.T)))
            res = list(zip(objective_names, list(y.T)))
            prms_dict = dict(prms)
            res_dict = dict(res)
            constr_dict = None
            ftrs_dict = None
            
            if constraints and c is not None:
                constr = list(zip(constraint_names, list(c.T)))
                constr_dict = dict(constr)
            if f is not None:
                ftrs = list(zip(feature_names, list(best_f.T)))
                ftrs_dict = dict(ftrs)
            n_res = y.shape[0]

        n_rows = len(objective_names)
        n_row = 0
        mpl.rcParams['font.size'] = font_size
        fig = plt.figure(constrained_layout=True, figsize=fig_size)
        fig_spec = gs.GridSpec(ncols=1, nrows=n_rows, figure=fig)
        for i, objective_name in enumerate(objective_names):
            if len(res_dict[objective_name].shape) == 1:
                f_ax = fig.add_subplot(fig_spec[n_row, 0])
                f_ax.set_title(f"Objective {objective_name}")
                plt.plot(res_dict[objective_name][start_iter:], marker='o')
                n_row += 1
                    
        save_figure(opt_id, fig)
                
        plt.show()
            
