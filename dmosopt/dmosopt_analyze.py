import os, sys, logging, datetime, gc, pprint
from functools import partial
from collections import OrderedDict
import click
import numpy as np
from scipy.spatial import cKDTree
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
@click.option("--constraints/--no-constraints", default=True)
@click.option("--file-path", "-p", required=True, type=click.Path())
@click.option("--opt-id", required=True, type=str)
@click.option("--sort-key", required=False, type=str, multiple=True)
@click.option("--knn", required=False, type=int, default=0)
@click.option("--filter-objectives", required=False, type=str)
@click.option("--output-file", required=False, type=click.Path())
@click.option("--verbose", "-v", is_flag=True)
def main(
    constraints,
    file_path,
    opt_id,
    sort_key,
    knn,
    filter_objectives,
    output_file,
    verbose,
):
    (
        _,
        max_epoch,
        old_evals,
        param_names,
        is_int,
        lo_bounds,
        hi_bounds,
        objective_names,
        feature_names,
        constraint_names,
        problem_parameters,
        problem_ids,
    ) = init_from_h5(file_path, None, opt_id, None)

    if problem_ids is None:
        problem_ids = [0]
    for problem_id in problem_ids:
        old_eval_epochs = [e.epoch for e in old_evals[problem_id]]
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
        epochs = None
        if len(old_eval_epochs) > 0 and old_eval_epochs[0] is not None:
            epochs = np.concatenate(old_eval_epochs, axis=None)

        if filter_objectives is not None:
            filtered_objective_set = set(filter_objectives.split(","))
            filtered_objective_index = []
            for i, objective_name in enumerate(objective_names):
                if objective_name in filtered_objective_set:
                    filtered_objective_index.append(i)
            filtered_objective_names = [
                objective_names[i] for i in filtered_objective_index
            ]
            filtered_y = y[filtered_objective_index]
            objective_names = filtered_objective_names
            y = filtered_y

        print(f"Found {x.shape[0]} results for id {problem_id}")
        sys.stdout.flush()

        best_x, best_y, best_f, best_c, best_epoch, _ = get_best(
            x,
            y,
            f,
            c,
            len(param_names),
            len(objective_names),
            epochs=epochs,
            feasible=constraints,
        )

        print(f"Found {best_x.shape[0]} best results for id {problem_id}")
        sys.stdout.flush()
        prms = list(zip(param_names, list(best_x.T)))
        res = list(zip(objective_names, list(best_y.T)))
        prms_dict = dict(prms)
        res_dict = dict(res)
        constr_dict = None

        if best_c is not None:
            constr = list(zip(constraint_names, list(best_c.T)))
            constr_dict = dict(constr)

        output_dict = OrderedDict()
        n_res = best_y.shape[0]
        m = len(objective_names)
        if len(sort_key) == 0:
            nn = range(n_res)
            if knn > 0:
                points = np.zeros((n_res, m))
                for i in range(n_res):
                    res_i = np.asarray([res_dict[k][i] for k in objective_names])
                    points[i, :] = res_i

                for m_i in range(m):
                    if np.max(points[:, m_i]) > 0.0:
                        points[:, m_i] = points[:, m_i] / np.max(points[:, m_i])

                if points.shape[0] < knn:
                    knn = points.shape[0]

                tree = cKDTree(points)
                qp = np.zeros((m,))
                dnn, nn = tree.query(qp, k=knn)

                if isinstance(nn, int):
                    nn = [nn]
                else:
                    nn = nn[~np.isinf(dnn)]

            for i in nn:
                res_i = {k: res_dict[k][i] for k in objective_names}
                prms_i = {k: prms_dict[k][i] for k in param_names}
                output_dict[i] = [float(prms_dict[k][i]) for k in param_names]
                constr_label = ""
                if constr_dict is not None:
                    constr_i = {k: constr_dict[k][i] for k in constraint_names}
                    constr_label = f"constr: {constr_i}"
                ftrs_label = ""
                if best_f is not None:
                    ftrs_i = best_f[i]
                    ftrs_label = f"[{ftrs_i}]"
                epoch_label = ""
                if best_epoch is not None:
                    epoch_i = best_epoch[i]
                    epoch_label = f"E{epoch_i}"
                print(
                    f"Best eval {i} for id {problem_id}: {epoch_label} {pprint.pformat(res_i)}@{prms_i} {ftrs_label} {constr_label}"
                )
        else:
            sort_tuple = tuple((res_dict[k] for k in sort_key[::-1]))
            sorted_index = np.lexsort(sort_tuple)
            for n in sorted_index:
                res_n = {k: res_dict[k][n] for k in objective_names}
                prms_n = {k: prms_dict[k][n] for k in param_names}
                output_dict[n] = [float(prms_dict[k][n]) for k in param_names]
                constr_label = ""
                if constr_dict is not None:
                    constr_n = {k: constr_dict[k][n] for k in constraint_names}
                    constr_label = f"constr: {constr_n}"
                ftrs_label = ""
                if best_f is not None:
                    ftrs_n = best_f[n]
                    ftrs_label = f"[{ftrs_n}]"
                epoch_label = ""
                if best_epoch is not None:
                    epoch_n = best_epoch[n]
                    epoch_label = f"E{epoch_n}"
                print(
                    f"Best eval {n} for id {problem_id} / {sort_key}: {epoch_label} {pprint.pformat(res_n)}@{prms_n} {ftrs_label} {constr_label}"
                )

        if output_file is not None:
            with open(output_file, "w") as out:
                for k, v in output_dict.items():
                    out.write(f"{k}: {pprint.pformat(output_dict[k])}\n")


if __name__ == "__main__":
    main(
        args=sys.argv[
            (list_find(lambda x: os.path.basename(x) == script_name, sys.argv) + 1) :
        ]
    )
