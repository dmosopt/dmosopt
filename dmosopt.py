import os, sys, importlib, logging, pprint, copy
from functools import partial
from collections import namedtuple
import numpy as np  
import distwq
import MOASMO as opt


try:
    import h5py
except ImportError as e:
    logger.warning('dmosopt: unable to import h5py: %s' % str(e))

sopt_dict = {}

ProblemSpec = namedtuple('ProblemSpec',
                           ['bound1',
                            'bound2',
                            'is_integer',
                            'n_objectives'
                            ])

class OptProblem():

    def __init__(self, spec, eval_fun):

        self.dim = len(spec.bound1)
        self.lb = spec.bound1
        self.ub = spec.bound2
        self.int_var = spec.is_integer
        self.eval_fun = eval_fun
        self.n_objectives = spec.n_objectives
        
    def eval(self, x):
        return self.eval_fun(x)

    
class OptStrategy():
    def __init__(self, prob, n_initial=10, resample_fraction=0.25):
        xinit = opt.xinit(n_initial, prob.dim, prob.n_objectives, prob.lb, prob.ub)
        assert(xinit.shape[1] == prob.dim)
        self.prob = prob
        self.x = None
        self.y = None #np.zeros((np.shape[0], prob.n_objectives))
        self.reqs = [ xinit[i,:] for i in range(xinit.shape[0]) ]
        self.completed = []
        self.resample_fraction = resample_fraction
        
    def get_next_x(self):
        result = None
        if len(self.reqs) > 0:
            result = self.reqs.pop(0)
        return result

    def complete_x(self, x, y):
        assert(x.shape[0] == self.prob.dim)
        assert(y.shape[0] == self.prob.n_objectives)
        self.completed.append((x,y))
    
    def step(self):
        if len(self.completed) > 0:
            x_completed = [x[0] for x in self.completed]
            y_completed = [x[1] for x in self.completed]

            x_completed = np.vstack(x_completed)
            y_completed = np.vstack(y_completed)

            assert(x_completed.shape[1] == self.prob.dim)
            assert(y_completed.shape[1] == self.prob.n_objectives)
            if self.x is None:
                self.x = x_completed
                self.y = y_completed
            else:
                self.x = np.vstack((self.x, x_completed))
                self.y = np.vstack((self.y, y_completed))
        x_resample = opt.onestep(self.prob.dim, self.prob.n_objectives,
                                 self.prob.lb, self.prob.ub, self.resample_fraction,
                                 self.x, self.y)
        for i in range(x_resample.shape[0]):
            self.reqs.append(x_resample[i,:])
        
    def get_best_evals(self):
        return opt.get_best(self.x, self.y, self.prob.dim, self.prob.n_objectives)

    def get_evals(self):
        return (self.x, self.y)

class DistOptimizer():
    def __init__(
        self,
        opt_id,
        obj_fun,
        n_objectives=None,
        n_initial=10,
        verbose=False,
        reduce_fun=None,
        problem_ids=None,
        problem_parameters=None,
        space=None,
        n_iter=100,
        nprocs_per_worker=1,
        save_iter=10,
        file_path=None,
        save=False,
        **kwargs
    ):
        """
        `Creates an optimizer based on the MO-ASMO optimizer. Supports
        distributed optimization runs via mpi4py.

        :param set problem_ids (optional): Set of problem ids.
        For solving sets of related problems with the same set of parameters.
        If this parameter is not None, it is expected that the objective function 
        will return a dictionary of the form { problem_id: value }
        :param dict problem_parameters: Problem parameters.
        All hyperparameters and their values for the objective
        function, including those not being optimized over. E.g: ``{'beta': 0.44}``.
        Can be an empty dict.
        Can include hyperparameters being optimized over, but does not need to.
        If a hyperparameter is specified in both 'problem_parameters' and 'space', its value
        in 'problem_parameters' will be overridden.
        :param dict space: Hyperparameters to optimize over.
        Entries should be of the form:
        ``parameter: (Low_Bound, High_Bound)`` e.g:
        ``{'alpha': (0.65, 0.85), 'gamma': (1, 8)}``. If both bounds for a
        parameter are Ints, then only integers within the (inclusive) range
        will be sampled and tested.
        :param func obj_fun: function to minimize.
        Must take as argument every parameter specified in
        both 'problem_parameters' and 'space',  and return the result as float.
        If you want to minimize instead,
        simply negate the result in the objective function before returning it.
        :param int n_iter: (optional) Number of times to sample and test params.
        :param int save_iter: (optional) How often to save progress.
        :param str file_path: (optional) File name for restoring and/or saving results and settings.
        :param bool save: (optional) Save settings and progress periodically.
        """

        self.opt_id = opt_id
        self.verbose = verbose

        self.logger = logging.getLogger(opt_id)
        if self.verbose:
            self.logger.setLevel(logging.INFO)

        # Verify inputs
        if file_path is None:
            if problem_parameters is None or space is None:
                raise ValueError(
                    "You must specify at least file name `file_path` or problem "
                    "parameters `problem_parameters` along with a hyperparameter space `space`."
                )
            if save:
                raise ValueError(
                    "If you want to save you must specify a file name `file_path`."
                )
        else:
            if not os.path.isfile(file_path):
                if problem_parameters is None or space is None:
                    raise FileNotFoundError(file_path)

        dim = 0
        param_names, is_int, lo_bounds, hi_bounds = [], [], [], []
        if space is not None:
            dim = len(space)
            for parm, conf in space.items():
                param_names.append(parm)
                lo, hi = conf
                is_int.append(type(lo) == int and type(hi) == int)
                lo_bounds.append(lo)
                hi_bounds.append(hi)
        old_evals = {}
        if file_path is not None:
            if os.path.isfile(file_path):
                old_evals, param_names, is_int, lo_bounds, hi_bounds, n_objectives, problem_parameters, problem_ids = \
                  init_from_h5(file_path, param_names, opt_id, self.logger)

        problem_spec = ProblemSpec(bound1=np.asarray(lo_bounds), bound2=np.asarray(hi_bounds),
                                   is_integer=is_int, n_objectives=n_objectives)
        self.problem_spec = problem_spec
        
        has_problem_ids = (problem_ids is not None)
        if not has_problem_ids:
            problem_ids = set([0])
        
        self.problem_parameters, self.param_names = problem_parameters, param_names
        self.is_int = is_int
        self.file_path, self.save = file_path, save

        self.n_iter = n_iter
        self.save_iter = save_iter

        if has_problem_ids:
            self.eval_fun = partial(eval_obj_fun_mp, obj_fun, self.problem_parameters, self.param_names, self.is_int, problem_ids)
        else:
            self.eval_fun = partial(eval_obj_fun_sp, obj_fun, self.problem_parameters, self.param_names, self.is_int, 0)
            
        self.reduce_fun = reduce_fun
        
        self.evals = { problem_id: {} for problem_id in problem_ids }

        self.has_problem_ids = has_problem_ids
        self.problem_ids = problem_ids

        optimizer_dict = {}
        for problem_id in problem_ids:
            opt_prob = OptProblem(problem_spec, self.eval_fun)
            opt_strategy = OptStrategy(opt_prob, n_initial)
            if problem_id in old_evals:
                old_eval_xs = [old_eval[0] for old_eval in old_evals[problem_id]]
                old_eval_ys = [old_eval[1] for old_eval in old_evals[problem_id]]
                opt_strategy.x = np.vstack(old_eval_xs)
                opt_strategy.y = np.vstack(old_eval_ys)
            optimizer_dict[problem_id] = opt_strategy
        self.optimizer_dict = optimizer_dict

    def save_evals(self):
        """Store results of finished evals to file; print best eval"""
        finished_evals = { problem_id: self.optimizer_dict[problem_id].get_evals()
                           for problem_id in self.problem_ids }
        save_to_h5(self.opt_id, self.problem_ids, self.has_problem_ids,
                   self.param_names, self.problem_spec, finished_evals, 
                   self.problem_parameters, 
                   self.file_path, self.logger)

    def get_best(self):
        best_results = {}
        for problem_id in self.problem_ids:
            best_x, best_y = self.optimizer_dict[problem_id].get_best_evals()
            prms = list(zip(self.param_names, list(best_x)))
            best_results[problem_id] = (prms, best_y)
        if self.has_problem_ids:
            return best_results
        else:
            return best_results[problem_id]
        
    def print_best(self):
        best_results = self.get_best()
        if self.has_problem_ids:
            for problem_id in self.problem_ids:
                prms, res = best_results[problem_id]
                n_res = res.shape[0]
                prms_dict = dict(prms)
                for i in range(n_res):
                    res_i = res[i,:]
                    prms_i = { k: prms_dict[i,:] for k in prms_dict }
                    self.logger.info(f"Best eval {i} so far for id {problem_id}: {res_i}@{prms_i}")
        else:
            prms, res = best_results
            n_res = res.shape[0]
            prms_dict = dict(prms)
            for i in range(n_res):
                res_i = res[i,:]
                prms_i = { k: prms_dict[k][i] for k in prms_dict }
                self.logger.info(f"Best eval {i} so far: {res_i}@{prms_i}")
            

def h5_get_group (h, groupname):
    if groupname in h.keys():
        g = h[groupname]
    else:
        g = h.create_group(groupname)
    return g

def h5_get_dataset (g, dsetname, **kwargs):
    if dsetname in g.keys():
        dset = g[dsetname]
    else:
        dset = g.create_dataset(dsetname, (0,), **kwargs)
    return dset

def h5_concat_dataset(dset, data):
    dsize = dset.shape[0]
    newshape = (dsize+len(data),)
    dset.resize(newshape)
    dset[dsize:] = data
    return dset

def h5_init_types(f, opt_id, param_names, problem_parameters, spec):
    
    opt_grp = h5_get_group(f, opt_id)

    param_keys = set(param_names)
    param_keys.update(problem_parameters.keys())
    # create an HDF5 enumerated type for the parameter label
    param_mapping = { name: idx for (idx, name) in
                      enumerate(param_keys) }

    dt = h5py.enum_dtype(param_mapping, basetype=np.uint16)
    opt_grp['parameter_enum'] = dt

    dt = np.dtype([("parameter", opt_grp['parameter_enum']),
                   ("value", np.float32)])
    opt_grp['problem_parameters_type'] = dt

    dset = h5_get_dataset(opt_grp, 'problem_parameters', maxshape=(len(param_mapping),),
                          dtype=opt_grp['problem_parameters_type'].dtype)
    dset.resize((len(param_mapping),))
    a = np.zeros(len(param_mapping), dtype=opt_grp['problem_parameters_type'].dtype)
    idx = 0
    for idx, (parm, val) in enumerate(problem_parameters.items()):
        a[idx]["parameter"] = param_mapping[parm]
        a[idx]["value"] = val
    dset[:] = a
    
    dt = np.dtype([("parameter", opt_grp['parameter_enum']),
                   ("is_integer", np.bool),
                   ("lower", np.float32),
                   ("upper", np.float32)])
    opt_grp['parameter_spec_type'] = dt

    is_integer = np.asarray(spec.is_integer, dtype=np.bool)
    upper = np.asarray(spec.bound2, dtype=np.float32)
    lower = np.asarray(spec.bound1, dtype=np.float32)

    dset = h5_get_dataset(opt_grp, 'parameter_spec', maxshape=(len(param_names),),
                          dtype=opt_grp['parameter_spec_type'].dtype)
    dset.resize((len(param_names),))
    a = np.zeros(len(param_names), dtype=opt_grp['parameter_spec_type'].dtype)
    for idx, (parm, is_int, hi, lo) in enumerate(zip(param_names, is_integer, upper, lower)):
        a[idx]["parameter"] = param_mapping[parm]
        a[idx]["is_integer"] = is_int
        a[idx]["lower"] = lo
        a[idx]["upper"] = hi
    dset[:] = a
    
def h5_load_raw(input_file, opt_id):
    ## N is number of trials
    ## M is number of hyperparameters
    f = h5py.File(input_file, 'r')
    opt_grp = h5_get_group(f, opt_id)

    n_objectives = int(opt_grp["n_objectives"][()])

    parameter_enum_dict = h5py.check_enum_dtype(opt_grp['parameter_enum'].dtype)
    parameters_idx_dict = { parm: idx for parm, idx in parameter_enum_dict.items() }
    parameters_name_dict = { idx: parm for parm, idx in parameters_idx_dict.items() }
    
    problem_parameters = { parameters_name_dict[idx]: val
                           for idx, val in opt_grp['problem_parameters'] }
    parameter_spec_dict = { parameters_name_dict[spec[0]]: tuple(spec)[1:] 
                            for spec in iter(opt_grp['parameter_spec']) }

    problem_ids = None
    if 'problem_ids' in opt_grp:
        problem_ids = set(opt_grp['problem_ids'])
    
    M = len(parameter_spec_dict)
    P = n_objectives
    raw_results = {}
    if problem_ids is not None:
        for problem_id in problem_ids:
            raw_results[problem_id] = opt_grp['%d' % problem_id]['results'][:].reshape((-1,M+P)) # np.array of shape [N, M+P]
    else:
        raw_results[0] = opt_grp['%d' % 0]['results'][:].reshape((-1,M+P)) # np.array of shape [N, M+P]
    f.close()
    
    param_names = []
    is_integer = []
    lower = []
    upper = []
    for parm, spec in parameter_spec_dict.items():
        param_names.append(parm)
        is_int, lo, hi = spec
        is_integer.append(is_int)
        lower.append(lo)
        upper.append(hi)
        

    raw_spec = (is_integer, lower, upper)
    info = { 'n_objectives': n_objectives,
             'params': param_names,                            
             'problem_parameters': problem_parameters,
             'problem_ids': problem_ids }
    
    return raw_spec, raw_results, info

def h5_load_all(file_path, opt_id):
    """
    Loads an HDF5 file containing
    (spec, results, info) where
      results: np.array of shape [N, M+1] where
        N is number of trials
        M is number of hyperparameters
        results[:, 0] is result/loss
        results[:, 1:] is [param1, param2, ...]
      spec: (is_integer, lower, upper)
        where each element is list of length M
      info: dict with keys
        params, problem
    Assumes the structure is located in group /{opt_id}
    Returns
    (dlib.function_spec, [dlib.function_eval], dict, prev_best)
      where prev_best: np.array[result, param1, param2, ...]
    """
    raw_spec, raw_problem_results, info = h5_load_raw(file_path, opt_id)
    is_integer, lo_bounds, hi_bounds = raw_spec
    n_objectives = info['n_objectives']
    spec = ProblemSpec(bound1=lo_bounds, bound2=hi_bounds, is_integer=is_integer, n_objectives=n_objectives)
    evals = { problem_id: [] for problem_id in raw_problem_results }
    for problem_id in raw_problem_results:
        raw_results = raw_problem_results[problem_id]
        for raw_result in raw_results:
            y = raw_result[:n_objectives]
            x = raw_result[n_objectives:]
            evals[problem_id].append((x, y))
    return raw_spec, spec, evals, info
    
def init_from_h5(file_path, param_names, opt_id, logger):        
    # Load progress and settings from file, then compare each
    # restored setting with settings specified by args (if any)
    old_raw_spec, old_spec, old_evals, info = h5_load_all(file_path, opt_id)
    saved_params = info['params']
    for problem_id in old_evals:
        n_old_evals = len(old_evals[problem_id])
        logger.info(f"Restored {n_old_evals} trials for problem {problem_id}")
    if (param_names is not None) and param_names != saved_params:
        # Switching params being optimized over would throw off the optimizer.
        # Must use restore params from specified
        logger.warning(
            f"Saved params {saved_params} differ from currently specified "
            f"{param_names}. Using saved."
            )
    params = saved_params
    raw_spec = old_raw_spec
    is_int, lo_bounds, hi_bounds = raw_spec
    if len(params) != len(is_int):
        raise ValueError(
            f"Params {params} and spec {raw_spec} are of different length"
            )
    problem_parameters = info['problem_parameters']
    n_objectives = info['n_objectives']
    problem_ids = info['problem_ids'] if 'problem_ids' in info else None

    return old_evals, params, is_int, lo_bounds, hi_bounds, n_objectives, problem_parameters, problem_ids

def save_to_h5(opt_id, problem_ids, has_problem_ids, param_names, spec, evals, problem_parameters, fpath, logger):
    """
    Save progress and settings to an HDF5 file 'fpath'.
    """

    f = h5py.File(fpath, "a")
    if opt_id not in f.keys():
        h5_init_types(f, opt_id, param_names, problem_parameters, spec)
        opt_grp = h5_get_group(f, opt_id)
        opt_grp['n_objectives'] = spec.n_objectives
        if has_problem_ids:
            opt_grp['problem_ids'] = np.asarray(list(problem_ids), dtype=np.int32)

    opt_grp = h5_get_group(f, opt_id)
    M = len(param_names)
    P = spec.n_objectives
    for problem_id in problem_ids:
        prob_evals_x, prob_evals_y = evals[problem_id]
        prob_evals_x = prob_evals_x.reshape((-1, M))
        prob_evals_y = prob_evals_y.reshape((-1, P))
        opt_prob = h5_get_group(opt_grp, '%d' % problem_id)
        dset = h5_get_dataset(opt_prob, 'results', maxshape=(None,),
                              dtype=np.float32) 
        old_size = int(dset.shape[0] / (M+P))
        raw_results = np.zeros((len(prob_evals_x)-old_size, M+P))
        for i in range(raw_results.shape[0]):
            x = prob_evals_x[i + old_size]
            y = prob_evals_y[i + old_size]
            raw_results[i][:P] = y
            raw_results[i][P:] = x
        logger.info(f"Saving {raw_results.shape[0]} trials for problem id %d to {fpath}." % problem_id)
        h5_concat_dataset(opt_prob['results'], raw_results.ravel())
    
    f.close()

    
def eval_obj_fun_sp(obj_fun, pp, space_params, is_int, problem_id, space_vals):
    """
    Objective function evaluation (single problem).
    """
    
    this_space_vals = space_vals[problem_id]
    for j, key in enumerate(space_params):
        pp[key] = int(this_space_vals[j]) if is_int[j] else this_space_vals[j]

    
    result = obj_fun(pp)
    return { problem_id: result }


def eval_obj_fun_mp(obj_fun, pp, space_params, is_int, problem_ids, space_vals):
    """
    Objective function evaluation (multiple problems).
    """

    mpp = {}
    for problem_id in problem_ids:
        this_pp = copy.deepcopy(pp)
        this_space_vals = space_vals[problem_id]
        for j, key in enumerate(space_params):
            this_pp[key] = int(this_space_vals[j]) if is_int[j] else this_space_vals[j]
        mpp[problem_id] = this_pp

    result_dict = obj_fun(mpp)
    return result_dict


def sopt_init(sopt_params, worker=None, verbose=False):
    objfun = None
    objfun_module = sopt_params.get('obj_fun_module', '__main__')
    objfun_name = sopt_params.get('obj_fun_name', None)
    if distwq.is_worker:
        if objfun_name is not None:
            if objfun_module not in sys.modules:
                importlib.import_module(objfun_module)
                
            objfun = eval(objfun_name, sys.modules[objfun_module].__dict__)
        else:
            objfun_init_module = sopt_params.get('obj_fun_init_module', '__main__')
            objfun_init_name = sopt_params.get('obj_fun_init_name', None)
            objfun_init_args = sopt_params.get('obj_fun_init_args', None)
            if objfun_init_name is None:
                raise RuntimeError("dmosopt.soptinit: objfun is not provided")
            if objfun_init_module not in sys.modules:
                importlib.import_module(objfun_init_module)
            objfun_init = eval(objfun_init_name, sys.modules[objfun_init_module].__dict__)
            objfun = objfun_init(**objfun_init_args, worker=worker)
            
    sopt_params['obj_fun'] = objfun
    reducefun_module = sopt_params.get('reduce_fun_module', '__main__')
    reducefun_name = sopt_params.get('reduce_fun_name', None)
    if reducefun_module not in sys.modules:
        importlib.import_module(reducefun_module)
    if reducefun_name is not None:
        reducefun = eval(reducefun_name, sys.modules[reducefun_module].__dict__)
        sopt_params['reduce_fun'] = reducefun        
    sopt = DistOptimizer(**sopt_params, verbose=verbose)
    sopt_dict[sopt.opt_id] = sopt
    return sopt


def sopt_ctrl(controller, sopt_params, verbose=False):
    """Controller for distributed surrogate optimization."""
    logger = logging.getLogger(sopt_params['opt_id'])
    if verbose:
        logger.setLevel(logging.INFO)
    sopt = sopt_init(sopt_params)
    logger.info("Optimizing for %d iterations..." % sopt.n_iter)
    iter_count = 0
    task_ids = []
    n_tasks = 0
    next_iter = False
    while (iter_count < sopt.n_iter):
        controller.recv()
        
        if (iter_count > 0) and sopt.save and (iter_count % sopt.save_iter == 0):
            sopt.save_evals()

        if len(task_ids) > 0:
            task_id, res = controller.get_next_result()

            logger.info(f"task_id {task_id}: res = {res}")
            
            if sopt.reduce_fun is None:
                rres = res
            else:
                rres = sopt.reduce_fun(res)

            for problem_id in rres:
                eval_x = sopt.evals[problem_id][task_id]
                sopt.optimizer_dict[problem_id].complete_x(eval_x, rres[problem_id])
                logger.info("problem id %d: optimization iteration %d: parameter coordinates %s: %s" % (problem_id, iter_count, str(eval_x), str(rres[problem_id])))
                
            task_ids.remove(task_id)

        while ((len(controller.ready_workers) > 0) and not next_iter):
            logger.info(f"controller.ready_workers: {controller.ready_workers}")
            logger.info(f"controller.workers_available: {controller.workers_available}")
            eval_x_dict = {}
            for problem_id in sopt.problem_ids:
                eval_x = sopt.optimizer_dict[problem_id].get_next_x()
                if eval_x is None:
                    next_iter = True
                else:
                    eval_x_dict[problem_id] = eval_x
            if next_iter:
                break
            task_id = controller.submit_call("eval_fun", module_name="dmosopt",
                                             args=(sopt.opt_id, eval_x_dict,))
            task_ids.append(task_id)
            n_tasks += 1
            for problem_id in sopt.problem_ids:
                sopt.evals[problem_id][task_id] = eval_x_dict[problem_id]

        if next_iter and (len(task_ids) == 0):
            sopt.optimizer_dict[problem_id].step()
            next_iter = False
            iter_count += 1

    if sopt.save:
        sopt.save_evals()
    controller.info()

def sopt_work(worker, sopt_params, verbose=False):
    """Worker for distributed surrogate optimization."""
    sopt_init(sopt_params, worker=worker, verbose=verbose)

def eval_fun(opt_id, *args):
    return sopt_dict[opt_id].eval_fun(*args)

def run(sopt_params, spawn_workers=False, nprocs_per_worker=1, verbose=True):
    if distwq.is_controller:
        distwq.run(fun_name="sopt_ctrl", module_name="dmosopt",
                   verbose=verbose, args=(sopt_params, verbose,),
                   spawn_workers=spawn_workers,
                   nprocs_per_worker=nprocs_per_worker)
        opt_id = sopt_params['opt_id']
        sopt = sopt_dict[opt_id]
        sopt.print_best()
        return sopt.get_best()
    else:
        if 'file_path' in sopt_params:
            del(sopt_params['file_path'])
        if 'save' in sopt_params:
            del(sopt_params['save'])
        distwq.run(fun_name="sopt_work", module_name="dmosopt",
                   broker_fun_name=sopt_params.get("broker_fun_name", None),
                   broker_module_name=sopt_params.get("broker_module_name", None),
                   verbose=verbose, args=(sopt_params, verbose, ),
                   spawn_workers=spawn_workers,
                   nprocs_per_worker=nprocs_per_worker)
        return None
        
