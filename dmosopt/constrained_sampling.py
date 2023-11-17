import numpy as np
from numpy.random import default_rng
import logging
from dmosopt import sampling
from sly import Lexer, Parser
from dmosopt.MOEA import (
    crossover_sbx,
    mutation,
    tournament_selection,
    remove_duplicates,
)


class ParamSpacePoints:
    def __init__(self, N, Space, Method=None, seed=None, parents=None):
        self.seed = seed
        self.rng = default_rng() if self.seed is None else default_rng(self.eed)

        self.N_params = N
        self.Space = Space
        self.parents_dict = parents
        self.analyze_param_space()
        self.MethodUnc = Method

        if Method is None and parents is None:
            self.SpaceUncMethod = "slh"

        self.generate_param()

    def analyze_param_space(self):
        Space = self.Space
        params_idx_unc = []
        params_idx_con = []
        self.param_keys = np.sort(list(Space.keys()))
        for kidx, key in enumerate(self.param_keys):
            typ = type(Space[key])
            if typ is list:
                params_idx_unc.append(kidx)
            elif typ is dict:
                params_idx_con.append(kidx)
        self.prm_idx_unc = np.array(params_idx_unc)
        self.prm_idx_con = np.array(params_idx_con)

        self.prm_unc_dim = np.array(params_idx_unc).shape[0]
        self.prm_con_dim = np.array(params_idx_con).shape[0]

        self.param_dim = self.prm_unc_dim + self.prm_con_dim

        self.unc_intervals = np.empty(shape=(self.prm_unc_dim, 2))
        #        self.param_arr = np.full(shape=(self.N_params, self.param_dim), fill_value=np.nan)

        for idx, kidx in enumerate(self.prm_idx_unc):
            key = self.param_keys[kidx]
            self.unc_intervals[idx, :] = Space[key]

        self.EvoMeth = False

        if self.parents_dict is not None:
            if len(self.parents_dict["params"]) == self.parents_dict["values"].shape[1]:
                if np.isin(
                    self.param_keys[self.prm_idx_unc],
                    self.parents_dict["params"],
                    assume_unique=True,
                ).all():
                    self.EvoMeth = True
                    self.SpaceUncMethod = "Evo"

                    class parents:
                        pass

                    self.parents = parents()
                    for key, val in self.parents_dict.items():
                        setattr(self.parents, key, val)

                    sort_parent_params_idx = []
                    for key in self.param_keys[self.prm_idx_unc]:
                        sort_parent_params_idx.append(
                            np.where(self.parents.params == key)[0][0]
                        )

                    self.parents.unc_values = self.parents.values[
                        :, sort_parent_params_idx
                    ]

                # can be extended to not ignore constrained params
                # would need to check whether cons params respect bounds... ideally they should...
                else:
                    print("Missing unconstrained params from parents")

            else:
                print("Mismatch between parent params and values dimensions.")

    def generate_param(self):
        self.generate_unconstrained()

        print(self.param_arr)

        if self.prm_con_dim:
            self.solve_constrained_dependency()
            self.generate_constrained()

    def generate_unconstrained(self):
        method = self.SpaceUncMethod
        if method in ["glp", "slh", "lh", "mc"]:
            self.param_arr = np.full(
                shape=(self.N_params, self.param_dim), fill_value=np.nan
            )
            Xinit = self.initial_sampling(method)
            xlb = self.unc_intervals[:, 0]
            xub = self.unc_intervals[:, 1]
            self.param_arr[:, self.prm_idx_unc] = Xinit * (xub - xlb) + xlb
        elif self.EvoMeth:
            Xinit = self.get_children()
        elif callable(method):
            Xinit = method(Ninit, nInput, local_random)
        else:
            raise RuntimeError(f"Unknown method {method}")

    def get_children(self):
        """
        model: the evaluated model function
        nInput: number of model input
        nOutput: number of output objectives
        xlb: lower bound of input
        xub: upper bound of input
        pop: number of population
        gen: number of generation
        crossover_rate: ratio of crossover in each generation
        mutation_rate: ratio of muration in each generation
        di_crossover: distribution index for crossover
        di_mutation: distribution index for mutation
        sampling_method: optional callable for initial sampling of parameters
        """

        self.parents.poolsize = int(round(self.parents.pop_size / 2.0))
        #
        self.parents.local_random = (
            self.rng if self.parents.local_random is None else self.parents.local_random
        )
        self.parents.nchildren = (
            1 if self.parents.feasibility_model is None else self.parents.poolsize
        )

        xs_gen = []
        count = 0

        while count < self.parents.pop_size - 1:
            if self.parents.local_random.random() < self.parents.crossover_rate:
                print(
                    count,
                    "Crossover",
                    self.parents.local_random.random(),
                    self.parents.crossover_rate,
                )
                parentidx = self.parents.local_random.choice(
                    self.parents.poolsize, 2, replace=False
                )

                parent1 = self.parents.unc_values[parentidx[0], :]
                parent2 = self.parents.unc_values[parentidx[1], :]
                children1, children2 = crossover_sbx(
                    self.parents.local_random,
                    parent1,
                    parent2,
                    self.parents.di_crossover,
                    self.unc_intervals[:, 0],
                    self.unc_intervals[:, 1],
                    nchildren=self.parents.nchildren,
                )
                if self.parents.feasibility_model is None:
                    child1 = children1[0]
                    child2 = children2[0]
                else:
                    child1, child2 = crossover_sbx_feasibility_selection(
                        self.parents.local_random,
                        self.parents.feasibility_model,
                        [children1, children2],
                        logger=logger,
                    )
                xs_gen.extend([child1, child2])
                count += 2
            elif self.parents.ranks is not None:
                #  else:
                print("Mutation")
                pool_idxs = tournament_selection(
                    self.parents.local_random,
                    self.parents.pop_size,
                    self.parents.poolsize,
                    self.parents.toursize,
                    self.parents.ranks,
                )

                # Need value_array to have been ranked beforehand
                pool = self.parents.unc_values[pool_idxs, :]

                parentidx = self.parents.local_random.integers(
                    low=0, high=self.parents.poolsize
                )
                parent = pool[parentidx, :]
                children = mutation(
                    self.parents.local_random,
                    parent,
                    self.parents.mutation_rate,
                    self.parents.di_mutation,
                    self.unc_intervals[:, 0],
                    self.unc_intervals[:, 1],
                    nchildren=self.parents.nchildren,
                )
                if self.parents.feasibility_model is None:
                    child = children[0]
                else:
                    child = feasibility_selection(
                        self.parents.local_random,
                        self.parents.feasibility_model,
                        children,
                        logger=logger,
                    )
                xs_gen.append(child)
                count += 1

        self.N_params = len(xs_gen)
        self.param_arr = np.full(
            shape=(self.N_params, self.param_dim), fill_value=np.nan
        )
        self.param_arr[:, self.prm_idx_unc] = np.vstack(xs_gen)

    def initial_sampling(self, method="glp", maxiter=5, local_random=None, logger=None):
        """
        Initialization for Multi-Objective Adaptive Surrogate Modelling-based Optimization
        nEval: number of evaluations per parameter
        nInput: number of model parameters
        nOutput: number of output objectives
        xlb: lower bound of input
        xub: upper bound of input
        """
        Ninit = self.N_params
        nInput = self.prm_unc_dim

        if local_random is None:
            local_random = self.rng

        if Ninit <= 0:
            return None

        if logger is not None:
            logger.info(f"xinit: generating {Ninit} initial parameters...")

        ini_func = getattr(sampling, method)
        Xinit = ini_func(Ninit, nInput, local_random=local_random, maxiter=maxiter)

        return Xinit

    def solve_constrained_dependency(self):
        consprm_dt = np.dtype(
            [
                ("param", "U64"),
                ("abs", bool),
                ("absbnds", float, 2),
                ("lbprms", np.ndarray),
                ("lbrels", np.ndarray),
                ("ubprms", np.ndarray),
                ("ubrels", np.ndarray),
                ("rank", int),
                ("perm_idx", int),
            ]
        )
        cons_arr = np.empty(shape=self.prm_idx_con.shape[0], dtype=consprm_dt)
        cons_arr["rank"] = 100
        unc_parms = self.param_keys[self.prm_idx_unc]
        for pidx, prm in enumerate(self.prm_idx_con):
            key = self.param_keys[self.prm_idx_con][pidx]
            cons_ent = cons_arr[pidx]
            cons_ent["param"] = key
            val = self.Space[key]
            valkys = val.keys()

            if "abs" in valkys:
                if np.isfinite(np.multiply(*val["abs"])):
                    if val["abs"][1] > val["abs"][0]:
                        cons_ent["abs"] = True
                        cons_ent["absbnds"] = val["abs"]

            if "lb" in valkys:
                lbcons = val["lb"]
                lbprms = []
                lbrels = []
                for const in lbcons:
                    lbprms.append(const[0])
                    lbrels.append(const[1])
                cons_ent["lbprms"] = np.array(lbprms)
                cons_ent["lbrels"] = np.array(lbrels)

            if "ub" in valkys:
                ubcons = val["ub"]
                ubprms = []
                ubrels = []
                for const in ubcons:
                    ubprms.append(const[0])
                    ubrels.append(const[1])
                cons_ent["ubprms"] = np.array(ubprms)
                cons_ent["ubrels"] = np.array(ubrels)

            if cons_ent["lbprms"] is None and cons_ent["ubprms"] is None:
                cons_ent["rank"] = 0
            else:
                lbprms = [] if cons_ent["lbprms"] is None else cons_ent["lbprms"]
                ubprms = [] if cons_ent["ubprms"] is None else cons_ent["ubprms"]
                depprms = np.union1d(lbprms, ubprms)
                test_depprms = np.in1d(depprms, unc_parms, assume_unique=True)
                if test_depprms.all():
                    cons_ent["rank"] = 0

        # Implemented for 1 level only... can expand later in a while loop
        # Need to catch circular dependencies
        zranked = np.where(cons_arr["rank"] == 0)[0]
        nzranks = np.delete(np.arange(cons_arr.shape[0]), zranked)
        rank_i = 0

        cunc_params = cons_arr["param"][zranked]
        for cnsidx in nzranks:
            cons_ent = cons_arr[cnsidx]
            lbprms = [] if cons_ent["lbprms"] is None else cons_ent["lbprms"]
            ubprms = [] if cons_ent["ubprms"] is None else cons_ent["ubprms"]
            depprms = np.union1d(lbprms, ubprms)
            con_depprms = np.setdiff1d(depprms, cunc_params, assume_unique=True)
            cons_ent["rank"] = len(con_depprms)

        self.perm_idx = np.argsort(cons_arr["rank"])
        self.cons_arr = cons_arr

    def generate_constrained(self):
        self.lexercons = BoundaryLexer()
        self.parsercons = BoundaryParser()
        cons_bounds = {}

        for constr in self.cons_arr[self.perm_idx]:
            absbnds = None
            lbbnds = None
            ubbnds = None
            key = constr["param"]
            absbnds = constr["absbnds"]
            if constr["lbprms"] is not None:
                lbbnds = self.get_bounds(constr["lbprms"], constr["lbrels"])
            if constr["ubprms"] is not None:
                ubbnds = self.get_bounds(constr["ubprms"], constr["ubrels"])

            bounds = self.solve_bounds(absbnds, lbbnds, ubbnds)
            cons_bounds[key] = (bounds, self.Space[key]["method"])

        cons_sampled_vals = self.sample_space_cons(cons_bounds)
        for kidx in self.prm_idx_con:
            key = self.param_keys[kidx]
            self.param_arr[:, kidx] = cons_sampled_vals[key]

    def solve_bounds(
        self, absbnds, lbbnds, ubbnds, absolute=True, default=True, singlevalid=True
    ):
        def calculate_bounds(bnd, lower=True):
            bnd_arr = np.empty(shape=self.N_params)
            if lower:
                for idx in range(self.N_params):
                    bnd_arr[idx] = np.max(bnd[idx])
            else:
                for idx in range(self.N_params):
                    bnd_arr[idx] = np.min(bnd[idx])
            return bnd_arr

        def validate_bounds(lb, ub):
            validate = lb < ub
            if type(lb) is np.float64:
                all_valid = validate
            else:
                all_valid = all(validate)
            return validate, all_valid

        if absbnds is None:
            if lbbnds is None or ubbnds is None:
                raise KeyError(
                    "Constrained parameter requires both lower and upper bounds when absolute bounds are not specified."
                )
            else:
                lb = calculate_bounds(lbbnds)
                ub = calculate_bounds(ubbnds)
                _, all_valid = validate_bounds(lb, ub)
                if not all_valid:
                    print(
                        "Unsolvable constraints for at least one sample - no absolute bounds specified"
                    )
                if not singlevalid:
                    raise ValueError("Requires all samples to be valid")
        else:
            if default:
                defaulted = False
                if lbbnds is None and ubbnds is not None:
                    print("Lower bounds defaulting to absolute range.")
                    lb = np.full(shape=self.N_params, fill_value=absbnds[0])
                    ub = calculate_bounds(ubbnds, lower=False)
                elif lbbnds is not None and ubbnds is None:
                    print("Upper bounds defaulting to absolute range.")
                    lb = calculate_bounds(lbbnds)
                    ub = np.full(shape=self.N_params, fill_value=absbnds[1])
                elif lbbnds is None and ubbnds is None:
                    print("Both bounds defaulting to absolute range.")
                    lb = np.full(shape=self.N_params, fill_value=absbnds[0])
                    ub = np.full(shape=self.N_params, fill_value=absbnds[1])
                    defaulted = True
                else:
                    lb = calculate_bounds(lbbnds)
                    ub = calculate_bounds(ubbnds, lower=False)

            if not defaulted:
                validate, all_valid = validate_bounds(lb, ub)
                if not all_valid:
                    non_valid_idx = np.where(validate == False)[0]
                    print(
                        "Substituting overconstrained sample range with absolute values"
                    )
                    for i in non_valid_idx:
                        lb_i = lb[i]
                        ub_i = ub[i]
                        ### TO DO
                        disub = abs(lb_i - absbnds[0])
                        dislb = abs(ub_i - absbnds[1])
                        lb[i] = absbnds[0]
                        ub[i] = absbnds[1]

        return lb, ub

    def get_bounds(self, prms, rels):
        uncparams = prms

        N_constr = uncparams.shape[0]
        bnds = np.empty(shape=(self.N_params, N_constr), dtype="O")
        param_idx = np.searchsorted(self.param_keys[self.prm_idx_unc], uncparams)

        for prmidx, prm in enumerate(param_idx):
            param_vals = self.param_arr[:, self.prm_idx_unc[prm]]
            rel = rels[prmidx]
            for validx, param_val in enumerate(param_vals):
                bnds[validx, prmidx] = self.parsercons.parse(
                    self.lexercons.tokenize(f"{param_val} {rel}")
                )
        return bnds

    def sample_space_cons(self, spcons_bnds):
        space_cns_val = {}
        for k, v in spcons_bnds.items():
            space_cns_val[k] = self.get_sample_val(*v, seed=None)
        return space_cns_val

    def get_sample_val(self, bnd, method, seed=None):
        rng = default_rng() if seed is None else default_rng(seed)

        bndmid = np.mean(bnd, axis=0)
        bndrange = bnd[1] - bnd[0]

        if method[0] == "uniform":
            val = rng.uniform(*bnd)
        elif method[0] == "normal":
            val_off = 0.5 * rng.vonmises(*method[1:3]) / np.pi * bndrange
            val = bndmid + val_off
        elif method[0] == "percentile":
            val = bnd[0] + method[1] * bndrange
        return val


class BoundaryLexer(Lexer):
    # Set of token names.   This is always required
    tokens = {
        MIN,
        MAX,
        ID,
        NUMBER,
        PLUS,
        MINUS,
        POW,
        TIMES,
        DIVIDE,
        EQ,
        ASSIGN,
        LPAREN,
        RPAREN,
        LE,
        LT,
        GE,
        GT,
        NE,
    }

    # String containing ignored characters between tokens
    ignore = " \t"

    # Regular expression rules for tokens
    ID = r"[a-zA-Z_][A-zA-Z0-9_]*"
    NUMBER = r"[-+]?([0-9]*[.])?[0-9]+([eE][-+]?\d+)?"
    PLUS = r"\+"
    MINUS = r"-"
    POW = r"\*\*"
    TIMES = r"\*"
    DIVIDE = r"/"
    EQ = r"=="
    ASSIGN = r"="
    LPAREN = r"\("
    RPAREN = r"\)"
    LE = r"<="
    LT = r"<"
    GE = r">="
    GT = r">"
    NE = r"!="

    ID["MIN"] = MIN
    ID["MAX"] = MAX
    ID["min"] = MIN
    ID["max"] = MAX

    def NUMBER(self, t):
        try:
            temp = float(t.value)
            t.value = (
                temp
                if ((temp - int(temp)) != 0 or "." in t.value or "e" in t.value)
                else int(temp)
            )
        except ValueError:
            t.value = float(t.value)
        except TypeError:
            pass
        return t


class BoundaryParser(Parser):
    tokens = set(["NUMBER", "PLUS", "MINUS", "TIMES", "DIVIDE", "MIN", "MAX"])
    precedence = (
        ("left", PLUS, MINUS),
        ("left", TIMES, DIVIDE),
        ("left", MIN, MAX),
    )

    # Grammar rules and actions
    @_("expr PLUS term")
    def expr(self, p):
        return p.expr + p.term

    @_("expr MINUS term")
    def expr(self, p):
        return p.expr - p.term

    @_("term")
    def expr(self, p):
        return p.term

    @_("term TIMES factor")
    def term(self, p):
        return p.term * p.factor

    @_("term DIVIDE factor")
    def term(self, p):
        return p.term / p.factor

    @_("term MIN factor")
    def term(self, p):
        return min(p.term, p.factor)

    @_("term MAX factor")
    def term(self, p):
        return max(p.term, p.factor)

    @_("factor")
    def term(self, p):
        return p.factor

    @_("NUMBER")
    def factor(self, p):
        return p.NUMBER
