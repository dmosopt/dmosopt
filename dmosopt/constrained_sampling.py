import numpy as np
from numpy.random import default_rng
from sly import Lexer, Parser

class ParamSpacePoints:
    def __init__(self, N, SpaceUnc, SpaceUncMethod=None, SpaceCons=None, seed=None):
        self.seed = seed
        self.rng = default_rng() if self.seed is None else default_rng(self.eed)

        self.N_params = N
        self.SpaceUnc = SpaceUnc
        self.SpaceUncMethod = SpaceUncMethod
        self.SpaceUnc_vals = self.generate_unconstrained(self.SpaceUnc)
        self.ParamSample = self.SpaceUnc_vals

        if SpaceCons is not None:
            cons_params = SpaceCons.keys()
            if len(cons_params):
                self.SpaceCons = SpaceCons
                self.SpaceCons_bnds = self.generate_constrained(self.SpaceUnc_vals, SpaceCons)
                self.SpaceUnc_val = self.sample_space_cons(self.SpaceCons_bnds)
                self.ParamSample.update(self.SpaceUnc_val)

    def generate_unconstrained(self, Space):
        
      #  self.SpaceUnc_val = # Call Method 

        # For isyntax testing only 
        Space_val = {
            key: np.random.uniform(*val, size=self.N_params) for key, val in Space.items()
        }
        return Space_val 


    def generate_constrained(self, SpaceUnc_val, SpaceCons):

        self.lexercons = BoundaryLexer()
        self.parsercons = BoundaryParser()
        cons_bounds = {}

        for key, val in SpaceCons.items(): 
            valkys = val.keys()
            absbnds = None
            lbbnds = None
            ubbnds = None

            if 'abs' in valkys:
                if np.isfinite(np.multiply(*val['abs'])):
                    if val['abs'][1] > val['abs'][0]:
                        absbnds = val['abs']

            if 'lb' in valkys:
                if len(val['lb']):
                    lbcons = val['lb']
                    lbbnds = self.get_bounds(lbcons, SpaceUnc_val) 

            if 'ub' in valkys:
                if len(val['ub']):
                    ubcons = val['ub']
                    ubbnds = self.get_bounds(ubcons, SpaceUnc_val) 

            bounds = self.solve_bounds(absbnds, lbbnds, ubbnds)

            cons_bounds[key] = (bounds, val['method']) 

        return cons_bounds

    def solve_bounds(self, absbnds, lbbnds, ubbnds, absolute=True, default=True, singlevalid=True):
    
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
            if type(lb) is np.float64 :
                all_valid = validate 
            else:
                all_valid = all(validate)
            return validate, all_valid

        if absbnds is None:
            if lbbnds is None or ubbnds is None:
                raise KeyError('Constrained parameter requires both lower and upper bounds when absolute bounds are not specified.')
            else:
                lb = calculate_bounds(lbbnds)
                ub = calculate_bounds(ubbnds)
                _ , all_valid = validate_bounds(lb, ub)
                if not all_valid:
                    print('Unsolvable constraints for at least one sample - no absolute bounds specified') 
                if not singlevalid:
                    raise ValueError('Requires all samples to be valid')
        else:
            if default:
                defaulted = False
                if lbbnds is None and ubbnds is not None:
                    print('Lower bounds defaulting to absolute range.')
                    lb = np.full(shape=self.N_params, fill_value=absbnds[0])
                    ub = calculate_bounds(ubbnds, lower=False)
                elif lbbnds is not None and ubbnds is None:
                    print('Upper bounds defaulting to absolute range.')
                    lb = calculate_bounds(lbbnds)
                    ub = np.full(shape=self.N_params, fill_value=absbnds[1])
                elif lbbnds is  None and ubbnds is None:
                    print('Both bounds defaulting to absolute range.')
                    lb = np.full(shape=self.N_params, fill_value=absbnds[0])
                    ub = np.full(shape=self.N_params, fill_value=absbnds[1])
                    defaulted = True
                else:
                    lb = calculate_bounds(lbbnds)
                    ub = calculate_bounds(ubbnds, lower=False)
                

            if not defaulted:
                validate, all_valid = validate_bounds(lb, ub)
                if not all_valid:
                    non_valid_idx = np.where(validate==False)[0]
                    print('Substituting overconstrained sample range with absolute values')
                    for i in non_valid_idx:
                        lb_i = lb[i]
                        ub_i = ub[i] 
                        ### TO DO
                        disub = abs(lb_i-absbnds[0]) 
                        dislb = abs(ub_i-absbnds[1]) 
                        lb[i] = absbnds[0] 
                        ub[i] = absbnds[1] 

        return lb, ub

    def get_bounds(self, bndcnst, spc_unc_val):
    
        bnds = np.empty(shape=self.N_params, dtype='O') 
        for idx in range(self.N_params):
            bnds_temp = []
            for cons in bndcnst:
                bnd = self.get_constr_bounds(idx, cons, spc_unc_val)
                bnds_temp.append(bnd)
            bnds[idx] = np.array(bnds_temp)
    
        return bnds

    def get_constr_bounds(self, idx, constr, space_unc_val):
        param, rel = constr
    
        param_val = space_unc_val[param][idx]
    
        val = self.parsercons.parse(self.lexercons.tokenize(f'{param_val} {rel}'))
    
        return val

    def sample_space_cons(self, spcons_bnds):
    
        space_cns_val = {}
        for k,v in spcons_bnds.items():
            space_cns_val[k] = self.get_sample_val(*v, seed=None)
        return space_cns_val
            
    def get_sample_val(self, bnd, method, seed=None):
        rng = default_rng() if seed is None else default_rng(seed)

        bndmid = np.mean(bnd, axis=0)
        bndrange = bnd[1]-bnd[0]
    
        if method[0] == 'uniform':
            val = rng.uniform(*bnd)
        elif method[0] == 'normal':
            val_off = 0.5*rng.vonmises(*method[1:3])/np.pi * bndrange 
            val = bndmid + val_off 
        elif method[0] == 'percentile':
            val = bnd[0] + method[1]*bndrange
        return val


class BoundaryLexer(Lexer):
    # Set of token names.   This is always required
    tokens = { MIN, MAX, ID, NUMBER, PLUS, MINUS, POW, TIMES,
               DIVIDE, EQ, ASSIGN, LPAREN, RPAREN,
               LE, LT, GE, GT, NE}

    # String containing ignored characters between tokens
    ignore = ' \t'

    # Regular expression rules for tokens
    ID      = r'[a-zA-Z_][A-zA-Z0-9_]*'
    NUMBER  = r'[-+]?([0-9]*[.])?[0-9]+([eE][-+]?\d+)?'
    PLUS    = r'\+'
    MINUS   = r'-'
    POW     = r'\*\*'
    TIMES   = r'\*'
    DIVIDE  = r'/'
    EQ      = r'=='
    ASSIGN  = r'='
    LPAREN  = r'\('
    RPAREN  = r'\)'
    LE      = r'<='
    LT      = r'<'
    GE      = r'>='
    GT      = r'>'
    NE      = r'!='

    ID['MIN'] = MIN
    ID['MAX'] = MAX
    ID['min'] = MIN
    ID['max'] = MAX

    def NUMBER(self, t):
        try:
            temp = float(t.value)
            t.value = temp if ((temp - int(temp)) != 0 or '.' in t.value or 'e' in t.value) else int(temp)
        except ValueError:
            t.value = float(t.value)
        except TypeError:
            pass
        return t         


class BoundaryParser(Parser):
    tokens = set(['NUMBER', 'PLUS', 'MINUS', 'TIMES', 'DIVIDE', 'MIN', 'MAX'])
    precedence = (
       ('left', PLUS, MINUS),
       ('left', TIMES, DIVIDE),
       ('left', MIN, MAX),
    )

    # Grammar rules and actions
    @_('expr PLUS term')
    def expr(self, p):
        return p.expr + p.term

    @_('expr MINUS term')
    def expr(self, p):
        return p.expr - p.term

    @_('term')
    def expr(self, p):
        return p.term

    @_('term TIMES factor')
    def term(self, p):
        return p.term * p.factor

    @_('term DIVIDE factor')
    def term(self, p):
        return p.term / p.factor

    @_('term MIN factor')
    def term(self, p):
        return min(p.term, p.factor)

    @_('term MAX factor')
    def term(self,p):
        return max(p.term, p.factor)

    @_('factor')
    def term(self, p):
        return p.factor

    @_('NUMBER')
    def factor(self, p):
        return p.NUMBER
