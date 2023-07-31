import copy
import numpy as np
from functools import partial

try:
    from SALib.sample import fast_sampler, finite_diff
    from SALib.analyze import fast, dgsm

except:
    _has_salib = False
else:
    _has_salib = True


class SA_FAST:
    def __init__(self, lo_bounds, hi_bounds, param_names, output_names, logger=None):
        if not _has_salib:
            raise RuntimeError("SA_FAST requires the SALib library to be installed.")

        bounds = list(zip(lo_bounds, hi_bounds))
        self.problem = {
            "num_vars": len(param_names),
            "names": param_names,
            "bounds": bounds,
        }
        self.output_names = output_names
        self.logger = logger

    def sample(self, num_samples=10000):
        param_values = fast_sampler.sample(self.problem, num_samples)
        return param_values

    def analyze(self, model, num_samples=10000):
        Y = model.evaluate(self.sample(num_samples=num_samples))
        Sis = list(
            [
                fast.analyze(self.problem, Y[:, i], print_to_console=False)
                for i in range(Y.shape[1])
            ]
        )
        S1s = [s["S1"] for s in Sis]
        STs = [s["ST"] for s in Sis]

        result_dict = {
            "S1": dict(zip(self.output_names, S1s)),
            "ST": dict(zip(self.output_names, STs)),
        }
        return result_dict


class SA_DGSM:
    def __init__(self, lo_bounds, hi_bounds, param_names, output_names, logger=None):
        if not _has_salib:
            raise RuntimeError("SA_DGSM requires the SALib library to be installed.")

        bounds = list(zip(lo_bounds, hi_bounds))
        self.problem = {
            "num_vars": len(param_names),
            "names": param_names,
            "bounds": bounds,
        }
        self.output_names = output_names
        self.logger = logger

    def sample(self, num_samples=10000):
        param_values = finite_diff.sample(self.problem, num_samples)
        return param_values

    def analyze(self, model, num_samples=10000):
        param_values = self.sample(num_samples=num_samples)
        Y = model.evaluate(param_values)
        Sis = list(
            [
                dgsm.analyze(
                    self.problem, param_values, Y[:, i], print_to_console=False
                )
                for i in range(Y.shape[1])
            ]
        )
        S1s = [s["dgsm"] for s in Sis]

        result_dict = {"S1": dict(zip(self.output_names, S1s))}

        return result_dict
