import sys
import importlib


def import_object_by_path(path):
    module_path, _, obj_name = path.rpartition(".")
    if module_path == "__main__" or module_path == "":
        module = sys.modules["__main__"]
    else:
        module = importlib.import_module(module_path)
    return getattr(module, obj_name)


default_sampling_methods = {
    "glp": "dmosopt.sampling.glp",
    "slh": "dmosopt.sampling.slh",
    "lh": "dmosopt.sampling.lh",
    "mc": "dmosopt.sampling.mc",
    "sobol": "dmosopt.sampling.sobol",
}

default_optimizers = {
    "nsga2": "dmosopt.NSGA2.NSGA2",
    "age": "dmosopt.AGEMOEA.AGEMOEA",
    "smpso": "dmosopt.SMPSO.SMPSO",
    "cmaes": "dmosopt.CMAES.CMAES",
    "trs": "dmosopt.TRS.TRS",
}

default_surrogate_methods = {
    "gpr": "dmosopt.model.GPR_Matern",
    "egp": "dmosopt.model.EGP_Matern",
    "megp": "dmosopt.model.MEGP_Matern",
    "mdgp": "dmosopt.model.MDGP_Matern",
    "mdspp": "dmosopt.model.MDSPP_Matern",
    "vgp": "dmosopt.model.VGP_Matern",
    "svgp": "dmosopt.model.SVGP_Matern",
    "spv": "dmosopt.model.SPV_Matern",
    "siv": "dmosopt.model.SIV_Matern",
    "crv": "dmosopt.model.CRV_Matern",
}

default_sa_methods = {
    "dgsm": "dmosopt.sa.SA_DGSM",
    "fast": "dmosopt.sa.SA_FAST",
}

default_feasibility_methods = {
    'logreg': "dmosopt.feasibility.LogisticFeasibilityModel"
}