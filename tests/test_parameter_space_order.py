from dmosopt import dmosopt
import numpy as np
import h5py
import pandas as pd


def objective_function(params):
    # return params as objectives to compare later
    return np.array(
        [
            params["0_1"],
            params["1_2"],
            params["2_3"],
        ]
    )


def check_order():
    parameter_space = {
        "1_2": [1, 2],
        "0_1": [0, 1],
        "2_3": [2, 3],
    }

    best_params, best_objectives = dmosopt.run(
        {
            "opt_id": "debug",
            "obj_fun_name": "objective_function",
            "space": parameter_space,
            "n_epochs": 1,
            "population_size": 3,
            "num_generations": 3,
            "problem_parameters": {},
            "objective_names": [
                "0_1",
                "1_2",
                "2_3",
            ],
            "file_path": "./test.h5",
            "save": True,
        }
    )

    # reload from file

    with h5py.File("./test.h5", "r") as h5:
        # objectives
        objective_enum = h5py.check_enum_dtype(h5["debug/objective_enum"].dtype)
        objective_enum_T = {v: k for k, v in objective_enum.items()}
        objective_names = [
            objective_enum_T[s[0]] for s in iter(h5["debug/objective_spec"])
        ]
        objectives = pd.DataFrame(h5["debug/0/objectives"][:], columns=objective_names)

        # parameters
        parameter_enum = h5py.check_enum_dtype(h5["debug/parameter_enum"].dtype)
        parameter_enum_T = {v: k for k, v in parameter_enum.items()}

        parameter_names = [
            parameter_enum_T[s[0]] for s in iter(h5["debug/parameter_spec"])
        ]

        parameters = pd.DataFrame(h5["debug/0/parameters"][:], columns=parameter_names)

    # at runtime, the samples were correct:
    for column in objectives.columns:
        values = objectives[column]
        min_val, max_val = map(int, column.split("_"))
        assert all(min_val <= v <= max_val for v in values), (
            f"Values in {column} outside range [{min_val}, {max_val}]"
        )

    # but they are being saved in a different order
    for column in parameters.columns:
        values = parameters[column]
        min_val, max_val = map(int, column.split("_"))
        assert all(min_val <= v <= max_val for v in values), (
            f"Values in {column} outside range [{min_val}, {max_val}]"
        )


check_order()
