import tempfile
from pathlib import Path

import h5py
import numpy as np

from dmosopt import dmosopt
from dmosopt.datatypes import ParameterSpace


OPT_ID = "test"
OBJECTIVE_NAMES = ["f0", "f1"]
FEATURE_DTYPES = [("feature", np.float32)]


def _make_inputs():
    parameter_space = ParameterSpace.from_dict({"x": [0.0, 1.0], "y": [0.0, 1.0]})
    problem_parameters = ParameterSpace.from_dict({}, is_value_only=True)

    prob_evals_f = [
        np.array([(0.5,)], dtype=FEATURE_DTYPES),
        np.array([(0.6,)], dtype=FEATURE_DTYPES),
    ]
    evals = {
        0: (
            [0, 0],  # epochs
            [(0.1, 0.2), (0.3, 0.4)],  # parameters (x, y)
            [(1.0, 2.0), (3.0, 4.0)],  # objectives (f0, f1)
            prob_evals_f,  # features
            None,  # constraints
            [(1.0, 2.0), (3.0, 4.0)],  # predictions
        )
    }
    return parameter_space, problem_parameters, evals


def _save_to_h5(fpath):
    parameter_space, problem_parameters, evals = _make_inputs()
    dmosopt.save_to_h5(
        OPT_ID,
        [0],  # problem_ids
        False,  # has_problem_ids
        OBJECTIVE_NAMES,
        FEATURE_DTYPES,
        None,  # constraint_names
        parameter_space,
        evals,
        problem_parameters,
        None,  # metadata
        1,  # random_seed
        str(fpath),
        None,  # logger
    )


def _assert_well_formed(fpath):
    with h5py.File(str(fpath), "r") as f:
        grp = f[OPT_ID]
        objective_enum = h5py.check_enum_dtype(grp["objective_enum"].dtype)
        assert set(objective_enum.keys()) == set(OBJECTIVE_NAMES)
        # feature_dtypes must survive round-trip: the group has a feature enum
        # named after the dtype field, which the buggy call could never build.
        feature_enum = h5py.check_enum_dtype(grp["feature_enum"].dtype)
        assert set(feature_enum.keys()) == {"feature"}
        assert grp["0/objectives"].shape[0] == 2
        assert grp["0/features"].shape[0] == 2
        assert list(grp["problem_ids"][:]) == [0]


def test_save_to_h5_reinitializes_missing_group(tmp_path):
    fpath = tmp_path / "truncated.h5"
    with h5py.File(str(fpath), "w") as f:
        f.create_group("unrelated")

    _save_to_h5(fpath)  # must not raise
    _assert_well_formed(fpath)


def test_save_to_h5_fresh_file(tmp_path):
    fpath = tmp_path / "fresh.h5"
    _save_to_h5(fpath)
    _assert_well_formed(fpath)


def test_init_h5_and_save_to_h5_agree(tmp_path):
    parameter_space, problem_parameters, _ = _make_inputs()

    fpath = tmp_path / "init.h5"
    dmosopt.init_h5(
        OPT_ID,
        [0],  # problem_ids
        False,  # has_problem_ids
        parameter_space,
        OBJECTIVE_NAMES,
        FEATURE_DTYPES,
        None,  # constraint_names
        problem_parameters,
        None,  # metadata
        1,  # random_seed
        str(fpath),
    )

    with h5py.File(str(fpath), "r") as f:
        grp = f[OPT_ID]
        assert set(h5py.check_enum_dtype(grp["feature_enum"].dtype)) == {"feature"}
        assert set(h5py.check_enum_dtype(grp["objective_enum"].dtype)) == set(
            OBJECTIVE_NAMES
        )
        assert list(grp["problem_ids"][:]) == [0]

    # Appending evaluations to the init_h5-created file must not re-init or crash.
    _save_to_h5(fpath)
    _assert_well_formed(fpath)


def test_save_to_h5():
    for test in (
        test_save_to_h5_reinitializes_missing_group,
        test_save_to_h5_fresh_file,
        test_init_h5_and_save_to_h5_agree,
    ):
        with tempfile.TemporaryDirectory() as tmp_dir:
            test(Path(tmp_dir))
        print(f"{test.__name__} PASSED")


if __name__ == "__main__":
    test_save_to_h5()
