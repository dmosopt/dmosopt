"""
Command-line tools dmosopt-analyze and dmosopt-train.

Each test writes a small optimization results file with save_to_h5 and
runs the click command on it in-process.  The problem has two parameters
(x0, x1), two objectives (f0 = x0, f1 = 1 - sqrt(x0) + x1), one feature
and one constraint (c0 = 0.5 - |x1 - 0.25|, feasible when positive).
Twenty evaluations are split over two epochs.
"""

import re

import numpy as np
import pytest
from click.testing import CliRunner
from joblib import load

from dmosopt import dmosopt, dmosopt_analyze, dmosopt_train
from dmosopt.datatypes import ParameterSpace
from dmosopt.MOASMO import get_best

OPT_ID = "cli_test"
OBJECTIVE_NAMES = ["f0", "f1"]
CONSTRAINT_NAMES = ["c0"]
FEATURE_DTYPES = [("feature", np.float32)]
N_EVALS = 20


def _problem_data():
    rng = np.random.default_rng(0)
    x = rng.uniform(0.0, 1.0, size=(N_EVALS, 2))
    y = np.column_stack([x[:, 0], 1.0 - np.sqrt(x[:, 0]) + x[:, 1]])
    c = np.column_stack([0.5 - np.abs(x[:, 1] - 0.25)])
    epochs = np.arange(N_EVALS) // 10
    return x, y, c, epochs


@pytest.fixture
def results_file(tmp_path):
    """Path of a results file holding the evaluations of _problem_data."""
    x, y, c, epochs = _problem_data()
    features = [np.array([(v,)], dtype=FEATURE_DTYPES) for v in x[:, 0]]
    evals = {
        0: (
            list(epochs),
            [tuple(row) for row in x],
            [tuple(row) for row in y],
            features,
            [tuple(row) for row in c],
            [tuple(row) for row in y],
        )
    }
    file_path = tmp_path / "results.h5"
    dmosopt.save_to_h5(
        OPT_ID,
        [0],
        False,
        OBJECTIVE_NAMES,
        FEATURE_DTYPES,
        CONSTRAINT_NAMES,
        ParameterSpace.from_dict({"x0": [0.0, 1.0], "x1": [0.0, 1.0]}),
        evals,
        ParameterSpace.from_dict({}, is_value_only=True),
        None,
        1,
        str(file_path),
        None,
    )
    return file_path


def _run(command, args):
    result = CliRunner().invoke(command, [str(a) for a in args])
    assert result.exit_code == 0, result.output + repr(result.exception)
    return result


def _read_best(output_file):
    """Parameter vectors written by dmosopt-analyze --output-file, in order."""
    rows = []
    for line in output_file.read_text().splitlines():
        _, values = line.split(":", 1)
        rows.append([float(v) for v in values.strip(" []").split(",")])
    return np.asarray(rows)


def _row_index(x, row):
    """Index of the evaluation whose parameters equal row (float32 storage)."""
    matches = np.flatnonzero(np.all(np.isclose(x, row, atol=1e-6), axis=1))
    assert len(matches) == 1
    return matches[0]


def test_analyze_reports_feasible_pareto_front(results_file, tmp_path):
    x, y, c, _ = _problem_data()
    output_file = tmp_path / "best.txt"
    result = _run(
        dmosopt_analyze.main,
        ["-p", results_file, "--opt-id", OPT_ID, "--output-file", output_file],
    )
    assert f"Found {N_EVALS} results for id 0" in result.output

    feasible = np.flatnonzero(c[:, 0] > 0.0)
    expected = set()
    for i in feasible:
        dominated = np.any(
            np.all(y[feasible] <= y[i], axis=1) & np.any(y[feasible] < y[i], axis=1)
        )
        if not dominated:
            expected.add(i)

    best = _read_best(output_file)
    assert {_row_index(x, row) for row in best} == expected
    assert f"Found {len(expected)} best results for id 0" in result.output


def test_analyze_sort_key_orders_by_objective(results_file, tmp_path):
    output_file = tmp_path / "best.txt"
    _run(
        dmosopt_analyze.main,
        [
            "-p",
            results_file,
            "--opt-id",
            OPT_ID,
            "--sort-key",
            "f0",
            "--output-file",
            output_file,
        ],
    )
    best = _read_best(output_file)
    # f0 equals x0, so the rows must come out in increasing x0.
    assert len(best) > 1
    assert np.all(np.diff(best[:, 0]) >= 0.0)


def test_analyze_knn_limits_output(results_file, tmp_path):
    output_file = tmp_path / "best.txt"
    _run(
        dmosopt_analyze.main,
        [
            "-p",
            results_file,
            "--opt-id",
            OPT_ID,
            "--knn",
            2,
            "--output-file",
            output_file,
        ],
    )
    assert len(_read_best(output_file)) == 2


def test_analyze_filter_objectives_selects_columns(results_file, tmp_path):
    x, _, c, _ = _problem_data()
    output_file = tmp_path / "best.txt"
    _run(
        dmosopt_analyze.main,
        [
            "-p",
            results_file,
            "--opt-id",
            OPT_ID,
            "--filter-objectives",
            "f0",
            "--output-file",
            output_file,
        ],
    )
    # With f0 = x0 as the only objective, the single best feasible
    # evaluation is the one with the smallest x0.
    feasible = np.flatnonzero(c[:, 0] > 0.0)
    best = _read_best(output_file)
    assert len(best) == 1
    assert _row_index(x, best[0]) == feasible[np.argmin(x[feasible, 0])]


def test_analyze_reports_epoch_of_each_best_evaluation(results_file):
    x, _, _, epochs = _problem_data()
    result = _run(dmosopt_analyze.main, ["-p", results_file, "--opt-id", OPT_ID])
    best_lines = [ln for ln in result.output.splitlines() if ln.startswith("Best eval")]
    assert best_lines
    pattern = re.compile(r": (E\d+) .*@\{'x0': (?:np\.float32\()?([-+0-9.eE]+)")
    for line in best_lines:
        match = pattern.search(line)
        assert match, line
        epoch_label, x0 = match.group(1), float(match.group(2))
        i = int(np.argmin(np.abs(x[:, 0] - x0)))
        assert epoch_label == f"E{epochs[i]}"


def test_train_writes_loadable_surrogate(results_file, tmp_path):
    x, y, c, _ = _problem_data()
    output_file = tmp_path / "surrogate.joblib"
    _run(
        dmosopt_train.main,
        ["-p", results_file, "--opt-id", OPT_ID, "-o", output_file],
    )
    assert output_file.exists()

    surrogate = load(output_file)
    mean, var = surrogate.predict(x[:3])
    assert mean.shape == (3, len(OBJECTIVE_NAMES))
    assert var.shape == (3, len(OBJECTIVE_NAMES))
    assert np.all(np.isfinite(mean))


def test_get_best_keeps_epochs_aligned_after_removing_duplicates():
    x = np.array([[0.1, 0.1], [0.2, 0.2], [0.9, 0.9], [0.5, 0.5]])
    y = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
    epochs = np.array([0, 1, 7, 3])

    best_x, _, _, _, best_epoch, _ = get_best(
        x, y, None, None, 2, 2, epochs=epochs, feasible=False
    )
    for row, epoch in zip(best_x, best_epoch):
        assert epoch == epochs[_row_index(x, row)]
