"""
MOASMO.epoch builds the feasibility model from the training data.
"""

import logging

import numpy as np
import pytest

from dmosopt import MOASMO


class RecordingFeasibility:
    calls = []

    def __init__(self, x, c):
        RecordingFeasibility.calls.append((x, c))

    def rank(self, x):
        return np.ones(x.shape[0])


def _start_epoch(feasibility_method_name, data):
    X, Y, C, xlb, xub = data
    gen = MOASMO.epoch(
        num_generations=2,
        param_names=[f"x{i}" for i in range(X.shape[1])],
        objective_names=[f"y{i}" for i in range(Y.shape[1])],
        xlb=xlb,
        xub=xub,
        pct=0.5,
        Xinit=X,
        Yinit=Y,
        C=C,
        pop=8,
        feasibility_method_name=feasibility_method_name,
        surrogate_method_name="gpr",
        local_random=np.random.default_rng(0),
        logger=logging.getLogger("dmosopt"),
    )
    with pytest.raises(StopIteration):
        next(gen)


def test_epoch_fits_feasibility_model_on_initial_samples(synthetic_data):
    X, Y, C, _, _ = synthetic_data
    RecordingFeasibility.calls.clear()

    _start_epoch(f"{__name__}.RecordingFeasibility", synthetic_data)

    assert len(RecordingFeasibility.calls) == 1
    x, c = RecordingFeasibility.calls[0]
    assert x is X
    assert c is C


def test_epoch_builds_default_feasibility_model(caplog, synthetic_data):
    with caplog.at_level(logging.WARNING, logger="dmosopt"):
        _start_epoch("logreg", synthetic_data)

    assert not [r for r in caplog.records if "feasibility model" in r.getMessage()]
