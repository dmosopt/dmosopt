"""
Integration tests for dmosopt.model_transformer (renamed from custom_training).

Verifies:
- Module import under the new name
- Dynamic path resolution via import_object_by_path (mirrors MOASMO usage)
- End-to-end joint() training on a small synthetic dataset
- Logging output instead of bare print statements
- Constraint weights keyed by constraint name
"""

import logging
import types

import pytest

pytest.importorskip("keras")

from dmosopt.config import import_object_by_path  # noqa: E402
from dmosopt.model_transformer import joint  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Import check
# ---------------------------------------------------------------------------


def test_import():
    from dmosopt.model_transformer import JointFTTransformer, joint  # noqa: F401


# ---------------------------------------------------------------------------
# 2. Dynamic path resolution (mirrors MOASMO.py line 269)
# ---------------------------------------------------------------------------


def test_path_resolution():
    fn = import_object_by_path("dmosopt.model_transformer.joint")
    assert fn is joint


# ---------------------------------------------------------------------------
# 3. End-to-end joint() call with fixed epochs (avoids expensive autoepoch CV)
# ---------------------------------------------------------------------------


def test_joint_returns_tuple(synthetic_data):
    X, Y, C, xlb, xub = synthetic_data

    result = joint(
        object,  # dummy optimizer_cls — not used inside joint()
        X,
        Y,
        C,
        xlb=xlb,
        xub=xub,
        file_path=None,
        options={},
        mode="c+o",
        objectives=True,
        constraints=True,
        sensitivity=False,
        epochs=5,  # fixed; avoids cross-validation in autoepoch
        iterations=[],
    )

    assert isinstance(result, tuple) and len(result) == 4
    optimizer_cls, model_obj, model_con, model_sens = result
    assert model_obj is not None
    assert model_con is not None
    assert model_sens is None  # sensitivity=False


# ---------------------------------------------------------------------------
# 4. Logging check — at least one INFO record from the "dmosopt" logger
# ---------------------------------------------------------------------------


def test_logging_emitted(caplog, synthetic_data):
    X, Y, C, xlb, xub = synthetic_data

    with caplog.at_level(logging.INFO, logger="dmosopt"):
        joint(
            object,
            X,
            Y,
            C,
            xlb=xlb,
            xub=xub,
            file_path=None,
            options={},
            mode="c+o",
            objectives=True,
            constraints=False,
            sensitivity=False,
            epochs=5,
            iterations=[],
        )

    dmosopt_records = [r for r in caplog.records if r.name == "dmosopt"]
    assert dmosopt_records, (
        "Expected at least one INFO log record from the 'dmosopt' logger"
    )


# ---------------------------------------------------------------------------
# 5. Constraint weights keyed by name
# ---------------------------------------------------------------------------


class _StubModel:
    def __init__(self, num_constraints, **kwargs):
        self.num_constraints = num_constraints

    def autofit(self, *args, **kwargs):
        pass

    def autoeval(self, *args, **kwargs):
        return {}


def _projected_weights(data, weights):
    X, Y, C, xlb, xub = data
    optimizer_cls, *_ = joint(
        lambda *args, **kwargs: types.SimpleNamespace(),
        X,
        Y,
        C,
        xlb=xlb,
        xub=xub,
        file_path=None,
        options={"constraint_names": ["g1", "g2", "g3"]},
        model_cls=_StubModel,
        sensitivity=False,
        feasibility_solving=True,
        feasibility_constraint_weights=weights,
    )
    return optimizer_cls()._constraint_weights


def test_named_weights_match_column_weights(synthetic_data):
    assert _projected_weights(synthetic_data, {"g2": 4, "g3": 2.5}) == (
        _projected_weights(synthetic_data, {1: 4, 2: 2.5})
    )


def test_unknown_constraint_name_raises(synthetic_data):
    with pytest.raises(KeyError, match="g4"):
        _projected_weights(synthetic_data, {"g4": 4})
