"""
Integration tests for dmosopt.model_transformer (renamed from custom_training).

Verifies:
- Module import under the new name
- Dynamic path resolution via import_object_by_path (mirrors MOASMO usage)
- End-to-end joint() training on a small synthetic dataset
- Logging output instead of bare print statements
"""

import logging
import numpy as np

from dmosopt.config import import_object_by_path


RNG = np.random.default_rng(42)


def _synthetic_data(n=40, n_params=3, n_obj=2, n_con=1):
    """Return (X, Y, C) with shapes (n, n_params), (n, n_obj), (n, n_con)."""
    X = RNG.uniform(0.0, 1.0, (n, n_params))
    Y = np.sum(X**2, axis=1, keepdims=True) * np.ones((1, n_obj)) + RNG.normal(
        0, 0.01, (n, n_obj)
    )
    C = np.sum(X, axis=1, keepdims=True) - n_params / 2.0  # feasible when > 0
    return X.astype(np.float32), Y.astype(np.float32), C.astype(np.float32)


# ---------------------------------------------------------------------------
# 1. Import check
# ---------------------------------------------------------------------------


def test_import():
    from dmosopt.model_transformer import joint, JointFTTransformer  # noqa: F401


# ---------------------------------------------------------------------------
# 2. Dynamic path resolution (mirrors MOASMO.py line 269)
# ---------------------------------------------------------------------------


def test_path_resolution():
    fn = import_object_by_path("dmosopt.model_transformer.joint")
    from dmosopt.model_transformer import joint

    assert fn is joint


# ---------------------------------------------------------------------------
# 3. End-to-end joint() call with fixed epochs (avoids expensive autoepoch CV)
# ---------------------------------------------------------------------------


def test_joint_returns_tuple():
    X, Y, C = _synthetic_data()
    xlb = np.zeros(X.shape[1], dtype=np.float32)
    xub = np.ones(X.shape[1], dtype=np.float32)

    from dmosopt.model_transformer import joint

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


def test_logging_emitted(caplog):
    X, Y, C = _synthetic_data()
    xlb = np.zeros(X.shape[1], dtype=np.float32)
    xub = np.ones(X.shape[1], dtype=np.float32)

    from dmosopt.model_transformer import joint

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
