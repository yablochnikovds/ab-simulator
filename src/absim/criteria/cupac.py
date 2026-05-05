"""CUPAC — CUPED with a model-based control variate.

Same idea as CUPED, but the covariate is the out-of-fold prediction of an ML
model trained on **pre-experiment features**. Two ingestion modes:

1. The caller supplies pre-computed predictions
   (``prediction_treatment``, ``prediction_control`` kwargs) — the typical
   production setup.
2. The caller supplies feature matrices
   (``features_treatment``, ``features_control`` kwargs) and CUPAC will fit
   a default :class:`~sklearn.linear_model.RidgeCV` regressor with K-fold
   out-of-fold predictions per group.

References
----------
Tang, Y., Bahmani, A., Liang, T., & Bakshy, E. (2020). CUPAC: variance
reduction with control variates. (DoorDash engineering blog write-up of the
ML-extension of CUPED.)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

from absim.criteria.base import register
from absim.criteria.cuped import CUPED
from absim.types import TestResult

if TYPE_CHECKING:
    from absim.types import FloatArray


class _RegressorLike(Protocol):
    def fit(self, X: FloatArray, y: FloatArray) -> Any: ...
    def predict(self, X: FloatArray) -> FloatArray: ...


def _default_regressor() -> _RegressorLike:
    model: _RegressorLike = RidgeCV(alphas=(0.1, 1.0, 10.0))
    return model


def _oof_predict_pooled(
    X_t: FloatArray,
    y_t: FloatArray,
    X_c: FloatArray,
    y_c: FloatArray,
    n_splits: int,
    regressor_factory: Any,
    seed: int,
) -> tuple[FloatArray, FloatArray]:
    """Out-of-fold predictions for both arms using a single pooled model.

    Fitting per-arm leaks the treatment effect into the predictor; pooling
    fits one regression of :math:`Y` on :math:`X` that is symmetric in arm,
    preserving the average treatment effect while reducing variance.
    """
    if X_t.shape[0] != y_t.shape[0] or X_c.shape[0] != y_c.shape[0]:
        raise ValueError("feature and outcome lengths must match within each arm")
    X = np.vstack([X_t, X_c])
    y = np.concatenate([y_t, y_c])
    n_t = X_t.shape[0]
    n = X.shape[0]
    with warnings.catch_warnings():
        # RidgeCV warns about LOO degeneracy on very small folds; harmless here.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        if n < n_splits:
            model = regressor_factory()
            model.fit(X, y)
            preds = np.asarray(model.predict(X), dtype=float)
        else:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
            preds = np.empty(n, dtype=float)
            for train_idx, test_idx in kf.split(X):
                model = regressor_factory()
                model.fit(X[train_idx], y[train_idx])
                preds[test_idx] = model.predict(X[test_idx])
    return preds[:n_t], preds[n_t:]


@register("cupac")
@dataclass(frozen=True, slots=True)
class CUPAC:
    """CUPAC: CUPED with an out-of-fold ML prediction as the covariate.

    Parameters
    ----------
    alpha
        Significance level.
    n_splits
        Number of CV folds when fitting the default model in-place. Ignored
        when callers pass ``prediction_*`` directly.
    seed
        Random state for the KFold splitter when fitting in-place.
    """

    alpha: float = 0.05
    n_splits: int = 5
    seed: int = 0
    name: str = "cupac"
    _cuped: CUPED = field(default_factory=CUPED, init=False, repr=False)

    def test(
        self,
        treatment: FloatArray,
        control: FloatArray,
        **kwargs: Any,
    ) -> TestResult:
        """Run a CUPAC-adjusted Welch t-test."""
        if "prediction_treatment" in kwargs and "prediction_control" in kwargs:
            cov_t: FloatArray = np.asarray(kwargs["prediction_treatment"], dtype=float)
            cov_c: FloatArray = np.asarray(kwargs["prediction_control"], dtype=float)
        elif "features_treatment" in kwargs and "features_control" in kwargs:
            feat_t: FloatArray = np.asarray(kwargs["features_treatment"], dtype=float)
            feat_c: FloatArray = np.asarray(kwargs["features_control"], dtype=float)
            cov_t, cov_c = _oof_predict_pooled(
                feat_t,
                treatment,
                feat_c,
                control,
                self.n_splits,
                _default_regressor,
                self.seed,
            )
        else:
            raise ValueError(
                "CUPAC needs either prediction_{treatment,control} or "
                "features_{treatment,control} kwargs"
            )
        cuped = CUPED(alpha=self.alpha)
        result = cuped.test(treatment, control, covariate_treatment=cov_t, covariate_control=cov_c)
        # Re-stamp the criterion-specific metadata.
        return TestResult(
            p_value=result.p_value,
            statistic=result.statistic,
            effect=result.effect,
            std_error=result.std_error,
            ci_low=result.ci_low,
            ci_high=result.ci_high,
            rejected=result.rejected,
            metadata={**result.metadata, "model": "RidgeCV", "n_splits": self.n_splits},
        )
