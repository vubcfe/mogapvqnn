"""Post-variational read-out: a linear SVM on classical-shadow features, and
majority-vote ensembles of such circuit models."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


# Liblinear may report non-convergence on tiny, nearly separable feature sets;
# the fitted model is still used. The filter is installed once at import time
# because warnings.catch_warnings() is not thread-safe.
warnings.filterwarnings("ignore", category=ConvergenceWarning, module=r"sklearn\.svm")


def make_linear_model(C: float = 1.0, max_iter: int = 20000):
    return make_pipeline(StandardScaler(), LinearSVC(C=C, dual="auto", max_iter=max_iter, random_state=0))


def fit_linear(F: np.ndarray, y: np.ndarray, C: float = 1.0, max_iter: int = 20000):
    model = make_linear_model(C, max_iter)
    model.fit(F, y)
    return model


@dataclass
class Scores:
    """Decision values of one fitted model on a split (positive -> class 1)."""

    decision: np.ndarray

    @property
    def pred(self) -> np.ndarray:
        return (self.decision > 0).astype(np.int64)


def accuracy(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(pred == y))


def majority_vote(decisions: list[np.ndarray]) -> np.ndarray:
    """Hard majority vote; ties (possible for even ensemble sizes) are broken
    by the sign of the summed standardised decision values."""
    D = np.stack(decisions)                       # (K, B)
    votes = (D > 0).mean(axis=0)
    scale = D.std(axis=1, keepdims=True)
    scale[scale == 0] = 1.0
    soft = (D / scale).sum(axis=0)
    pred = (votes > 0.5).astype(np.int64)
    tie = votes == 0.5
    pred[tie] = (soft[tie] > 0).astype(np.int64)
    return pred
