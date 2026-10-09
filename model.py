"""Shared neural-network model + metric definitions for US-130 readmission.

This module is the single source of truth for the estimator and metric set used
by the federated app and the offline evaluation harness. The estimator is a
small scikit-learn MLPClassifier; federated rounds exchange and average layer
weights, never raw patient rows.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight

CLASS_NAMES = ["Not Readmitted", "Readmitted <30d"]

DEFAULT_HIDDEN_LAYER_SIZES = (12, 8)
DEFAULT_ACTIVATION = "relu"
DEFAULT_SOLVER = "adam"
DEFAULT_ALPHA = 0.0001
DEFAULT_LEARNING_RATE_INIT = 0.001
DEFAULT_BATCH_SIZE = 10
DEFAULT_CLASS_WEIGHT = "balanced"
DEFAULT_RANDOM_STATE = 42
DEFAULT_MAX_ITER = 150

# Maps each headline validation metric to the key produced by ``score``.
EVAL_SCORE_KEYS: dict[str, str] = {
    "val_auc": "auc",
    "val_auprc": "auprc",
    "val_f1": "f1",
    "val_precision": "precision",
    "val_recall": "recall",
    "val_accuracy": "accuracy",
    "val_brier": "brier",
}


def _hidden_layers(value: str | int | tuple[int, ...] | list[int]) -> tuple[int, ...]:
    """Parse app.yml-friendly hidden layer values like ``"32"`` or ``"64,32"``."""
    if isinstance(value, int):
        return (value,)
    if isinstance(value, (tuple, list)):
        layers = tuple(int(v) for v in value)
    else:
        layers = tuple(int(part.strip()) for part in str(value).split(",") if part.strip())
    if not layers or any(layer <= 0 for layer in layers):
        raise ValueError("hidden_layer_sizes must contain one or more positive integers.")
    return layers


def new_model(
    hidden_layer_sizes: str | int | tuple[int, ...] | list[int] = DEFAULT_HIDDEN_LAYER_SIZES,
    activation: str = DEFAULT_ACTIVATION,
    solver: str = DEFAULT_SOLVER,
    alpha: float = DEFAULT_ALPHA,
    learning_rate_init: float = DEFAULT_LEARNING_RATE_INIT,
    batch_size: int | str = DEFAULT_BATCH_SIZE,
    max_iter: int = DEFAULT_MAX_ITER,
    random_state: int = DEFAULT_RANDOM_STATE,
    warm_start: bool = True,
) -> MLPClassifier:
    """Construct the readmission estimator."""
    parsed_batch_size = "auto" if batch_size == "auto" else int(batch_size)
    return MLPClassifier(
        hidden_layer_sizes=_hidden_layers(hidden_layer_sizes),
        activation=activation,
        solver=solver,
        alpha=float(alpha),
        learning_rate_init=float(learning_rate_init),
        batch_size=parsed_batch_size,
        max_iter=max(1, int(max_iter)),
        random_state=random_state,
        warm_start=warm_start,
    )


def new_model_from_cfg(cfg: Any, max_iter: int, warm_start: bool = True) -> MLPClassifier:
    """Convenience for the app: build the estimator from a config-like object."""
    return new_model(
        hidden_layer_sizes=cfg.hidden_layer_sizes,
        activation=cfg.activation,
        solver=cfg.solver,
        alpha=cfg.alpha,
        learning_rate_init=cfg.learning_rate_init,
        batch_size=cfg.batch_size,
        max_iter=max_iter,
        random_state=cfg.random_state,
        warm_start=warm_start,
    )


def _sample_weight(y: np.ndarray, class_weight: str | None = DEFAULT_CLASS_WEIGHT):
    if class_weight in (None, "none"):
        return None
    if class_weight != "balanced":
        raise ValueError(f"Unsupported class_weight={class_weight!r}; use 'balanced' or 'none'.")
    return compute_sample_weight(class_weight="balanced", y=y)


def _init_for_features(model: MLPClassifier, n_features: int) -> MLPClassifier:
    """Public-API initialization so manual weight seeding remains sklearn-compatible."""
    if hasattr(model, "coefs_"):
        return model
    X0 = np.zeros((2, n_features), dtype=float)
    y0 = np.array([0, 1], dtype=int)
    old_max_iter = model.max_iter
    model.max_iter = 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)
        model.fit(X0, y0)
    model.max_iter = old_max_iter
    model.n_iter_ = 0
    return model


def fit_model(
    model: MLPClassifier,
    X: np.ndarray,
    y: np.ndarray,
    class_weight: str | None = DEFAULT_CLASS_WEIGHT,
) -> MLPClassifier:
    """Fit with the shared class-balancing policy and quiet expected max-iter warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        weights = _sample_weight(y, class_weight)
        try:
            model.fit(X, y, sample_weight=weights)
        except TypeError:
            if weights is not None:
                warnings.warn(
                    "This scikit-learn MLPClassifier does not support sample_weight; "
                    "continuing without balanced sample weights.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            model.fit(X, y)
    return model


def seed(model: MLPClassifier, weights: dict[str, list[np.ndarray]], n_features: int) -> MLPClassifier:
    """Initialize a model from aggregated global layer weights."""
    _init_for_features(model, n_features)
    coefs = [np.asarray(layer, dtype=float) for layer in weights["coefs"]]
    intercepts = [np.asarray(layer, dtype=float) for layer in weights["intercepts"]]
    expected = [tuple(c.shape) for c in model.coefs_]
    actual = [tuple(c.shape) for c in coefs]
    if expected != actual:
        raise ValueError(f"Aggregated MLP shape mismatch: expected {expected}, got {actual}.")
    model.coefs_ = coefs
    model.intercepts_ = intercepts
    model.n_iter_ = 0
    return model


def payload(model: MLPClassifier, features: list[str], n_samples: int) -> dict:
    """Local update sent to the aggregator: MLP weights plus the input feature schema."""
    return {
        "feature_columns": list(features),
        "coefs": [coef.tolist() for coef in model.coefs_],
        "intercepts": [intercept.tolist() for intercept in model.intercepts_],
        "n_samples": n_samples,
    }


def parse_global(aggregated: dict[str, Any]) -> tuple[list[str], dict[str, list[np.ndarray]]]:
    """Parse an aggregated global payload into feature names and layer weights."""
    features = [str(f) for f in aggregated.get("feature_columns", [])]
    coefs = [np.asarray(layer, dtype=float) for layer in aggregated.get("coefs", [])]
    intercepts = [np.asarray(layer, dtype=float) for layer in aggregated.get("intercepts", [])]
    if not coefs or len(coefs) != len(intercepts):
        raise ValueError("Aggregated payload must contain matching coefs/intercepts layers.")
    if coefs[0].shape[0] != len(features):
        raise ValueError("Mismatched input layer rows and feature count in aggregated payload.")
    return features, {"coefs": coefs, "intercepts": intercepts}


def align(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Project a preprocessed matrix onto ``features`` (global schema), zero-filling gaps."""
    return frame.reindex(columns=features, fill_value=0.0).values


def feature_importance_table(feature_names: list[str], model: MLPClassifier) -> pd.DataFrame:
    """Approximate feature importance from the signed mean of first-layer weights."""
    first_layer = np.asarray(model.coefs_[0], dtype=float)
    rows = [
        {"feature": name, "coefficient": float(weight)}
        for name, weight in zip(feature_names, first_layer.mean(axis=1))
    ]
    rows.append({"feature": "__bias_mean__", "coefficient": float(np.mean(model.intercepts_[0]))})
    return (
        pd.DataFrame(rows)
        .sort_values("coefficient", ascending=False)
        .reset_index(drop=True)
    )


def n_parameters(model: MLPClassifier) -> int:
    return int(sum(layer.size for layer in model.coefs_) + sum(layer.size for layer in model.intercepts_))


def score(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    """The standard classification metrics, reused everywhere."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, y_proba)),
        "auprc": float(average_precision_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_proba)),
    }


def evaluate(model: MLPClassifier, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Score a fitted model on (X, y) using the shared metric set."""
    return score(y, model.predict(X), model.predict_proba(X)[:, 1])
