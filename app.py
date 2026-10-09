import os
import time
from dataclasses import fields
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from pyfedappwrap.learning.federated import BaseFederatedApp
from pyfedappwrap.learning.metrics import EVAL_METRICS, MetricName
from pyfedappwrap.learning.privacy import secure_count
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from config import (
    FederatedHospitalReadmissionConfig,
    HospitalReadmissionInputConfig,
    HospitalReadmissionOutputConfig,
)
from helper import (
    build_prediction_df,
    generate_explanation_html,
    preprocess_diabetic,
)
# Estimator + metrics live in the shared model module so the app, the aggregator and the offline
# evaluation harness all use the identical model and scoring.
from model import (
    CLASS_NAMES,
    align as _align,
    feature_importance_table as _feature_importance_table,
    fit_model as _fit_model,
    new_model_from_cfg,
    n_parameters as _n_parameters,
    parse_global as _parse_global,
    payload as _payload,
    score as _score,
    seed as _seed,
)

# Maps each selectable evaluation metric to the key produced by score(), so the per-round emission is
# driven by the shared MetricName contract instead of hand-written strings.
_VAL_SCORE_KEY: dict[MetricName, str] = {
    MetricName.VAL_AUC: "auc",
    MetricName.VAL_AUPRC: "auprc",
    MetricName.VAL_F1: "f1",
    MetricName.VAL_PRECISION: "precision",
    MetricName.VAL_RECALL: "recall",
    MetricName.VAL_ACCURACY: "accuracy",
    MetricName.VAL_BRIER: "brier",
}


# --- Module-level helpers (pure, no app state) ------------------------------

def _round_communication_id(base: str, round_nr: int, n_rounds: int) -> str:
    if n_rounds <= 1:
        return base
    if "{round}" in base:
        return base.format(round=round_nr)
    return f"{base}-round-{round_nr}"


def _new_model(cfg, max_iter: int):
    """Federated rounds seed layer weights then fit locally, so warm_start must be on."""
    return new_model_from_cfg(cfg, max_iter, warm_start=True)


class FederatedHospitalReadmissionClient(
    BaseFederatedApp[
        FederatedHospitalReadmissionConfig,
        HospitalReadmissionInputConfig,
        HospitalReadmissionOutputConfig,
    ]
):
    def __init__(self):
        super().__init__()
        self.model: Optional[MLPClassifier] = None
        self.feature_columns: list[str] = []
        self.target_column: str = "hospital_readmission"

    # --- Test-mode fixtures -------------------------------------------------

    def get_test_data(self) -> Optional[dict[str, Path]]:
        return {"data": Path("data/test_us130_wide.csv")}

    def get_test_hyperparams(self) -> dict[str, Any]:
        cfg = FederatedHospitalReadmissionConfig()
        return {field.name: getattr(cfg, field.name) for field in fields(cfg)}

    # --- Training -----------------------------------------------------------

    def run_train(self, data: HospitalReadmissionInputConfig) -> HospitalReadmissionOutputConfig:
        cfg = self._cfg()
        self.target_column = cfg.target_column
        if data.data is None:
            raise ValueError("Training data is required.")

        # Preprocess once; realign to the (possibly wider) global feature set per round via reindex.
        X, y, local_features = preprocess_diabetic(data.data, target_col=cfg.target_column, is_training=True)
        feature_frame = pd.DataFrame(X, columns=local_features)
        train_idx, val_idx = train_test_split(
            np.arange(len(y)), test_size=cfg.test_size, random_state=cfg.random_state, stratify=y,
        )

        # Privacy-safe cohort sizes & class balance, emitted up-front (x=0) so the evaluation populates
        # even if a run is interrupted before the final round.
        self._emit_cohort_descriptors(X, y, train_idx, val_idx)

        # Institution-local baseline: a standalone model trained to convergence on this site's own
        # data and evaluated on the SAME validation split used for the federated model below. Emitting
        # it as local_val_* lets the global backend compare local vs federated on identical held-out
        # rows (a clean paired comparison), instead of using a half-trained round-1 proxy.
        self._emit_local_baseline(cfg, X, y, train_idx, val_idx)

        n_rounds = max(1, int(cfg.federated_rounds))
        features = local_features
        global_weights: Optional[dict[str, list[np.ndarray]]] = None
        history: list[dict[str, float]] = []

        for round_nr in range(1, n_rounds + 1):
            round_start = time.monotonic()
            X_round = _align(feature_frame, features)
            local = _new_model(cfg, cfg.local_max_iter)
            if global_weights is not None:
                _seed(local, global_weights, len(features))
            _fit_model(local, X_round[train_idx], y[train_idx], cfg.class_weight)

            comm_start = time.monotonic()
            response = self.communicator.aggregate(
                _payload(local, features, len(train_idx)),
                "mean",  # aggregator key registered in main.py
                communication_id=_round_communication_id(cfg.communication_id, round_nr, n_rounds),
                data_type=dict,
            )
            comm_seconds = time.monotonic() - comm_start
            features, global_weights = _parse_global(response.data)
            self.feature_columns = features
            self.model = _seed(_new_model(cfg, cfg.local_max_iter), global_weights, len(features))

            tr, va = self._score_train_val(feature_frame, y, train_idx, val_idx)
            history.append({"x": float(round_nr), "train_accuracy": tr["accuracy"],
                            "val_accuracy": va["accuracy"], "train_auc": tr["auc"], "val_auc": va["auc"]})
            # n_params is the per-round payload size (weights + biases) — independent of cohort
            # size, the scalability claim the evaluation makes for dimension D4.
            self._emit_round_metrics(round_nr, tr, va, comm_seconds,
                                     time.monotonic() - round_start, n_params=_n_parameters(self.model))

        return self._build_output(feature_frame, y, train_idx, val_idx, history)

    # --- Prediction ---------------------------------------------------------

    def run_prediction(self, data: HospitalReadmissionInputConfig) -> HospitalReadmissionOutputConfig:
        self._load()
        df = data.data
        X, y_true, _ = preprocess_diabetic(
            df, target_col=self.target_column, is_training=False, feature_columns=self.feature_columns,
        )
        ids = (df["encounter_identifier"].tolist()
               if "encounter_identifier" in df.columns else list(range(len(df))))

        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)[:, 1]
        readmit_rate = float(y_pred.mean())
        self.send_metric("predicted_readmission_rate", readmit_rate, 0)

        metrics: dict[str, Any] = {
            "n_samples_prediction": len(y_pred),
            "n_features": X.shape[1],
            "predicted_readmission_rate": readmit_rate,
        }
        cm, report_str, val_fpr, val_tpr = [[0, 0], [0, 0]], "", None, None
        report = f"Prediction complete — {len(y_pred)} samples, rate: {readmit_rate:.1%}"

        if y_true is not None:
            s = _score(y_true, y_pred, y_proba)
            metrics.update({f"prediction_{k}": v for k, v in s.items()})
            val_fpr, val_tpr, _ = roc_curve(y_true, y_proba)
            cm = confusion_matrix(y_true, y_pred).tolist()
            report_str = classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0)
            report = (f"Prediction complete — {len(y_pred)} samples\n"
                      f"Readmission rate: {readmit_rate:.1%}\n"
                      f"  Accuracy: {s['accuracy']:.4f}  AUC: {s['auc']:.4f}  F1: {s['f1']:.4f}\n")

        coef_df = _feature_importance_table(self.feature_columns, self.model)
        explanation = self._write_explanation(
            f"federated_prediction_{self._safe_name()}.html",
            coef_df, metrics, cm, report_str, history=[], val_fpr=val_fpr, val_tpr=val_tpr,
        )
        return HospitalReadmissionOutputConfig(
            predictions=build_prediction_df(ids, y_pred, y_proba),
            coefficients=coef_df, explanation=explanation, report=report,
        )

    # --- Internal helpers ---------------------------------------------------

    def _cfg(self) -> FederatedHospitalReadmissionConfig:
        return self.config or FederatedHospitalReadmissionConfig()

    def _emit_cohort_descriptors(self, X, y, train_idx, val_idx) -> None:
        """Emit privacy-safe cohort sizes and validation class balance once, up-front. Counts are
        rounded with the platform's secure-count policy (see pyfedappwrap.learning.privacy) so exact
        patient numbers never leave the clinic; the global backend weights the federated aggregate by
        these secured validation sizes."""
        self.send_metric(MetricName.N_SAMPLES_TRAIN, secure_count(len(train_idx)), 0, "round")
        self.send_metric(MetricName.N_SAMPLES_VAL, secure_count(len(val_idx)), 0, "round")
        self.send_metric(MetricName.N_FEATURES, X.shape[1], 0, "round")
        self.send_metric(MetricName.VAL_POSITIVE_RATE, float(y[val_idx].mean()), 0, "round")

    def _emit_round_metrics(self, round_nr, tr, va, comm_seconds, round_seconds, n_params) -> None:
        """Emit the full evaluation metric set (the MetricName contract) for one federated round."""
        for metric in EVAL_METRICS:
            self.send_metric(metric, va[_VAL_SCORE_KEY[metric]], round_nr, "round")
        self.send_metric(MetricName.TRAIN_AUC, tr["auc"], round_nr, "round")
        self.send_metric(MetricName.TRAIN_ACCURACY, tr["accuracy"], round_nr, "round")
        self.send_metric(MetricName.COMM_SECONDS, comm_seconds, round_nr, "round")
        self.send_metric(MetricName.ROUND_SECONDS, round_seconds, round_nr, "round")
        self.send_metric(MetricName.N_PARAMS, n_params, round_nr, "round")

    def _emit_local_baseline(self, cfg, X, y, train_idx, val_idx) -> None:
        """Train a standalone local model (full local budget, no aggregation) on this site's own data
        and score it on the held-out validation split, emitting local_val_* metrics."""
        model = _new_model(cfg, cfg.max_iter)
        _fit_model(model, X[train_idx], y[train_idx], cfg.class_weight)
        scores = _score(
            y[val_idx],
            model.predict(X[val_idx]),
            model.predict_proba(X[val_idx])[:, 1],
        )
        for metric in EVAL_METRICS:
            # local_val_auc, local_val_f1, … — the lower-reference counterpart of each val_* metric.
            self.send_metric(metric.local(), scores[_VAL_SCORE_KEY[metric]], 0, "round")

    def _score_train_val(self, frame, y, train_idx, val_idx) -> tuple[dict[str, float], dict[str, float]]:
        X = _align(frame, self.feature_columns)
        tr = _score(y[train_idx], self.model.predict(X[train_idx]),
                    self.model.predict_proba(X[train_idx])[:, 1])
        va = _score(y[val_idx], self.model.predict(X[val_idx]),
                    self.model.predict_proba(X[val_idx])[:, 1])
        return tr, va

    def _train_val_metrics(self, frame, y, train_idx, val_idx) -> dict[str, float]:
        tr, va = self._score_train_val(frame, y, train_idx, val_idx)
        return {
            "train_accuracy": tr["accuracy"], "train_auc": tr["auc"], "train_auprc": tr["auprc"],
            "val_accuracy": va["accuracy"], "val_auc": va["auc"], "val_auprc": va["auprc"],
            "val_precision": va["precision"], "val_recall": va["recall"], "val_f1": va["f1"],
            "val_brier": va["brier"],
        }

    def _build_output(self, frame, y, train_idx, val_idx, history) -> HospitalReadmissionOutputConfig:
        X = _align(frame, self.feature_columns)
        y_val = y[val_idx]
        val_pred = self.model.predict(X[val_idx])
        val_proba = self.model.predict_proba(X[val_idx])[:, 1]

        metrics = self._train_val_metrics(frame, y, train_idx, val_idx)
        # Secure (privacy-rounded) counts for the saved report and explanation — the raw cohort sizes
        # never leave the clinic. The per-round metric series (emitted via the MetricName contract) and
        # the up-front secure descriptors already carry everything the global backend reads.
        metrics.update(
            n_samples_train=secure_count(len(train_idx)), n_samples_val=secure_count(len(val_idx)),
            n_features=X.shape[1], federated_rounds=len(history),
        )

        coef_df = _feature_importance_table(self.feature_columns, self.model)
        val_fpr, val_tpr, _ = roc_curve(y_val, val_proba)
        explanation = self._write_explanation(
            f"federated_explanation_{self._safe_name()}.html",
            coef_df, metrics, confusion_matrix(y_val, val_pred).tolist(),
            classification_report(y_val, val_pred, target_names=CLASS_NAMES, zero_division=0),
            history=history, val_fpr=val_fpr, val_tpr=val_tpr,
        )

        pred_df = build_prediction_df(val_idx.tolist(), val_pred, val_proba)
        report = (
            f"Federated Training Summary\n{'=' * 40}\n"
            f"Samples: {metrics['n_samples_train']} train / {metrics['n_samples_val']} val\n"
            f"Features: {metrics['n_features']}, Rounds: {metrics['federated_rounds']}\n"
            f"Val — Acc: {metrics['val_accuracy']:.4f}, AUC: {metrics['val_auc']:.4f}, "
            f"F1: {metrics['val_f1']:.4f}\n"
        )
        if self.logger:
            self.logger.info(report)

        self._save()
        self.save_local_csv(pred_df, self._cfg().output_filename)
        return HospitalReadmissionOutputConfig(
            predictions=pred_df, coefficients=coef_df, explanation=explanation, report=report,
        )

    @staticmethod
    def _write_explanation(filename, coef_df, metrics, cm, report_str, history, val_fpr, val_tpr) -> Path:
        # Relative path on purpose: the framework's save_results moves it into the output volume.
        path = Path(filename)
        path.write_text(
            generate_explanation_html(coef_df, metrics, cm, report_str,
                                      training_history=history, top_k=20,
                                      val_fpr=val_fpr, val_tpr=val_tpr),
            encoding="utf-8",
        )
        return path

    def _safe_name(self) -> str:
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(self.app_id or "client"))

    def _model_path(self) -> Path:
        return Path(os.environ.get("MODEL_DIR", "./model/")) / "model.joblib"

    def _save(self) -> str:
        path = self._model_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": self.model, "feature_columns": self.feature_columns,
             "target_column": self.target_column},
            path,
        )
        return path.name

    def _load(self) -> None:
        if self.model is not None:
            return
        artifact = joblib.load(self._model_path())
        self.model = artifact["model"]
        self.feature_columns = artifact["feature_columns"]
        self.target_column = artifact["target_column"]
