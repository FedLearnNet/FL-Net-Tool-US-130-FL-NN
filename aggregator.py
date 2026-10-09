from __future__ import annotations

import re
from typing import Any, Optional

import numpy as np

from pyfedappwrap.engine.federated import AppAggregator, FLNetMessageMetaDTO


def _round_from_meta(meta: Optional[FLNetMessageMetaDTO]) -> Optional[int]:
    """FLNetMessageMetaDTO carries no explicit round; the round is encoded in the
    communication id (e.g. 'hospital-readmission-round-2' or 'AUTOMATIC_COMM_ID_2').
    Return the trailing integer when present, otherwise None."""
    if meta is None or not meta.communication_id:
        return None
    match = re.search(r"(\d+)\D*$", meta.communication_id)
    return int(match.group(1)) if match else None


class MeanVectorAggregator(AppAggregator):
    def aggregate(
        self,
        data: list[Any],
        n_clients: int,
        meta: Optional[FLNetMessageMetaDTO] = None,
    ) -> Any:
        if not data:
            raise ValueError("Cannot aggregate an empty payload list.")

        if all(isinstance(item, dict) and "coefs" in item for item in data):
            return self._aggregate_mlp(data, n_clients, meta)

        return np.mean(np.stack(data), axis=0)

    @staticmethod
    def _aggregate_mlp(
        payloads: list[dict[str, Any]],
        n_clients: int,
        meta: Optional[FLNetMessageMetaDTO] = None,
    ) -> dict[str, Any]:
        feature_columns: list[str] = []
        seen: set[str] = set()
        total_samples = 0
        weighted_first_layer: dict[str, np.ndarray] = {}
        weighted_hidden_layers: list[np.ndarray] | None = None
        weighted_intercepts: list[np.ndarray] | None = None

        for payload in payloads:
            weight = int(payload.get("n_samples", 1))
            total_samples += weight
            features = [str(feature) for feature in payload.get("feature_columns", [])]
            coefs = [np.asarray(layer, dtype=float) for layer in payload.get("coefs", [])]
            intercepts = [np.asarray(layer, dtype=float) for layer in payload.get("intercepts", [])]
            if not coefs or len(coefs) != len(intercepts):
                raise ValueError("MLP payload must contain matching coefs/intercepts layers.")
            if len(features) != coefs[0].shape[0]:
                raise ValueError("MLP payload feature count does not match first layer rows.")

            for feature, weights in zip(features, coefs[0]):
                if feature not in seen:
                    seen.add(feature)
                    feature_columns.append(feature)
                weighted_first_layer[feature] = (
                    weighted_first_layer.get(feature, np.zeros_like(weights)) + weights * weight
                )

            hidden_layers = [layer * weight for layer in coefs[1:]]
            if weighted_hidden_layers is None:
                weighted_hidden_layers = hidden_layers
            else:
                if len(weighted_hidden_layers) != len(hidden_layers):
                    raise ValueError("All MLP payloads must use the same hidden-layer architecture.")
                weighted_hidden_layers = [old + new for old, new in zip(weighted_hidden_layers, hidden_layers)]

            if weighted_intercepts is None:
                weighted_intercepts = [layer * weight for layer in intercepts]
            else:
                if len(weighted_intercepts) != len(intercepts):
                    raise ValueError("All MLP payloads must use the same number of intercept layers.")
                weighted_intercepts = [old + layer * weight for old, layer in zip(weighted_intercepts, intercepts)]

        divisor = total_samples or n_clients
        if weighted_hidden_layers is None or weighted_intercepts is None:
            raise ValueError("Cannot aggregate MLP payloads without layer weights.")
        return {
            "feature_columns": feature_columns,
            "coefs": [
                [(weighted_first_layer.get(f, np.zeros_like(next(iter(weighted_first_layer.values())))) / divisor).tolist()
                 for f in feature_columns],
                *[(layer / divisor).tolist() for layer in weighted_hidden_layers],
            ],
            "intercepts": [(layer / divisor).tolist() for layer in weighted_intercepts],
            "n_clients": n_clients,
            "n_samples": total_samples,
            "round": _round_from_meta(meta),
            "communication_id": meta.communication_id if meta else None,
        }
