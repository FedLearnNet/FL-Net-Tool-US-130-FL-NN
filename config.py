from pathlib import Path
from typing import Any

from pydantic.dataclasses import dataclass
from pyfedappwrap.learning.run_runfig import AppConfig, AppInputConfig, AppOutputConfig


@dataclass
class HospitalReadmissionConfig(AppConfig):
    target_column: str = "hospital_readmission"
    standardize: bool = True
    hidden_layer_sizes: str = "12,8"
    activation: str = "relu"
    max_iter: int = 150
    solver: str = "adam"
    alpha: float = 0.0001
    learning_rate_init: float = 0.001
    batch_size: str = "10"
    class_weight: str = "balanced"
    random_state: int = 42
    test_size: float = 0.2


@dataclass
class HospitalReadmissionInputConfig(AppInputConfig):
    data: Any = None


@dataclass
class HospitalReadmissionOutputConfig(AppOutputConfig):
    predictions: Any | None = None
    coefficients: Any | None = None
    explanation: Path | None = None
    report: str | None = None


@dataclass
class FederatedHospitalReadmissionConfig(HospitalReadmissionConfig):
    communication_id: str = "hospital-readmission"
    output_filename: str = "federated_readmission_predictions.csv"
    federated_rounds: int = 10
    local_max_iter: int = 15
    standardize: bool = False
