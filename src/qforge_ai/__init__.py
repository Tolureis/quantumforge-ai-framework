"""QuantumForge AI public package API."""

from .automl import AutoMLStudy, OptunaSearch, QuantumSearchSpace, SearchParameter
from .benchmark import compare_engines, resource_report
from .capabilities import CapabilityRegistry
from .catalog import (
    capability_report,
    catalog,
    executable_components,
    is_executable,
    list_components,
)
from .experiment import ExperimentRunner, PreparedExperiment
from .ir import CircuitIR, Measurement, Observable, Operation, ParameterRef
from .registry import ComponentDescriptor, ComponentRegistry
from .specs import (
    AnsatzSpec,
    BackendSpec,
    CircuitSpec,
    ComponentStatus,
    EncodingSpec,
    Engine,
    ExperimentSpec,
    GradientMethod,
    MeasurementSpec,
    ModelMode,
    ModelSpec,
    NoiseSpec,
    TrainingSpec,
)
from .tracking import MLflowTracker
from .training import EpochMetrics, TorchTrainer, TrainingResult, TrainingRun
from .verification import (
    VerificationResult,
    verification_report,
    verify_catalog,
    verify_component,
)

__version__ = "1.2.5"

__all__ = [
    "__version__",
    "catalog",
    "list_components",
    "capability_report",
    "executable_components",
    "is_executable",
    "VerificationResult",
    "verify_component",
    "verify_catalog",
    "verification_report",
    "ExperimentRunner",
    "PreparedExperiment",
    "CapabilityRegistry",
    "CircuitIR",
    "Operation",
    "Measurement",
    "Observable",
    "ParameterRef",
    "ComponentDescriptor",
    "ComponentRegistry",
    "CircuitSpec",
    "EncodingSpec",
    "AnsatzSpec",
    "MeasurementSpec",
    "NoiseSpec",
    "BackendSpec",
    "ModelSpec",
    "TrainingSpec",
    "ExperimentSpec",
    "Engine",
    "ModelMode",
    "GradientMethod",
    "ComponentStatus",
    "resource_report",
    "compare_engines",
    "TorchTrainer",
    "TrainingResult",
    "TrainingRun",
    "EpochMetrics",
    "MLflowTracker",
    "SearchParameter",
    "QuantumSearchSpace",
    "AutoMLStudy",
    "OptunaSearch",
]
