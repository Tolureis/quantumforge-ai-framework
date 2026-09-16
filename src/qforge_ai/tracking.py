"""Reproducibility manifests and lightweight local experiment tracking."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from .backends._optional import require
from .ir import CircuitIR
from .specs import ExperimentSpec, to_primitive
from .training import EpochMetrics, TrainingResult


@dataclass(frozen=True, slots=True)
class DatasetFingerprint:
    """Location-independent dataset digest with separate provenance paths."""

    content_sha256: str
    source_paths: tuple[str, ...]
    file_count: int


@dataclass(slots=True)
class RunManifest:
    experiment: dict[str, Any]
    circuit_summary: dict[str, Any]
    created_at: str
    python_version: str
    platform: str
    package_versions: dict[str, str | None]
    # Backward-compatible alias for dataset.content_sha256. It no longer
    # includes source paths in the digest.
    dataset_hash: str | None = None
    dataset: DatasetFingerprint | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(asdict(self), indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )


def build_manifest(
    experiment: ExperimentSpec,
    circuit: CircuitIR,
    *,
    dataset_paths: Iterable[str | Path] = (),
    metrics: dict[str, Any] | None = None,
) -> RunManifest:
    resolved_dataset_paths = tuple(Path(path) for path in dataset_paths)
    dataset = (
        fingerprint_files(resolved_dataset_paths) if resolved_dataset_paths else None
    )
    packages = (
        "quantumforge-ai",
        "qiskit",
        "qiskit-aer",
        "qiskit-machine-learning",
        "pennylane",
        "pennylane-lightning",
        "torch",
        "numpy",
    )
    return RunManifest(
        # Best-effort: a manifest records what a run looked like and must not
        # abort the run over an option object it cannot render.
        experiment=cast(dict[str, Any], to_primitive(experiment, strict=False)),
        circuit_summary=circuit.summary(),
        created_at=datetime.now(UTC).isoformat(),
        python_version=platform.python_version(),
        platform=platform.platform(),
        package_versions={name: _version(name) for name in packages},
        dataset_hash=dataset.content_sha256 if dataset is not None else None,
        dataset=dataset,
        metrics=metrics or {},
    )


def hash_files(paths: Iterable[str | Path]) -> str:
    """Hash file contents only; source locations are intentionally excluded."""

    return fingerprint_files(paths).content_sha256


def fingerprint_files(paths: Iterable[str | Path]) -> DatasetFingerprint:
    resolved = tuple(Path(path) for path in paths)
    file_digests: list[str] = []
    for path in resolved:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        file_digests.append(digest.hexdigest())
    if len(file_digests) == 1:
        content_sha256 = file_digests[0]
    else:
        aggregate = hashlib.sha256(b"quantumforge-dataset-content-v2\0")
        for file_digest in sorted(file_digests):
            aggregate.update(bytes.fromhex(file_digest))
        content_sha256 = aggregate.hexdigest()
    return DatasetFingerprint(
        content_sha256=content_sha256,
        source_paths=tuple(sorted(str(path) for path in resolved)),
        file_count=len(resolved),
    )


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


class MLflowTracker:
    """Optional MLflow run context and ``TorchTrainer`` epoch callback."""

    def __init__(
        self,
        experiment_name: str | ExperimentSpec = "QuantumForge AI",
        *,
        run_name: str | None = None,
        tracking_uri: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> None:
        self._default_spec = experiment_name if isinstance(experiment_name, ExperimentSpec) else None
        self.experiment_name = (
            experiment_name.name if isinstance(experiment_name, ExperimentSpec) else experiment_name
        )
        if not self.experiment_name.strip():
            raise ValueError("MLflow experiment_name cannot be empty.")
        self.run_name = run_name
        self.tracking_uri = tracking_uri
        self.tags = tags or {}
        self._mlflow: Any | None = None
        self._run: Any | None = None

    def __enter__(self) -> MLflowTracker:
        if self._run is not None:
            return self
        return self.start_run()

    def start_run(
        self,
        run_name: str | None = None,
        *,
        tags: dict[str, str] | None = None,
    ) -> MLflowTracker:
        """Start a run explicitly (the 1.1 usage) or via a context manager."""

        if self._run is not None:
            raise RuntimeError("An MLflowTracker run is already active.")
        mlflow = require("mlflow", package="mlflow", extra="tracking", feature="MLflow tracking")
        if self.tracking_uri is not None:
            mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        self._run = mlflow.start_run(
            run_name=run_name if run_name is not None else self.run_name,
            tags={**self.tags, **(tags or {})},
        )
        self._mlflow = mlflow
        return self

    @property
    def active_run(self) -> Any | None:
        return self._run

    def log_experiment(self, spec: ExperimentSpec | None = None) -> None:
        mlflow = self._require_active()
        resolved = spec or self._default_spec
        if resolved is None:
            raise ValueError("An ExperimentSpec must be given to log_experiment.")
        flat = _flatten(cast(dict[str, Any], to_primitive(resolved, strict=False)))
        mlflow.log_params({key: str(value)[:500] for key, value in flat.items()})

    def log_params(self, parameters: dict[str, Any]) -> None:
        """Compatibility passthrough for manually prepared parameters."""

        self._require_active().log_params(
            {key: str(value)[:500] for key, value in _flatten(parameters).items()}
        )

    def log_metrics(self, metrics: dict[str, float], *, step: int | None = None) -> None:
        """Compatibility passthrough for manually prepared metrics."""

        mlflow = self._require_active()
        if step is None:
            mlflow.log_metrics(metrics)
        else:
            mlflow.log_metrics(metrics, step=step)

    def __call__(self, metrics: EpochMetrics) -> None:
        mlflow = self._require_active()
        values = {
            "train_loss": metrics.train_loss,
            "learning_rate": metrics.learning_rate,
            "epoch_runtime_seconds": metrics.runtime_seconds,
        }
        if metrics.validation_loss is not None:
            values["validation_loss"] = metrics.validation_loss
        mlflow.log_metrics(values, step=metrics.epoch)

    def log_result(self, result: TrainingResult) -> None:
        mlflow = self._require_active()
        mlflow.log_metrics(
            {
                "best_epoch": float(result.best_epoch),
                "stopped_early": float(result.stopped_early),
                **(
                    {"best_validation_loss": result.best_validation_loss}
                    if result.best_validation_loss is not None
                    else {}
                ),
            }
        )

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        status = "FAILED" if exc_type is not None else "FINISHED"
        self.end_run(status=status)

    def end_run(self, *, status: str = "FINISHED") -> None:
        """End an explicitly started run; safe when no run is active."""

        if self._mlflow is not None and self._run is not None:
            self._mlflow.end_run(status=status)
        self._run = None
        self._mlflow = None

    def _require_active(self) -> Any:
        if self._mlflow is None or self._run is None:
            raise RuntimeError("MLflowTracker must be used inside a 'with' block.")
        return self._mlflow


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten(item, child))
        return result
    if isinstance(value, list):
        return {prefix: ",".join(map(str, value))}
    return {prefix: value}


__all__ = [
    "DatasetFingerprint",
    "RunManifest",
    "MLflowTracker",
    "build_manifest",
    "hash_files",
    "fingerprint_files",
]
