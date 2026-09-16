"""End-to-end declarative experiment orchestration."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .backends import BackendRouter, ExecutionResult
from .benchmark import ResourceReport, resource_report
from .capabilities import CapabilityRegistry
from .circuits import CircuitFactory
from .ir import CircuitIR
from .models import ModelFactory
from .specs import ExperimentSpec
from .tracking import RunManifest, build_manifest
from .training import EpochMetrics, TorchTrainer, TrainingRun


@dataclass(slots=True)
class PreparedExperiment:
    spec: ExperimentSpec
    circuit: CircuitIR
    resources: ResourceReport
    warnings: list[str]


class ExperimentRunner:
    def __init__(self) -> None:
        self.circuits = CircuitFactory()
        self.backends = BackendRouter()
        self.models = ModelFactory()
        self.capabilities = CapabilityRegistry()

    def prepare(self, spec: ExperimentSpec | str | Path) -> PreparedExperiment:
        resolved = ExperimentSpec.from_yaml(spec) if isinstance(spec, (str, Path)) else spec
        circuit = self.circuits.build(resolved.circuit)
        warnings = self.capabilities.validate(circuit, resolved.backend)
        return PreparedExperiment(resolved, circuit, resource_report(circuit), warnings)

    def run_circuit(
        self,
        prepared: PreparedExperiment,
        *,
        inputs: Iterable[float],
        weights: Iterable[float],
    ) -> ExecutionResult:
        backend = self.backends.create(prepared.spec.backend)
        return backend.run(prepared.circuit, inputs=inputs, weights=weights)

    def build_model(self, prepared: PreparedExperiment) -> Any:
        """Build the declared model with the experiment seed already applied.

        Passing ``spec.seed`` here is what makes two builds of the same
        specification start from identical weights.  Seeding used to happen
        inside the training loop, by which point PyTorch had already drawn the
        initial parameters.
        """

        return self.models.build(
            prepared.circuit,
            prepared.spec.model,
            prepared.spec.backend,
            seed=prepared.spec.seed,
        )

    def train(
        self,
        prepared: PreparedExperiment,
        train_loader: Iterable[Any],
        validation_loader: Iterable[Any] | None = None,
        callbacks: Iterable[Callable[[EpochMetrics], None]] = (),
        device: str = "cpu",
        *,
        model: Any | None = None,
    ) -> TrainingRun:
        """Build (or accept) a model and execute the declared TrainingSpec."""

        trainable = model if model is not None else self.build_model(prepared)
        trainer = TorchTrainer(prepared.spec.training, seed=prepared.spec.seed)
        result = trainer.fit(
            trainable,
            train_loader,
            validation_loader=validation_loader,
            callbacks=callbacks,
            device=device,
        )
        result.metadata.update(
            {
                "experiment": prepared.spec.name,
                "template": prepared.spec.model.template,
                "backend": str(prepared.spec.backend.engine),
            }
        )
        return TrainingRun.from_result(result, model=trainable)

    def manifest(
        self,
        prepared: PreparedExperiment,
        *,
        metrics: dict[str, Any] | None = None,
        dataset_paths: Iterable[str | Path] = (),
    ) -> RunManifest:
        return build_manifest(
            prepared.spec,
            prepared.circuit,
            metrics=metrics,
            dataset_paths=dataset_paths,
        )


__all__ = ["PreparedExperiment", "ExperimentRunner", "TrainingRun"]
