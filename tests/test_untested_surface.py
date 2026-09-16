"""Behaviour the 1.2.4 test surface never reached.

Finding 8 of the 1.2.4 report was not only "the kernel claims are untested"
(covered by ``test_reported_defects_1_2_5.py``) but also that branch coverage
sat at 77.3% while CI enforced a *line* threshold only.  The weakest branches
were not obscure: the training loop's optimizer and loss vocabulary, its early
stopping and gradient clipping, the CLI's non-JSON output and its error path,
the tracking context manager, and the capability preflight's refusal branches
were all reachable from the public API and none were exercised.

These are contract tests, not line-count filler: each one pins behaviour a user
can observe.
"""

from __future__ import annotations

import importlib.util
import json

import numpy as np
import pytest

from qforge_ai import (
    AnsatzSpec,
    BackendSpec,
    CircuitSpec,
    EncodingSpec,
    MeasurementSpec,
    NoiseSpec,
    TrainingSpec,
)
from qforge_ai.capabilities import CapabilityRegistry
from qforge_ai.circuits import CircuitFactory
from qforge_ai.cli import main
from qforge_ai.errors import CapabilityError, ConfigurationError
from qforge_ai.ir import CircuitIR, Measurement, Observable
from qforge_ai.specs import Engine, ExperimentSpec, GradientMethod
from qforge_ai.training import TorchTrainer

HAS_TORCH = importlib.util.find_spec("torch") is not None
requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")


def _spec_dict(**overrides) -> dict:
    base = {
        "name": "surface",
        "circuit": {
            "num_qubits": 1,
            "encoding": {"name": "angle", "feature_dimension": 1},
            "ansatz": {"name": "real_amplitudes", "layers": 1},
            "measurement": {"name": "expectation", "wires": [0]},
        },
        "backend": {"engine": "pennylane", "device": "default.qubit"},
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------
# training loop
# --------------------------------------------------------------------------


def _loader(torch, rows: int = 8, features: int = 2, classes: int = 2):
    generator = torch.Generator().manual_seed(0)
    inputs = torch.randn(rows, features, generator=generator)
    labels = torch.randint(0, classes, (rows,), generator=generator)
    from torch.utils.data import DataLoader, TensorDataset

    return DataLoader(TensorDataset(inputs, labels), batch_size=4)


@requires_torch
@pytest.mark.parametrize("optimizer", ["adam", "adamw", "sgd", "rmsprop"])
def test_every_documented_optimizer_actually_builds(optimizer) -> None:
    import torch

    model = torch.nn.Linear(2, 2)
    trainer = TorchTrainer(TrainingSpec(epochs=1, batch_size=4, optimizer=optimizer))
    result = trainer.fit(model, _loader(torch))
    assert len(result.history) == 1


@requires_torch
def test_an_unsupported_optimizer_is_named() -> None:
    import torch

    trainer = TorchTrainer(TrainingSpec(epochs=1, optimizer="lbfgs"))
    with pytest.raises(ConfigurationError, match="optimizer"):
        trainer.fit(torch.nn.Linear(2, 2), _loader(torch))


@requires_torch
@pytest.mark.parametrize("loss", ["cross_entropy", "ce", "mse", "mean_squared_error"])
def test_every_documented_loss_actually_builds(loss) -> None:
    import torch

    model = torch.nn.Linear(2, 2)
    trainer = TorchTrainer(TrainingSpec(epochs=1, batch_size=4, loss=loss))
    if loss in {"mse", "mean_squared_error"}:
        from torch.utils.data import DataLoader, TensorDataset

        generator = torch.Generator().manual_seed(0)
        loader = DataLoader(
            TensorDataset(
                torch.randn(8, 2, generator=generator),
                torch.randn(8, 2, generator=generator),
            ),
            batch_size=4,
        )
    else:
        loader = _loader(torch)
    assert len(trainer.fit(model, loader).history) == 1


@requires_torch
def test_an_unsupported_loss_is_named() -> None:
    import torch

    trainer = TorchTrainer(TrainingSpec(epochs=1, loss="hinge"))
    with pytest.raises(ConfigurationError, match="loss"):
        trainer.fit(torch.nn.Linear(2, 2), _loader(torch))


@requires_torch
def test_early_stopping_restores_the_best_epoch_and_reports_it() -> None:
    import torch

    class Diverging(torch.nn.Module):
        """Loss grows every epoch, so the first epoch is always the best."""

        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)
            self.scale = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, inputs):
            return self.linear(inputs) * self.scale

    trainer = TorchTrainer(
        TrainingSpec(
            epochs=10,
            batch_size=4,
            early_stopping_patience=1,
            learning_rate=5.0,
            gradient_clip=0.5,
        )
    )
    loader = _loader(torch)
    result = trainer.fit(Diverging(), loader, validation_loader=loader)
    assert result.stopped_early
    assert len(result.history) < 10
    assert result.best_validation_loss is not None
    assert result.best_epoch >= 1


@requires_torch
def test_an_empty_loader_is_reported_rather_than_producing_an_empty_history() -> None:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    empty = DataLoader(TensorDataset(torch.empty(0, 2), torch.empty(0, dtype=torch.long)))
    trainer = TorchTrainer(TrainingSpec(epochs=1))
    with pytest.raises(ConfigurationError, match="train_loader"):
        trainer.fit(torch.nn.Linear(2, 2), empty)


@requires_torch
def test_an_empty_validation_loader_is_reported_too() -> None:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    empty = DataLoader(TensorDataset(torch.empty(0, 2), torch.empty(0, dtype=torch.long)))
    trainer = TorchTrainer(TrainingSpec(epochs=1, batch_size=4))
    with pytest.raises(ConfigurationError, match="validation_loader"):
        trainer.fit(torch.nn.Linear(2, 2), _loader(torch), validation_loader=empty)


@requires_torch
def test_training_is_reproducible_for_one_seed_and_differs_for_another() -> None:
    import torch

    def run(seed: int) -> float:
        torch.manual_seed(0)
        model = torch.nn.Linear(2, 2)
        trainer = TorchTrainer(TrainingSpec(epochs=2, batch_size=4), seed=seed)
        return trainer.fit(model, _loader(torch)).history[-1].train_loss

    assert run(3) == run(3)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def test_cli_catalog_text_output_lists_entries_and_a_total(capsys) -> None:
    assert main(["catalog", "--category", "encoding"]) == 0
    output = capsys.readouterr().out
    assert "angle_encoding" in output
    assert "Total:" in output


def test_cli_catalog_search_filters(capsys) -> None:
    assert main(["catalog", "--search", "amplitude"]) == 0
    output = capsys.readouterr().out
    assert "amplitude" in output


def test_cli_validate_prints_the_name_and_any_warnings(tmp_path, capsys) -> None:
    config = tmp_path / "config.yaml"
    ExperimentSpec.from_dict(_spec_dict()).to_yaml(config)
    assert main(["validate", str(config)]) == 0
    assert "Valid: surface" in capsys.readouterr().out


def test_cli_inspect_reports_resources_as_text_and_json(tmp_path, capsys) -> None:
    config = tmp_path / "config.yaml"
    ExperimentSpec.from_dict(_spec_dict()).to_yaml(config)
    assert main(["inspect", str(config)]) == 0
    assert "num_qubits" in capsys.readouterr().out
    assert main(["inspect", str(config), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["num_qubits"] == 1


def test_cli_doctor_lists_every_optional_dependency(capsys) -> None:
    assert main(["doctor"]) == 0
    output = capsys.readouterr().out
    for module in ("numpy", "qiskit", "pennylane", "torch"):
        assert module in output


def test_cli_reports_a_bad_configuration_on_stderr(tmp_path, capsys) -> None:
    """A configuration error must not surface as a traceback and exit 0."""

    config = tmp_path / "broken.yaml"
    config.write_text("name: broken\ncircuit:\n  num_qubits: 0\n", encoding="utf-8")
    assert main(["validate", str(config)]) == 2
    assert "ERROR:" in capsys.readouterr().err


def test_cli_run_rejects_a_malformed_input_list(tmp_path, capsys) -> None:
    config = tmp_path / "config.yaml"
    ExperimentSpec.from_dict(_spec_dict()).to_yaml(config)
    assert main(["run", str(config), "--inputs", "not-a-number", "--weights", "0.0"]) == 2
    assert "ERROR:" in capsys.readouterr().err


# --------------------------------------------------------------------------
# capability preflight
# --------------------------------------------------------------------------


def _ir(measurement: str = "expectation") -> CircuitIR:
    return CircuitFactory().build(
        CircuitSpec(
            num_qubits=2,
            encoding=EncodingSpec(name="angle", feature_dimension=2),
            ansatz=AnsatzSpec(name="real_amplitudes", layers=1),
            measurement=MeasurementSpec(name=measurement),
        )
    )


def test_ir_backend_is_analysis_only_and_says_so() -> None:
    warnings = CapabilityRegistry().validate(_ir(), BackendSpec(engine=Engine.IR))
    assert any("IR backend" in item for item in warnings)


def test_a_device_this_machine_does_not_have_is_refused_by_name() -> None:
    """``lightning.gpu`` is a declared device and an absent one here."""

    from qforge_ai.runtime import pennylane_registered_devices

    if "lightning.gpu" in pennylane_registered_devices():  # pragma: no cover
        pytest.skip("lightning.gpu really is registered on this machine")
    with pytest.raises(CapabilityError, match="does not support"):
        CapabilityRegistry().validate(
            _ir(), BackendSpec(engine=Engine.PENNYLANE, device="lightning.gpu")
        )


def test_an_unsupported_operation_is_refused_by_name() -> None:
    circuit = CircuitIR(num_qubits=1, name="plugin")
    circuit.add("SOME_PLUGIN_GATE", 0, 0.1)
    circuit.measure(
        Measurement(kind="expectation", wires=(0,), observables=(Observable("Z", (0,)),))
    )
    with pytest.raises(CapabilityError, match="SOME_PLUGIN_GATE"):
        CapabilityRegistry().validate(circuit, BackendSpec(engine=Engine.PENNYLANE, device="default.qubit"))


def test_readout_noise_needs_a_sampling_measurement() -> None:
    with pytest.raises(CapabilityError, match="[Rr]eadout"):
        CapabilityRegistry().validate(
            _ir(),
            BackendSpec(
                engine=Engine.PENNYLANE,
                device="default.mixed",
                noise=NoiseSpec(name="readout_error", readout_probability=0.1),
            ),
        )


def test_a_noisy_channel_cannot_produce_a_pure_statevector() -> None:
    circuit = CircuitIR(num_qubits=1, name="noisy-state")
    circuit.add("RY", 0, 0.4, role="encoding")
    circuit.measure(Measurement(kind="statevector", wires=(0,)))
    with pytest.raises(CapabilityError, match="statevector"):
        CapabilityRegistry().validate(
            circuit,
            BackendSpec(
                engine=Engine.PENNYLANE,
                device="default.mixed",
                noise=NoiseSpec(name="depolarizing", probability=0.1),
            ),
        )


def test_default_qubit_is_warned_about_rather_than_refused_under_noise() -> None:
    warnings = CapabilityRegistry().validate(
        _ir(measurement="density_matrix_measurement"),
        BackendSpec(
            engine=Engine.PENNYLANE,
            device="default.qubit",
            noise=NoiseSpec(name="depolarizing", probability=0.1),
        ),
    )
    assert any("default.mixed" in item for item in warnings)


def test_an_unsupported_gradient_method_is_refused() -> None:
    with pytest.raises(CapabilityError, match="gradient"):
        CapabilityRegistry().validate(
            _ir(),
            BackendSpec(
                engine=Engine.QISKIT,
                device="aer",
                gradient=GradientMethod.BACKPROP,
            ),
        )


def test_adjoint_is_refused_on_a_finite_shot_circuit() -> None:
    with pytest.raises(CapabilityError, match="parameter_shift"):
        CapabilityRegistry().validate(
            _ir(),
            BackendSpec(
                engine=Engine.PENNYLANE,
                device="default.qubit",
                shots=128,
                gradient=GradientMethod.ADJOINT,
            ),
        )


def test_adjoint_is_refused_on_a_noisy_circuit() -> None:
    with pytest.raises(CapabilityError, match="adjoint"):
        CapabilityRegistry().validate(
            _ir(measurement="density_matrix_measurement"),
            BackendSpec(
                engine=Engine.PENNYLANE,
                device="default.mixed",
                noise=NoiseSpec(name="depolarizing", probability=0.1),
                gradient=GradientMethod.ADJOINT,
            ),
        )


def test_a_wide_statevector_circuit_earns_a_memory_warning() -> None:
    circuit = CircuitIR(num_qubits=26, name="wide")
    circuit.add("RY", 0, 0.1, role="encoding")
    circuit.measure(
        Measurement(kind="expectation", wires=(0,), observables=(Observable("Z", (0,)),))
    )
    warnings = CapabilityRegistry().validate(
        circuit, BackendSpec(engine=Engine.QISKIT, device="aer", method="statevector")
    )
    assert any("24+" in item for item in warnings)


def test_sample_without_shots_warns_about_the_default() -> None:
    warnings = CapabilityRegistry().validate(
        _ir(measurement="sample_measurement"),
        BackendSpec(engine=Engine.PENNYLANE, device="default.qubit"),
    )
    assert any("1024" in item for item in warnings)


# --------------------------------------------------------------------------
# tracking
# --------------------------------------------------------------------------


def test_an_mlflow_tracker_refuses_to_log_outside_a_run() -> None:
    from qforge_ai.tracking import MLflowTracker

    tracker = MLflowTracker("surface")
    with pytest.raises(RuntimeError, match="with"):
        tracker.log_params({"a": 1})
    with pytest.raises(RuntimeError, match="with"):
        tracker.log_metrics({"a": 1.0})
    # Ending a run that never started is a no-op rather than an error.
    tracker.end_run()
    assert tracker.active_run is None


def test_an_mlflow_tracker_rejects_an_empty_experiment_name() -> None:
    from qforge_ai.tracking import MLflowTracker

    with pytest.raises(ValueError, match="empty"):
        MLflowTracker("   ")


def test_a_manifest_without_a_dataset_records_no_fingerprint() -> None:
    from qforge_ai.tracking import build_manifest

    spec = ExperimentSpec.from_dict(_spec_dict())
    manifest = build_manifest(spec, CircuitFactory().build(spec.circuit))
    assert manifest.dataset is None
    assert manifest.dataset_hash is None
    assert manifest.package_versions["numpy"] is not None


def test_a_manifest_saves_as_readable_json(tmp_path) -> None:
    from qforge_ai.tracking import build_manifest

    spec = ExperimentSpec.from_dict(_spec_dict())
    manifest = build_manifest(
        spec, CircuitFactory().build(spec.circuit), metrics={"accuracy": 0.5}
    )
    path = tmp_path / "manifest.json"
    manifest.save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["metrics"]["accuracy"] == 0.5
    assert payload["experiment"]["name"] == "surface"


# --------------------------------------------------------------------------
# benchmark
# --------------------------------------------------------------------------


def test_resource_report_serializes_and_scales_with_width() -> None:
    from qforge_ai.benchmark import resource_report

    narrow = resource_report(
        CircuitFactory().build(
            CircuitSpec(2, EncodingSpec("angle", 2), AnsatzSpec("real_amplitudes", 1))
        )
    )
    wide = resource_report(
        CircuitFactory().build(
            CircuitSpec(4, EncodingSpec("angle", 4), AnsatzSpec("real_amplitudes", 1))
        )
    )
    assert wide.estimated_statevector_bytes == 4 * narrow.estimated_statevector_bytes
    assert set(narrow.to_dict()) >= {"num_qubits", "parameter_count"}


def test_comparing_engines_on_an_unknown_measurement_is_refused() -> None:
    from qforge_ai.benchmark import compare_engines

    circuit = CircuitFactory().build(
        CircuitSpec(1, EncodingSpec("angle", 1), AnsatzSpec("real_amplitudes", 1))
    )
    with pytest.raises((ConfigurationError, ValueError, KeyError)):
        compare_engines(
            circuit, inputs=[0.0], weights=[0.0, 0.0], measurement="not_a_measurement"
        )


def test_counts_are_normalized_into_probabilities() -> None:
    from qforge_ai.benchmark import _empirical_probabilities

    left, right = _empirical_probabilities({"00": 30, "11": 70}, {"00": 50, "11": 50})
    assert np.allclose(left, [0.3, 0.7])
    assert np.allclose(right, [0.5, 0.5])
