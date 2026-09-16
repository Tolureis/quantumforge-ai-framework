"""Real (unmocked) coverage for the optional MLflow, Optuna and plugin paths.

Finding 11 in the 1.2.1 report: MLflow was only exercised through mocks,
Optuna had no workflow test at all, and ``qforge_ai.plugins`` had zero
coverage -- while CI never installed the optional extras, so even the mocked
paths never ran against the real libraries.  These tests drive the actual
packages against local, hermetic backing stores.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from qforge_ai import CircuitSpec, ExperimentSpec, TrainingSpec
from qforge_ai.errors import DuplicateFactoryError, RegistryError
from qforge_ai.registry import ComponentDescriptor, ComponentRegistry
from qforge_ai.specs import ComponentStatus, Engine

HAS_MLFLOW = importlib.util.find_spec("mlflow") is not None
HAS_OPTUNA = importlib.util.find_spec("optuna") is not None

requires_mlflow = pytest.mark.skipif(not HAS_MLFLOW, reason="mlflow not installed")
requires_optuna = pytest.mark.skipif(not HAS_OPTUNA, reason="optuna not installed")


# --------------------------------------------------------------------------
# MLflow -- driven against a real local file store, not a mock
# --------------------------------------------------------------------------


@pytest.fixture()
def tracking_uri(tmp_path: Path) -> str:
    """A real, hermetic MLflow backend.

    SQLite rather than the ``./mlruns`` file store: recent MLflow puts the
    filesystem backend in maintenance mode and refuses it outright, so a
    file-store fixture would only prove that the test is skipped.
    """

    return f"sqlite:///{tmp_path / 'mlflow.db'}"


@requires_mlflow
def test_mlflow_tracker_records_a_real_run(tracking_uri: str) -> None:
    import mlflow

    from qforge_ai.tracking import MLflowTracker
    from qforge_ai.training import EpochMetrics, TrainingResult

    spec = ExperimentSpec(
        name="mlflow-integration",
        circuit=CircuitSpec(num_qubits=2),
        training=TrainingSpec(epochs=2),
        tags=("integration",),
    )
    with MLflowTracker(spec, tracking_uri=tracking_uri, tags={"suite": "integration"}) as tracker:
        run_id = tracker.active_run.info.run_id
        tracker.log_experiment()
        tracker(EpochMetrics(1, 0.9, 0.8, 1e-3, 0.01))
        tracker(EpochMetrics(2, 0.4, 0.3, 1e-3, 0.01))
        tracker.log_result(
            TrainingResult(history=[], best_epoch=2, best_validation_loss=0.3, stopped_early=False)
        )

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(run_id)
    assert run.info.status == "FINISHED"
    assert run.data.params["name"] == "mlflow-integration"
    assert run.data.params["circuit.num_qubits"] == "2"
    assert run.data.metrics["train_loss"] == pytest.approx(0.4)
    assert run.data.metrics["best_validation_loss"] == pytest.approx(0.3)
    assert run.data.tags["suite"] == "integration"
    history = client.get_metric_history(run_id, "train_loss")
    assert [point.step for point in history] == [1, 2]


@requires_mlflow
def test_mlflow_tracker_marks_failed_runs(tracking_uri: str) -> None:
    import mlflow

    from qforge_ai.tracking import MLflowTracker

    run_id = None
    with pytest.raises(RuntimeError):
        with MLflowTracker("failure-case", tracking_uri=tracking_uri) as tracker:
            run_id = tracker.active_run.info.run_id
            raise RuntimeError("training blew up")

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    assert client.get_run(run_id).info.status == "FAILED"


@requires_mlflow
def test_mlflow_tracker_rejects_use_outside_a_run(tracking_uri: str) -> None:
    from qforge_ai.tracking import MLflowTracker

    tracker = MLflowTracker("no-run", tracking_uri=tracking_uri)
    with pytest.raises(RuntimeError):
        tracker.log_metrics({"loss": 1.0})
    with pytest.raises(RuntimeError):
        tracker.log_experiment(
            ExperimentSpec(name="x", circuit=CircuitSpec(num_qubits=1))
        )


@requires_mlflow
def test_mlflow_tracker_end_run_is_idempotent(tracking_uri: str) -> None:
    from qforge_ai.tracking import MLflowTracker

    tracker = MLflowTracker("idempotent", tracking_uri=tracking_uri).start_run()
    with pytest.raises(RuntimeError, match="already active"):
        tracker.start_run()
    tracker.end_run()
    tracker.end_run()
    assert tracker.active_run is None


@requires_mlflow
def test_mlflow_tracker_survives_unserializable_options(tracking_uri: str) -> None:
    """A spec carrying an engine object must still be loggable."""

    from qforge_ai.specs import ModelSpec
    from qforge_ai.tracking import MLflowTracker

    spec = ExperimentSpec(
        name="opaque-options",
        circuit=CircuitSpec(num_qubits=2),
        model=ModelSpec(options={"interpret": lambda value: value}),
    )
    with MLflowTracker(spec, tracking_uri=tracking_uri) as tracker:
        tracker.log_experiment()
        assert tracker.active_run is not None


# --------------------------------------------------------------------------
# Optuna -- a real study, executed end to end
# --------------------------------------------------------------------------


@requires_optuna
def test_optuna_study_optimizes_a_real_objective() -> None:
    from qforge_ai.automl import AutoMLStudy, QuantumSearchSpace, SearchParameter

    space = QuantumSearchSpace(
        parameters={
            "layers": SearchParameter(kind="int", low=1, high=4),
            "learning_rate": SearchParameter(kind="float", low=1e-4, high=1e-1, log=True),
            "ansatz": SearchParameter(
                kind="categorical", choices=("real_amplitudes", "efficient_su2")
            ),
        }
    )
    seen: list[dict] = []

    def objective(parameters: dict, trial: object) -> float:
        seen.append(parameters)
        # Deterministic bowl with a known minimum at layers=3.
        return (parameters["layers"] - 3) ** 2 + parameters["learning_rate"]

    study = AutoMLStudy(space, direction="minimize", seed=7).optimize(objective, n_trials=12)
    assert len(study.trials) == 12
    assert len(seen) == 12
    assert set(seen[0]) == {"layers", "learning_rate", "ansatz"}
    assert study.best_params["layers"] == 3
    assert study.best_value < 1.0


@requires_optuna
def test_optuna_study_is_reproducible_for_a_fixed_seed() -> None:
    from qforge_ai.automl import AutoMLStudy, QuantumSearchSpace, SearchParameter

    space = QuantumSearchSpace(
        parameters={"x": SearchParameter(kind="float", low=-5.0, high=5.0)}
    )

    def run() -> list[float]:
        study = AutoMLStudy(space, seed=11).optimize(
            lambda parameters, trial: (parameters["x"] - 1.5) ** 2, n_trials=8
        )
        return [trial.params["x"] for trial in study.trials]

    assert run() == run()


@requires_optuna
def test_optuna_study_persists_to_storage(tmp_path: Path) -> None:
    import optuna

    from qforge_ai.automl import AutoMLStudy, QuantumSearchSpace, SearchParameter

    storage = f"sqlite:///{tmp_path / 'study.db'}"
    space = QuantumSearchSpace(
        parameters={"x": SearchParameter(kind="int", low=0, high=5)}
    )
    AutoMLStudy(space, study_name="persisted", storage=storage, seed=3).optimize(
        lambda parameters, trial: float(parameters["x"]), n_trials=5
    )
    reloaded = optuna.load_study(study_name="persisted", storage=storage)
    assert len(reloaded.trials) == 5


@requires_optuna
def test_optuna_deprecated_facade_still_runs() -> None:
    from qforge_ai.automl import OptunaSearch, QuantumSearchSpace, SearchParameter

    space = QuantumSearchSpace(parameters={"x": SearchParameter(kind="int", low=0, high=3)})
    study = OptunaSearch(space, seed=5).run(
        lambda parameters, trial: float(parameters["x"]), n_trials=4
    )
    assert len(study.trials) == 4


@requires_optuna
def test_optuna_trial_callbacks_are_invoked() -> None:
    from qforge_ai.automl import AutoMLStudy, QuantumSearchSpace, SearchParameter

    calls: list[int] = []
    space = QuantumSearchSpace(parameters={"x": SearchParameter(kind="int", low=0, high=2)})
    AutoMLStudy(space, seed=1).optimize(
        lambda parameters, trial: float(parameters["x"]),
        n_trials=3,
        callbacks=(lambda study, trial: calls.append(trial.number),),
    )
    assert calls == [0, 1, 2]


def test_search_parameter_validation_is_explicit() -> None:
    from qforge_ai.automl import AutoMLStudy, QuantumSearchSpace, SearchParameter
    from qforge_ai.errors import ConfigurationError

    space = QuantumSearchSpace(parameters={})
    with pytest.raises(ConfigurationError):
        AutoMLStudy(space, direction="sideways")
    with pytest.raises(ConfigurationError):
        AutoMLStudy(space).optimize(lambda parameters, trial: 0.0, n_trials=0)
    with pytest.raises(ConfigurationError):
        SearchParameter(kind="categorical").suggest(object(), "empty")
    with pytest.raises(ConfigurationError):
        SearchParameter(kind="float").suggest(object(), "no_bounds")
    with pytest.raises(ConfigurationError):
        SearchParameter(kind="triangular", low=0, high=1).suggest(object(), "unknown_kind")


# --------------------------------------------------------------------------
# Plugin discovery -- previously at 0% coverage
# --------------------------------------------------------------------------


@dataclass
class _FakeEntryPoint:
    name: str
    value: object

    def load(self) -> object:
        return self.value


def _install_fake_entry_points(monkeypatch, entries: list[_FakeEntryPoint]) -> None:
    import qforge_ai.plugins as plugins

    def fake_entry_points(*, group: str) -> list[_FakeEntryPoint]:
        assert group == "qforge_ai.plugins"
        return entries

    monkeypatch.setattr(plugins, "entry_points", fake_entry_points)


def _descriptor(key: str, category: str = "ansatz") -> ComponentDescriptor:
    return ComponentDescriptor(
        key=key,
        display_name=key.replace("_", " ").title(),
        category=category,
        description="plugin supplied component",
        status=ComponentStatus.EXPERIMENTAL,
        engines=frozenset({Engine.IR}),
    )


def test_load_plugins_accepts_a_callable_plugin(monkeypatch) -> None:
    from qforge_ai import plugins

    registered: list[str] = []

    def register(registry: ComponentRegistry) -> None:
        registered.append(type(registry).__name__)

    _install_fake_entry_points(monkeypatch, [_FakeEntryPoint("callable_plugin", register)])
    assert plugins.load_plugins() == ["callable_plugin"]
    assert registered == ["ComponentRegistry"]


def test_load_plugins_accepts_a_module_style_plugin(monkeypatch) -> None:
    from qforge_ai import plugins

    seen: list[str] = []

    class Module:
        @staticmethod
        def register(registry: ComponentRegistry) -> None:
            seen.append("registered")

    _install_fake_entry_points(monkeypatch, [_FakeEntryPoint("module_plugin", Module)])
    assert plugins.load_plugins() == ["module_plugin"]
    assert seen == ["registered"]


def test_load_plugins_rejects_an_unusable_entry_point(monkeypatch) -> None:
    from qforge_ai import plugins

    _install_fake_entry_points(monkeypatch, [_FakeEntryPoint("broken", object())])
    with pytest.raises(TypeError, match="cannot be registered"):
        plugins.load_plugins()


def test_load_plugins_returns_empty_without_entry_points(monkeypatch) -> None:
    from qforge_ai import plugins

    _install_fake_entry_points(monkeypatch, [])
    assert plugins.load_plugins() == []


def test_plugin_registered_component_becomes_usable() -> None:
    """A plugin descriptor plus factory must survive resolution and execution."""

    from qforge_ai.ir import CircuitIR, ParameterRef
    from qforge_ai.specs import AnsatzSpec

    registry = ComponentRegistry()
    registry.register(_descriptor("plugin_double_ry"))

    def builder(ir: CircuitIR, spec: AnsatzSpec) -> None:
        index = ir.parameter_groups().get("weight", 0)
        for wire in ir.data_wires:
            ir.add("RY", wire, ParameterRef("weight", index))
            index += 1

    registry.register_factory("plugin_double_ry", builder)
    descriptor = registry.get("plugin_double_ry")
    # Attaching a factory promotes the entry to a runnable native component.
    assert descriptor.status == ComponentStatus.NATIVE
    assert descriptor.metadata["runtime_contract"] == "runnable"
    assert registry.is_runnable("plugin_double_ry")

    ir = CircuitIR(num_qubits=2)
    registry.get_factory("plugin_double_ry")(ir, AnsatzSpec(name="plugin_double_ry"))
    assert ir.gate_counts() == {"RY": 2}


def test_plugin_factory_registration_is_not_silently_overwritten() -> None:
    registry = ComponentRegistry()
    registry.register(_descriptor("plugin_conflict"))
    registry.register_factory("plugin_conflict", lambda *args, **kwargs: None)
    with pytest.raises(DuplicateFactoryError):
        registry.register_factory("plugin_conflict", lambda *args, **kwargs: None)
    registry.register_factory("plugin_conflict", lambda *args, **kwargs: "replaced", replace=True)
    assert registry.get_factory("plugin_conflict")() == "replaced"


def test_descriptor_without_factory_reports_a_clear_error() -> None:
    registry = ComponentRegistry()
    registry.register(_descriptor("plugin_metadata_only"))
    with pytest.raises(RegistryError, match="executable factory"):
        registry.get_factory("plugin_metadata_only")


def test_plugin_register_factory_targets_the_shared_catalog() -> None:
    from qforge_ai import plugins
    from qforge_ai.catalog import catalog

    marker = object()
    plugins.register_factory("ry_ansatz", lambda *args, **kwargs: marker, replace=True)
    try:
        assert catalog.get_factory("ry_ansatz")() is marker
    finally:
        # Restore the built-in builder so later tests see a clean catalog.
        from qforge_ai.circuits.ansatz import ANSATZES

        plugins.register_factory("ry_ansatz", ANSATZES["ry_ansatz"], replace=True)


def test_python_version_supports_the_declared_floor() -> None:
    assert sys.version_info >= (3, 11)
