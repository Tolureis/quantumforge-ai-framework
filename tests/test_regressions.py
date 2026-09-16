import json

import pytest

from qforge_ai import (
    AnsatzSpec,
    CircuitIR,
    CircuitSpec,
    ComponentStatus,
    EncodingSpec,
    ExperimentSpec,
    Measurement,
    MeasurementSpec,
    ModelSpec,
    NoiseSpec,
    Observable,
    OptunaSearch,
    ParameterRef,
    QuantumSearchSpace,
    TrainingRun,
)
from qforge_ai.catalog import catalog
from qforge_ai.circuits import CircuitFactory
from qforge_ai.errors import ConfigurationError, DuplicateComponentError, RegistryError
from qforge_ai.registry import ComponentDescriptor, ComponentRegistry
from qforge_ai.training import EpochMetrics, TrainingResult


def test_custom_observable_must_be_finite_square_and_hermitian() -> None:
    with pytest.raises(ConfigurationError, match="Hermitian"):
        Observable("CUSTOM", (0,), matrix=((1, 1), (0, -1)))
    with pytest.raises(ConfigurationError, match="finite"):
        Observable("CUSTOM", (0,), matrix=((1, complex("nan")), (0, -1)))
    with pytest.raises(ConfigurationError, match="2x2"):
        Observable("CUSTOM", (0,), matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)))


@pytest.mark.parametrize("wires", [(), (0, 0)])
def test_measurement_wires_are_non_empty_and_unique(wires: tuple[int, ...]) -> None:
    with pytest.raises(ConfigurationError, match="wire"):
        Measurement("probability", wires)
    with pytest.raises(ConfigurationError, match="wire"):
        MeasurementSpec("probability", wires=wires)


def test_high_value_spec_validations_are_fail_fast() -> None:
    with pytest.raises(ConfigurationError, match="hidden_dimensions"):
        ModelSpec(hidden_dimensions=(4, 0))
    with pytest.raises(ConfigurationError, match="Ansatz name"):
        AnsatzSpec(name="   ")
    with pytest.raises(ConfigurationError, match="rotation_blocks"):
        AnsatzSpec(rotation_blocks=("RY", "INVALID"))
    with pytest.raises(ConfigurationError, match=r"t2 <= 2\*t1"):
        NoiseSpec(name="thermal_relaxation", t1=10.0, t2=21.0, gate_time=1.0)
    with pytest.raises(ConfigurationError, match="gate_time"):
        NoiseSpec(name="thermal_relaxation", t1=10.0, t2=20.0, gate_time=0.0)
    with pytest.raises(ConfigurationError, match="t1, t2 and gate_time"):
        NoiseSpec(name="thermal_relaxation")
    with pytest.raises(ConfigurationError, match="Experiment name"):
        ExperimentSpec(name=" ", circuit=CircuitSpec(1))
    with pytest.raises(ConfigurationError, match="scale"):
        ParameterRef("input", 0, scale=float("nan"))
    with pytest.raises(ConfigurationError, match="offset"):
        ParameterRef("input", 0, offset=float("inf"))


def test_subsystem_statevector_is_rejected_and_full_default_includes_ancilla() -> None:
    with pytest.raises(ConfigurationError, match="Subsystem statevector"):
        CircuitFactory().build(
            CircuitSpec(
                2,
                EncodingSpec("angle", 2),
                AnsatzSpec("real_amplitudes"),
                MeasurementSpec("statevector", wires=(0,)),
            )
        )

    direct = CircuitIR(2)
    with pytest.raises(ConfigurationError, match="reduced_density_matrix"):
        direct.measure(Measurement("statevector", (0,)))

    full = CircuitFactory().build(
        CircuitSpec(
            2,
            EncodingSpec("angle", 2),
            AnsatzSpec("real_amplitudes"),
            MeasurementSpec("statevector"),
            ancilla_qubits=1,
        )
    )
    assert full.measurements[0].wires == (0, 1, 2)


def test_reduced_density_matrix_public_component_is_runnable() -> None:
    assert catalog.is_runnable("reduced_density_matrix_measurement")
    ir = CircuitFactory().build(
        CircuitSpec(
            2,
            EncodingSpec("angle", 2),
            AnsatzSpec("real_amplitudes"),
            MeasurementSpec("reduced_density_matrix", wires=(0,)),
        )
    )
    assert ir.measurements[0].kind == "density_matrix"
    assert ir.measurements[0].wires == (0,)
    assert ir.metadata["measurement_component"] == "reduced_density_matrix_measurement"


def test_unsupported_measurement_error_renders_supported_keys() -> None:
    with pytest.raises(RegistryError) as captured:
        CircuitFactory().build(
            CircuitSpec(
                1,
                EncodingSpec("angle", 1),
                AnsatzSpec("real_amplitudes"),
                MeasurementSpec("shadow_measurement"),
            )
        )
    assert "{sorted(SUPPORTED_MEASUREMENTS)}" not in str(captured.value)
    assert "expectation_measurement" in str(captured.value)


def test_registry_registration_is_transaction_safe() -> None:
    registry = ComponentRegistry()
    registry.register(
        ComponentDescriptor("owner", "Owner", "test", "Owner", aliases=("taken",)),
        lambda: "owner",
    )
    registry.register(
        ComponentDescriptor("first", "First", "test", "First", aliases=("legacy",)),
        lambda: "first",
    )
    before = list(registry.list())

    with pytest.raises(DuplicateComponentError, match="Alias conflict"):
        registry.register(
            ComponentDescriptor(
                "candidate",
                "Candidate",
                "test",
                "Candidate",
                aliases=("fresh", "taken"),
            )
        )
    assert registry.list() == before
    with pytest.raises(RegistryError, match="Unknown component"):
        registry.get("fresh")

    with pytest.raises(DuplicateComponentError, match="Alias conflict"):
        registry.register(
            ComponentDescriptor(
                "first",
                "First v2",
                "test",
                "First v2",
                aliases=("new_alias", "taken"),
            ),
            replace=True,
        )
    assert registry.get("legacy").key == "first"
    assert registry.create("first") == "first"
    with pytest.raises(RegistryError, match="Unknown component"):
        registry.get("new_alias")


def test_restored_serialization_and_status_apis() -> None:
    ir = CircuitIR(1)
    ir.add("RY", 0, ParameterRef("input", 0)).measure(
        Measurement("expectation", (0,), (Observable("Z", (0,)),))
    )
    serialized_ir = ir.to_dict()
    json.dumps(serialized_ir)
    assert serialized_ir["operations"][0]["params"][0]["group"] == "input"

    result = TrainingResult(
        history=[EpochMetrics(1, 0.5, 0.4, 0.01, 0.1)],
        best_epoch=1,
        best_validation_loss=0.4,
        stopped_early=False,
    )
    run = TrainingRun.from_result(result, model="model")
    assert isinstance(run, TrainingResult)
    assert run.result is run
    assert run.model == "model"
    assert run.to_dict()["history"][0]["epoch"] == 1
    assert ComponentStatus.BUILTIN is ComponentStatus.NATIVE
    assert issubclass(OptunaSearch, object)
    assert isinstance(OptunaSearch(QuantumSearchSpace()), OptunaSearch)
