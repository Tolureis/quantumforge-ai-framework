import json

import pytest

from qforge_ai import AnsatzSpec, CircuitSpec, EncodingSpec, MeasurementSpec
from qforge_ai.catalog import build_catalog
from qforge_ai.circuits import CircuitFactory, entanglement_edges
from qforge_ai.errors import ConfigurationError, RegistryError


def test_hardware_efficient_ir() -> None:
    ir = CircuitFactory().build(
        CircuitSpec(
            4,
            EncodingSpec("angle", 4),
            AnsatzSpec("hardware_efficient", layers=2, entanglement="ring"),
            MeasurementSpec("expectation"),
        )
    )
    assert ir.parameter_groups() == {"input": 4, "weight": 24}
    assert ir.two_qubit_gate_count() == 8
    assert ir.depth() > 0
    assert len(ir.measurements[0].observables) == 4


@pytest.mark.parametrize(
    "encoding",
    [
        "basis",
        "angle",
        "dense_angle",
        "phase",
        "amplitude",
        "z_feature_map",
        "zz_feature_map",
        "iqp_feature_map",
    ],
)
def test_native_encodings_build(encoding: str) -> None:
    dimension = 4 if encoding in {"amplitude", "dense_angle"} else 2
    ir = CircuitFactory().build(
        CircuitSpec(
            2,
            EncodingSpec(encoding, dimension, normalize=True),
            AnsatzSpec("real_amplitudes", layers=1),
            MeasurementSpec("probability"),
        )
    )
    assert ir.operations


def test_descriptor_without_factory_is_explicit() -> None:
    with pytest.raises(RegistryError, match="no executable IR producer"):
        CircuitFactory().build(
            CircuitSpec(
                2,
                EncodingSpec("qram_encoding", 2),
                AnsatzSpec("real_amplitudes"),
            )
        )


def test_entanglement_edges() -> None:
    assert entanglement_edges(4, "linear") == [(0, 1), (1, 2), (2, 3)]
    assert len(entanglement_edges(4, "full")) == 6
    assert (3, 0) in entanglement_edges(4, "ring")
    assert entanglement_edges(2, "ring") == [(0, 1)]


def test_parameter_count_uses_logical_parameter_identity() -> None:
    ir = CircuitFactory().build(
        CircuitSpec(
            3,
            EncodingSpec("zz_feature_map", 3),
            AnsatzSpec("real_amplitudes", layers=1),
        )
    )
    assert ir.parameter_count() == sum(ir.parameter_groups().values())
    products = [operation for operation in ir.operations if operation.name == "RZ_PRODUCT"]
    assert products
    assert products[0].params[0].offset == pytest.approx(3.141592653589793)
    assert products[0].params[0].scale == -1.0


def test_data_reuploading_is_interleaved_and_lossless() -> None:
    ir = CircuitFactory().build(
        CircuitSpec(
            3,
            EncodingSpec("data_reuploading", 6, repeats=2),
            AnsatzSpec("hardware_efficient", layers=2),
        )
    )
    roles = [operation.metadata.get("role") for operation in ir.operations]
    cycles = [operation.metadata.get("cycle") for operation in ir.operations]
    assert ir.parameter_groups()["input"] == 6
    assert roles[0] == "encoding"
    assert "ansatz" in roles
    assert cycles == sorted(cycles)
    per_cycle = {
        cycle: [
            operation.params[0].index
            for operation in ir.operations
            if operation.metadata.get("role") == "encoding"
            and operation.metadata.get("cycle") == cycle
        ]
        for cycle in range(2)
    }
    assert per_cycle == {0: list(range(6)), 1: list(range(6))}


def test_data_reuploading_supports_full_vectors_and_rejects_invalid_cycles() -> None:
    wide = CircuitFactory().build(
        CircuitSpec(
            3,
            EncodingSpec("data_reuploading", 7, repeats=2),
            AnsatzSpec("hardware_efficient", layers=2),
        )
    )
    assert wide.parameter_groups()["input"] == 7
    with pytest.raises(ConfigurationError, match="at least twice"):
        CircuitFactory().build(
            CircuitSpec(
                3,
                EncodingSpec("data_reuploading", 3, repeats=1),
                AnsatzSpec("hardware_efficient", layers=1),
            )
        )
    with pytest.raises(ConfigurationError, match="must be equal"):
        CircuitFactory().build(
            CircuitSpec(
                3,
                EncodingSpec("data_reuploading", 6, repeats=2),
                AnsatzSpec("hardware_efficient", layers=1),
            )
        )


def test_qcnn_defaults_to_active_wires() -> None:
    ir = CircuitFactory().build(
        CircuitSpec(
            4,
            EncodingSpec("angle", 4),
            AnsatzSpec("qcnn", layers=2),
            MeasurementSpec("expectation"),
        )
    )
    assert ir.metadata["active_qcnn_wires"] == [0]
    assert ir.measurements[0].wires == (0,)


def test_unsupported_measurement_fails_early() -> None:
    with pytest.raises(RegistryError, match="no core measurement implementation"):
        CircuitFactory().build(
            CircuitSpec(
                2,
                EncodingSpec("angle", 2),
                AnsatzSpec("real_amplitudes", 1),
                MeasurementSpec("shadow_measurement"),
            )
        )


def test_ir_metadata_is_json_serializable_and_feature_map_has_no_ansatz() -> None:
    factory = CircuitFactory()
    spec = CircuitSpec(2, EncodingSpec("angle", 2), AnsatzSpec("real_amplitudes", 1))
    ir = factory.build(spec)
    json.dumps(ir.metadata)
    feature_ir = factory.build_feature_map(spec)
    assert feature_ir.operations
    assert all(operation.metadata["role"] == "encoding" for operation in feature_ir.operations)


def test_dynamic_ir_uses_classical_bits() -> None:
    from qforge_ai.ir import CircuitIR

    ir = CircuitIR(2)
    ir.add("H", 0).measure_wire(0, 0).add_conditional("X", 1, clbit=0).reset(0)
    assert ir.num_clbits == 1
    assert ir.operations[2].condition == (0, 1)


def test_ancilla_wires_are_reserved_from_encoding_and_ansatz() -> None:
    ir = CircuitFactory().build(
        CircuitSpec(
            2,
            EncodingSpec("angle", 2),
            AnsatzSpec("hardware_efficient", layers=2, entanglement="ring"),
            MeasurementSpec("expectation"),
            ancilla_qubits=1,
        )
    )
    assert ir.num_qubits == 3
    assert ir.num_data_qubits == 2
    assert ir.data_wires == (0, 1)
    assert ir.ancilla_wires == (2,)
    assert all(2 not in operation.wires for operation in ir.operations)
    assert ir.measurements[0].wires == (0, 1)


def test_ancilla_can_be_measured_only_when_explicitly_requested() -> None:
    ir = CircuitFactory().build(
        CircuitSpec(
            2,
            EncodingSpec("angle", 2),
            AnsatzSpec("real_amplitudes", layers=1),
            MeasurementSpec("probability", wires=(2,)),
            ancilla_qubits=1,
        )
    )
    assert ir.measurements[0].wires == (2,)
    assert all(2 not in operation.wires for operation in ir.operations)


def test_registered_plugin_factory_is_used_by_circuit_factory() -> None:
    registry = build_catalog()

    def one_hot_plugin(ir, spec) -> None:
        del spec
        ir.add("X", ir.data_wires[0], plugin="one_hot")

    registry.register_factory("one_hot_encoding", one_hot_plugin)
    factory = CircuitFactory(registry)
    ir = factory.build(
        CircuitSpec(
            2,
            EncodingSpec("one_hot_encoding", 2),
            AnsatzSpec("real_amplitudes", layers=1),
        )
    )
    assert ir.operations[0].name == "X"
    assert ir.operations[0].metadata["plugin"] == "one_hot"
    assert registry.has_factory("one_hot_encoding")
    assert registry.is_runnable("one_hot_encoding")


def test_u3_ansatz_uses_genuine_three_parameter_u_gates() -> None:
    ir = CircuitFactory().build(
        CircuitSpec(
            2,
            EncodingSpec("angle", 2),
            AnsatzSpec("u3_ansatz", layers=1),
        )
    )
    ansatz_operations = [
        operation for operation in ir.operations if operation.metadata.get("role") == "ansatz"
    ]
    assert any(operation.name == "U" and len(operation.params) == 3 for operation in ansatz_operations)
