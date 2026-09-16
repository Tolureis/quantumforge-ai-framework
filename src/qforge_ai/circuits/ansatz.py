"""Native variational ansatz builders."""

from __future__ import annotations

from collections.abc import Callable

from ..ir import CircuitIR, ParameterRef
from ..specs import AnsatzSpec
from .topology import entanglement_edges

AnsatzBuilder = Callable[[CircuitIR, AnsatzSpec], None]


def _rotations(
    ir: CircuitIR,
    gates: tuple[str, ...],
    weight_index: int,
) -> int:
    for wire in ir.data_wires:
        for gate in gates:
            ir.add(gate, wire, ParameterRef("weight", weight_index))
            weight_index += 1
    return weight_index


def _entangle(ir: CircuitIR, spec: AnsatzSpec, layer: int, gate: str = "CX") -> None:
    options = dict(spec.options)
    layer += int(options.get("layer_offset", 0))
    edges = entanglement_edges(
        ir.num_data_qubits,
        spec.entanglement,
        layer=layer,
        seed=int(options.get("seed", 42)),
        custom_edges=options.get("custom_edges"),
    )
    for control, target in edges:
        ir.add(gate, (control, target))


def hardware_efficient(ir: CircuitIR, spec: AnsatzSpec) -> None:
    index = ir.parameter_groups().get("weight", 0)
    gates = tuple(g.upper() for g in spec.rotation_blocks) or ("RY", "RZ")
    for layer in range(spec.layers):
        index = _rotations(ir, gates, index)
        _entangle(ir, spec, layer)
    if spec.layers and not spec.skip_final_rotation:
        _rotations(ir, gates, index)


def real_amplitudes(ir: CircuitIR, spec: AnsatzSpec) -> None:
    customized = AnsatzSpec(
        name=spec.name,
        layers=spec.layers,
        entanglement=spec.entanglement,
        rotation_blocks=("RY",),
        skip_final_rotation=spec.skip_final_rotation,
        options=spec.options,
    )
    hardware_efficient(ir, customized)


def efficient_su2(ir: CircuitIR, spec: AnsatzSpec) -> None:
    customized = AnsatzSpec(
        name=spec.name,
        layers=spec.layers,
        entanglement=spec.entanglement,
        rotation_blocks=("RY", "RZ"),
        skip_final_rotation=spec.skip_final_rotation,
        options=spec.options,
    )
    hardware_efficient(ir, customized)


def xyz(ir: CircuitIR, spec: AnsatzSpec) -> None:
    customized = AnsatzSpec(
        name=spec.name,
        layers=spec.layers,
        entanglement=spec.entanglement,
        rotation_blocks=("RX", "RY", "RZ"),
        skip_final_rotation=spec.skip_final_rotation,
        options=spec.options,
    )
    hardware_efficient(ir, customized)


def u3(ir: CircuitIR, spec: AnsatzSpec) -> None:
    """Apply genuine three-parameter U3 rotations with declared entanglement."""

    index = ir.parameter_groups().get("weight", 0)
    for layer in range(spec.layers):
        for wire in ir.data_wires:
            ir.add(
                "U",
                wire,
                ParameterRef("weight", index),
                ParameterRef("weight", index + 1),
                ParameterRef("weight", index + 2),
            )
            index += 3
        _entangle(ir, spec, layer)
    if spec.layers and not spec.skip_final_rotation:
        for wire in ir.data_wires:
            ir.add(
                "U",
                wire,
                ParameterRef("weight", index),
                ParameterRef("weight", index + 1),
                ParameterRef("weight", index + 2),
            )
            index += 3


def single_axis(ir: CircuitIR, spec: AnsatzSpec) -> None:
    axis = spec.name.upper().split("_")[0]
    customized = AnsatzSpec(
        name=spec.name,
        layers=spec.layers,
        entanglement=spec.entanglement,
        rotation_blocks=(axis,),
        skip_final_rotation=spec.skip_final_rotation,
        options=spec.options,
    )
    hardware_efficient(ir, customized)


def qcnn(ir: CircuitIR, spec: AnsatzSpec) -> None:
    """Log-depth QCNN-inspired convolution/pooling circuit."""
    index = ir.parameter_groups().get("weight", 0)
    active = list(ir.data_wires)
    layer = 0
    while len(active) > 1 and layer < max(1, spec.layers):
        pairs = list(zip(active[0::2], active[1::2], strict=False))
        for left, right in pairs:
            ir.add("RY", left, ParameterRef("weight", index))
            ir.add("RY", right, ParameterRef("weight", index + 1))
            ir.add("CX", (left, right))
            ir.add("RZ", right, ParameterRef("weight", index + 2))
            ir.add("CX", (right, left))
            index += 3
        # Pool right wire into left wire; measurement/discard remains adapter-safe.
        for left, right in pairs:
            ir.add("CRY", (right, left), ParameterRef("weight", index))
            index += 1
        active = active[0::2]
        layer += 1
    ir.metadata["active_qcnn_wires"] = active


ANSATZES: dict[str, AnsatzBuilder] = {
    "hardware_efficient_ansatz": hardware_efficient,
    "two_local": hardware_efficient,
    "n_local": hardware_efficient,
    "alternating_layer_ansatz": hardware_efficient,
    "strongly_entangling_layers": xyz,
    "basic_entangler_layers": real_amplitudes,
    "random_layers": xyz,
    "real_amplitudes": real_amplitudes,
    "efficient_su2": efficient_su2,
    "rx_ansatz": single_axis,
    "ry_ansatz": single_axis,
    "rz_ansatz": single_axis,
    "xyz_rotation_ansatz": xyz,
    "euler_rotation_ansatz": xyz,
    "u3_ansatz": u3,
    "ising_ansatz": efficient_su2,
    "pauli_rotation_ansatz": xyz,
    "qcnn": qcnn,
    "quantum_convolution": qcnn,
    "quantum_pooling": qcnn,
}


__all__ = ["ANSATZES", "AnsatzBuilder"]
