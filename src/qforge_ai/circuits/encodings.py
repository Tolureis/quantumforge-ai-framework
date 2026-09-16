"""Native encoding builders targeting :class:`CircuitIR`."""

from __future__ import annotations

import math
from collections.abc import Callable

from ..errors import ConfigurationError, RegistryError
from ..ir import CircuitIR, ParameterRef
from ..specs import EncodingSpec

EncodingBuilder = Callable[[CircuitIR, EncodingSpec], None]


def _feature_dim(ir: CircuitIR, spec: EncodingSpec) -> int:
    return spec.feature_dimension or ir.num_data_qubits


def _require_capacity(ir: CircuitIR, spec: EncodingSpec, slots_per_repeat: int) -> int:
    dimension = _feature_dim(ir, spec)
    capacity = slots_per_repeat * spec.repeats
    if dimension > capacity:
        raise ConfigurationError(
            f"{spec.name} has feature_dimension={dimension}, while the circuit's "
            f"num_qubits/repeats capacity is {capacity}. Increase the qubit or repeats "
            "value to avoid losing data."
        )
    return dimension


#: Rotations about the same axis commute and simply add:
#: ``RY(a) RY(b) == RY(a + b)``.  Any encoding that stacks more than one
#: feature onto a wire must therefore alternate the rotation axis, otherwise
#: two different feature vectors collapse onto the same state and the encoded
#: information is genuinely destroyed.
_AXIS_CYCLE = ("X", "Y", "Z")


def _axis_for_slot(rotation: str, slot: int) -> str:
    """Return the rotation axis for the *slot*-th load onto a single wire.

    ``slot`` counts how many features have already been written to that wire,
    so consecutive loads always land on different axes and cannot merge.
    """

    start = _AXIS_CYCLE.index(rotation.upper())
    return _AXIS_CYCLE[(start + slot) % len(_AXIS_CYCLE)]


def _rotation_gate(rotation: str, slot: int) -> str:
    return f"R{_axis_for_slot(rotation, slot)}"


def angle(ir: CircuitIR, spec: EncodingSpec) -> None:
    dim = _require_capacity(ir, spec, ir.num_data_qubits)
    for repeat in range(spec.repeats):
        for wire in ir.data_wires:
            index = (repeat * ir.num_data_qubits + wire) % dim
            # ``repeat`` is exactly the number of features already loaded onto
            # this wire, so it is the correct slot counter.  With repeats=1
            # this reduces to the declared rotation and nothing changes.
            ir.add(
                _rotation_gate(spec.rotation, repeat),
                wire,
                ParameterRef("input", index, spec.scaling),
            )


def dense_angle(ir: CircuitIR, spec: EncodingSpec) -> None:
    dim = _require_capacity(ir, spec, 2 * ir.num_data_qubits)
    for repeat in range(spec.repeats):
        for wire in ir.data_wires:
            base = (repeat * 2 * ir.num_data_qubits + 2 * wire) % dim
            ir.add("RY", wire, ParameterRef("input", base, spec.scaling))
            ir.add("RZ", wire, ParameterRef("input", (base + 1) % dim, spec.scaling))


def phase(ir: CircuitIR, spec: EncodingSpec) -> None:
    dim = _require_capacity(ir, spec, ir.num_data_qubits)
    for repeat in range(spec.repeats):
        for wire in ir.data_wires:
            ir.add("H", wire)
            ir.add(
                "RZ",
                wire,
                ParameterRef(
                    "input", (repeat * ir.num_data_qubits + wire) % dim, spec.scaling
                ),
            )


def basis(ir: CircuitIR, spec: EncodingSpec) -> None:
    dim = _feature_dim(ir, spec)
    if dim > ir.num_data_qubits:
        raise ConfigurationError(
            f"Basis encoding has feature_dimension={dim}, "
            f"data_qubits={ir.num_data_qubits}; "
            "one qubit is required per bit."
        )
    for wire in range(dim):
        ir.add("BASIS", wire, ParameterRef("input", wire))


def amplitude(ir: CircuitIR, spec: EncodingSpec) -> None:
    dimension = _feature_dim(ir, spec)
    max_dimension = 2**ir.num_data_qubits
    if dimension > max_dimension:
        raise RegistryError(
            f"Amplitude encoding requires feature_dimension <= 2**num_qubits; "
            f"{dimension} > {max_dimension}."
        )
    refs = tuple(ParameterRef("input", i, spec.scaling) for i in range(dimension))
    ir.add("STATE_PREP", ir.data_wires, *refs, normalize=spec.normalize)


def superposition(ir: CircuitIR, spec: EncodingSpec) -> None:
    del spec
    for wire in ir.data_wires:
        ir.add("H", wire)


def z_feature_map(ir: CircuitIR, spec: EncodingSpec) -> None:
    dim = _require_capacity(ir, spec, ir.num_data_qubits)
    for repeat in range(spec.repeats):
        for wire in ir.data_wires:
            ir.add("H", wire)
            ir.add(
                "RZ",
                wire,
                ParameterRef(
                    "input",
                    (repeat * ir.num_data_qubits + wire) % dim,
                    spec.scaling * 2.0,
                ),
            )


def zz_feature_map(ir: CircuitIR, spec: EncodingSpec) -> None:
    """Qiskit-compatible second-order ZZ feature map.

    The pair phase follows ``2 * (pi - s*x_i) * (pi - s*x_j)`` instead of the
    older direct ``2*x_i*x_j`` shortcut.
    """

    dim = _require_capacity(ir, spec, ir.num_data_qubits)
    for repeat in range(spec.repeats):
        base = repeat * ir.num_data_qubits
        for wire in ir.data_wires:
            index = (base + wire) % dim
            ir.add("H", wire)
            ir.add("RZ", wire, ParameterRef("input", index, spec.scaling * 2.0))
        for wire in range(ir.num_data_qubits - 1):
            left = ParameterRef(
                "input", (base + wire) % dim, scale=-spec.scaling, offset=math.pi
            )
            right = ParameterRef(
                "input", (base + wire + 1) % dim, scale=-spec.scaling, offset=math.pi
            )
            ir.add("CX", (wire, wire + 1))
            ir.add("RZ_PRODUCT", wire + 1, left, right, coefficient=2.0)
            ir.add("CX", (wire, wire + 1))


def iqp(ir: CircuitIR, spec: EncodingSpec) -> None:
    dim = _require_capacity(ir, spec, ir.num_data_qubits)
    for repeat in range(spec.repeats):
        base = repeat * ir.num_data_qubits
        for wire in ir.data_wires:
            ir.add("H", wire)
            ir.add("RZ", wire, ParameterRef("input", (base + wire) % dim, spec.scaling))
        for wire in range(ir.num_data_qubits - 1):
            ir.add(
                "RZZ_PRODUCT",
                (wire, wire + 1),
                ParameterRef("input", (base + wire) % dim, spec.scaling),
                ParameterRef("input", (base + wire + 1) % dim, spec.scaling),
            )


def trainable_feature_map(ir: CircuitIR, spec: EncodingSpec) -> None:
    dim = _require_capacity(ir, spec, ir.num_data_qubits)
    weight_index = ir.parameter_groups().get("weight", 0)
    for repeat in range(spec.repeats):
        for wire in ir.data_wires:
            index = repeat * ir.num_data_qubits + wire
            ir.add("RY", wire, ParameterRef("input", index % dim, spec.scaling))
            ir.add("RZ", wire, ParameterRef("weight", weight_index))
            weight_index += 1


def data_reuploading_block(ir: CircuitIR, spec: EncodingSpec, repeat: int) -> None:
    """Encode the complete feature vector in one genuine re-upload cycle.

    Features are spread across the data wires and, whenever a wire receives
    more than one feature, across alternating rotation axes.  Writing several
    features to one wire about a *single* axis would sum the angles and make
    distinct feature vectors indistinguishable; alternating axes keeps the
    loads non-commuting.

    This preserves the distinctness that a same-axis stack destroys.  It is
    not a claim of injectivity: a single qubit carries three real parameters,
    so packing arbitrarily many features onto one wire cannot be lossless in
    general.  Use enough qubits (or ``repeats``) for the feature dimension
    when exact recoverability matters.
    """

    dim = _feature_dim(ir, spec)
    if repeat < 0 or repeat >= spec.repeats:
        raise ConfigurationError(f"Invalid data re-uploading repeat: {repeat}")
    num_wires = ir.num_data_qubits
    # Slots per cycle, so the axis keeps advancing across cycle boundaries as
    # well as within one cycle.
    slots_per_cycle = -(-dim // num_wires)
    for index in range(dim):
        wire = ir.data_wires[index % num_wires]
        slot = repeat * slots_per_cycle + index // num_wires
        ir.add(
            _rotation_gate(spec.rotation, slot),
            wire,
            ParameterRef("input", index, spec.scaling),
            reupload_cycle=repeat,
            feature_index=index,
        )


def data_reuploading(ir: CircuitIR, spec: EncodingSpec) -> None:
    """Encoding-only representation; CircuitFactory interleaves ansatz layers."""

    if spec.repeats < 2:
        raise ConfigurationError(
            "Data re-uploading repeats must be at least 2; use angle encoding for a single load."
        )
    for repeat in range(spec.repeats):
        data_reuploading_block(ir, spec, repeat)


ENCODINGS: dict[str, EncodingBuilder] = {
    "angle_encoding": angle,
    "rotation_encoding": angle,
    "dense_angle_encoding": dense_angle,
    "phase_encoding": phase,
    "basis_encoding": basis,
    "binary_encoding": basis,
    "amplitude_encoding": amplitude,
    "state_preparation_encoding": amplitude,
    "superposition_encoding": superposition,
    "z_feature_map": z_feature_map,
    "zz_feature_map": zz_feature_map,
    "pauli_feature_map": zz_feature_map,
    "iqp_feature_map": iqp,
    "trainable_feature_map": trainable_feature_map,
    "data_reuploading_feature_map": data_reuploading,
}


def parameter_product(left: float, right: float, coefficient: float = 1.0) -> float:
    return coefficient * left * right


__all__ = [
    "ENCODINGS",
    "EncodingBuilder",
    "data_reuploading_block",
    "parameter_product",
]
