"""Backend-independent intermediate representation for quantum circuits."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, cast

from .errors import ConfigurationError
from .specs import (
    freeze_value,
    is_complex_scalar,
    is_finite_number,
    is_finite_real_number,
    require_int,
    to_primitive,
    validate_hermitian_matrix,
    validate_metadata,
)

TWO_QUBIT_GATES = frozenset(
    {"CX", "CNOT", "CZ", "CY", "SWAP", "CRX", "CRY", "CRZ", "RXX", "RYY", "RZZ"}
)


@dataclass(frozen=True, slots=True)
class OperationSignature:
    """Declared wire/parameter arity of an IR operation.

    Signature validation is what stops the IR from silently changing the
    meaning of a circuit.  Without it an ``X`` gate could be declared on two
    wires, ``validate()`` would accept it, and the Qiskit compiler would drop
    the second wire -- producing a circuit that ran without error and computed
    something other than what was written.
    """

    wires: int | None = 1
    params: int | None = 0
    min_wires: int = 1
    max_wires: int | None = None
    #: ``STATE_PREP`` carries one amplitude per basis state, so its parameter
    #: budget follows the wire count instead of a fixed number.
    params_follow_wire_space: bool = False
    #: Whether this operation's parameters may legitimately be complex.  Only
    #: state preparation carries amplitudes; every other declared operation
    #: takes a real angle, phase or bit, so a complex value there is a physics
    #: error rather than a translation detail.  See ``_validate_params``.
    complex_params: bool = False
    description: str = ""

    def validate(self, name: str, wires: tuple[int, ...], params: tuple[Any, ...]) -> None:
        if self.wires is not None and len(wires) != self.wires:
            raise ConfigurationError(
                f"{name} is defined on exactly {self.wires} wire(s); {len(wires)} were given "
                f"({list(wires)}). Extra wires cannot be dropped silently: the meaning of the circuit would change."
            )
        if self.wires is None:
            if len(wires) < self.min_wires:
                raise ConfigurationError(
                    f"{name} requires at least {self.min_wires} wire(s); {len(wires)} were given."
                )
            if self.max_wires is not None and len(wires) > self.max_wires:
                raise ConfigurationError(
                    f"{name} accepts at most {self.max_wires} wire(s); {len(wires)} were given."
                )
        if self.params_follow_wire_space:
            capacity = 2 ** len(wires)
            if not params:
                raise ConfigurationError(f"{name} requires at least one amplitude parameter.")
            if len(params) > capacity:
                raise ConfigurationError(
                    f"{name} accepts at most {capacity} parameter(s) for {len(wires)} wire(s); "
                    f"{len(params)} were given."
                )
            return
        if self.params is not None and len(params) != self.params:
            raise ConfigurationError(
                f"{name} is defined with exactly {self.params} parameter(s); "
                f"{len(params)} were given."
            )

    def validate_params(self, name: str, params: tuple[Any, ...]) -> None:
        """Reject parameter values this operation cannot physically carry.

        A rotation angle is a real number.  ``RX(1j)`` used to validate, then
        Qiskit rejected it ("Invalid param type <class 'complex'>") while
        PennyLane built a non-unitary generator and returned a statevector with
        norm 1.24 -- an unphysical state presented as a result.  The IR is the
        only place both engines share, so this is where the rule belongs.
        """

        if self.complex_params:
            return
        offenders = [
            value
            for value in params
            if not isinstance(value, ParameterRef) and is_complex_scalar(value)
        ]
        if offenders:
            raise ConfigurationError(
                f"{name} parameters must be real numbers; a complex value was given: "
                f"{offenders}. A complex angle produces a non-unitary gate "
                "(Qiskit rejects it, PennyLane returns a state whose norm is not 1). "
                "Complex amplitudes are defined only for STATE_PREP."
            )


_SINGLE_QUBIT_GATES = ("X", "Y", "Z", "H", "S", "T", "SX")
_SINGLE_ROTATIONS = ("RX", "RY", "RZ")
_TWO_QUBIT_PLAIN = ("CX", "CNOT", "CZ", "CY", "SWAP")
_TWO_QUBIT_ROTATIONS = ("CRX", "CRY", "CRZ", "RXX", "RYY", "RZZ")

OPERATION_SIGNATURES: dict[str, OperationSignature] = {
    **{
        name: OperationSignature(wires=1, params=0, description="single-qubit Clifford/Pauli gate")
        for name in _SINGLE_QUBIT_GATES
    },
    **{
        name: OperationSignature(wires=1, params=1, description="single-qubit rotation")
        for name in _SINGLE_ROTATIONS
    },
    "U": OperationSignature(wires=1, params=3, description="general single-qubit U3 rotation"),
    **{
        name: OperationSignature(wires=2, params=0, description="two-qubit non-parametric gate")
        for name in _TWO_QUBIT_PLAIN
    },
    **{
        name: OperationSignature(wires=2, params=1, description="two-qubit parametric gate")
        for name in _TWO_QUBIT_ROTATIONS
    },
    "BASIS": OperationSignature(wires=1, params=1, description="computational basis bit encoding"),
    "STATE_PREP": OperationSignature(
        wires=None,
        params=None,
        min_wires=1,
        params_follow_wire_space=True,
        # Amplitudes are the one genuinely complex parameter family in the IR.
        complex_params=True,
        description="amplitude/state preparation",
    ),
    "RZ_PRODUCT": OperationSignature(
        wires=1, params=2, description="RZ phase from the product of two features"
    ),
    "RZZ_PRODUCT": OperationSignature(
        wires=2, params=2, description="RZZ phase from the product of two features"
    ),
    "RESET": OperationSignature(wires=1, params=0, description="qubit reset"),
    "MEASURE": OperationSignature(wires=1, params=0, description="mid-circuit measurement"),
    "BARRIER": OperationSignature(
        wires=None, params=0, min_wires=1, description="compiler barrier"
    ),
}


def register_operation_signature(
    name: str,
    signature: OperationSignature,
    *,
    replace: bool = False,
) -> None:
    """Declare the arity of a plugin-provided IR operation.

    Operations without a registered signature stay permissive at the IR level
    and are rejected by the backend translators instead, so third-party gates
    keep working; registering a signature moves that rejection earlier, to the
    point where the mistake is actually made.
    """

    key = name.strip().upper()
    if not key:
        raise ConfigurationError("Operation name cannot be empty.")
    if key in OPERATION_SIGNATURES and not replace:
        raise ConfigurationError(
            f"A signature is already registered for {key}; use replace=True to overwrite it."
        )
    OPERATION_SIGNATURES[key] = signature


def operation_signature(name: str) -> OperationSignature | None:
    """Return the declared signature of *name*, or ``None`` when unconstrained."""

    return OPERATION_SIGNATURES.get(name.strip().upper())


@dataclass(frozen=True, slots=True)
class ParameterRef:
    """Symbolic parameter reference resolved at execution time."""

    group: str
    index: int
    scale: float = 1.0
    offset: float = 0.0

    def __post_init__(self) -> None:
        if not self.group.strip():
            raise ConfigurationError("Parameter group cannot be empty.")
        require_int(self.index, label="Parameter index", minimum=0)
        # ``scale``/``offset`` are the affine map onto a rotation angle, so a
        # complex value here reintroduces exactly the non-unitary gate that
        # ``OperationSignature.validate_params`` rejects for literal angles.
        if not is_finite_real_number(self.scale):
            raise ConfigurationError("Parameter scale must be a finite real number.")
        if not is_finite_real_number(self.offset):
            raise ConfigurationError("Parameter offset must be a finite real number.")

    @property
    def name(self) -> str:
        return f"{self.group}[{self.index}]"


ParameterValue = float | int | complex | ParameterRef


@dataclass(frozen=True, slots=True)
class Operation:
    name: str
    wires: tuple[int, ...]
    params: tuple[ParameterValue, ...] = ()
    condition: tuple[int, int] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "metadata", validate_metadata(self.metadata, label="Operation.metadata")
        )
        if not self.name.strip():
            raise ConfigurationError("Operation name cannot be empty.")
        # Gate identity is case-insensitive across the IR; normalizing here
        # means a signature lookup can never be bypassed by casing alone.
        object.__setattr__(self, "name", self.name.strip().upper())
        if not self.wires:
            raise ConfigurationError(f"{self.name} requires at least one wire.")
        if any(isinstance(wire, bool) or not isinstance(wire, int) or wire < 0 for wire in self.wires):
            raise ConfigurationError(f"{self.name} wire values must be non-negative integers.")
        if len(set(self.wires)) != len(self.wires):
            raise ConfigurationError(f"{self.name} cannot use the same wire more than once.")
        invalid_params = [
            value
            for value in self.params
            if not isinstance(value, ParameterRef) and not is_finite_number(value)
        ]
        if invalid_params:
            raise ConfigurationError(f"{self.name} accepts only finite parameters.")
        signature = OPERATION_SIGNATURES.get(self.name)
        if signature is not None:
            signature.validate(self.name, self.wires, self.params)
            signature.validate_params(self.name, self.params)
        if self.condition is not None:
            clbit, value = self.condition
            require_int(clbit, label="Classical condition clbit", minimum=0)
            require_int(value, label="Classical condition value", minimum=0)
            if value > 1:
                # ``condition`` compares a single classical bit, so the only
                # values it can ever match are 0 and 1.  ``value=2`` used to be
                # accepted and then never fire on either engine -- the gate was
                # simply, silently, dead.
                raise ConfigurationError(
                    f"A classical condition is compared against a single clbit; value can only be 0 or 1 "
                    f"({value} was given). For a multi-bit condition, add gates that check each bit "
                    "separately."
                )
        if self.name == "MEASURE":
            if "clbit" not in self.metadata:
                # Every backend reads ``metadata['clbit']`` to place the
                # result.  Without it the IR validated cleanly and the failure
                # surfaced as a bare ``KeyError: 'clbit'`` inside the adapter.
                raise ConfigurationError(
                    "A MEASURE operation must carry its target classical bit in metadata['clbit']; "
                    "use CircuitIR.measure_wire(wire, clbit)."
                )
            require_int(self.metadata["clbit"], label="MEASURE clbit", minimum=0)


@dataclass(frozen=True, slots=True)
class Observable:
    name: str = "Z"
    wires: tuple[int, ...] = (0,)
    coefficient: float = 1.0
    matrix: tuple[tuple[complex, ...], ...] | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ConfigurationError("Observable name cannot be empty.")
        if not self.wires:
            raise ConfigurationError("An observable requires at least one wire.")
        if any(isinstance(wire, bool) or not isinstance(wire, int) or wire < 0 for wire in self.wires):
            raise ConfigurationError("Observable wire values must be non-negative integers.")
        if len(set(self.wires)) != len(self.wires):
            raise ConfigurationError("An observable cannot use the same wire more than once.")
        if not is_finite_real_number(self.coefficient):
            # ``c * H`` is Hermitian only for real ``c``.  ``coefficient=1j``
            # used to validate here and then split the engines: Qiskit raised
            # "Non-Hermitian input observable", PennyLane returned an imaginary
            # expectation value for a quantity that is real by definition.
            raise ConfigurationError(
                "Observable coefficient must be a finite real number; a complex "
                f"coefficient ({self.coefficient!r}) stops the observable from being Hermitian "
                "and the expectation value is no longer a real measurement result. "
                "For complex entries use Observable(matrix=...); the matrix is "
                "validated as Hermitian."
            )
        if self.matrix is None and len(self.wires) != 1:
            # A named Pauli acts on exactly one wire.  Extra wires were dropped
            # silently by the PennyLane translator (which read ``wires[0]``)
            # and rejected by Qiskit, so the same IR meant two different things.
            raise ConfigurationError(
                f"The {self.name} observable is defined on a single wire; "
                f"{len(self.wires)} were given ({list(self.wires)}). For a multi-qubit "
                "observable use Observable(matrix=...)."
            )
        if self.matrix is not None:
            matrix = freeze_value(self.matrix)
            object.__setattr__(self, "matrix", matrix)
            dimension = 2 ** len(self.wires)
            validate_hermitian_matrix(matrix, expected_dimension=dimension)


@dataclass(frozen=True, slots=True)
class Measurement:
    kind: str
    wires: tuple[int, ...]
    observables: tuple[Observable, ...] = ()
    shots: int | None = None
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "options", validate_metadata(self.options, label="Measurement.options")
        )
        if not self.kind.strip():
            raise ConfigurationError("Measurement kind cannot be empty.")
        if not self.wires:
            raise ConfigurationError("Measurement wires cannot be empty.")
        if any(isinstance(wire, bool) or not isinstance(wire, int) or wire < 0 for wire in self.wires):
            raise ConfigurationError("Measurement wire values must be non-negative integers.")
        if len(set(self.wires)) != len(self.wires):
            raise ConfigurationError("Measurement cannot contain the same wire more than once.")
        require_int(self.shots, label="Measurement shots", minimum=1, allow_none=True)


@dataclass(slots=True)
class CircuitIR:
    num_qubits: int
    name: str = "quantum_circuit"
    operations: list[Operation] = field(default_factory=list)
    measurements: list[Measurement] = field(default_factory=list)
    num_clbits: int = 0
    global_phase: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    ancilla_qubits: int = 0

    def __post_init__(self) -> None:
        require_int(self.num_qubits, label="CircuitIR num_qubits", minimum=1)
        if not self.name.strip():
            raise ConfigurationError("CircuitIR name cannot be empty.")
        require_int(self.ancilla_qubits, label="CircuitIR ancilla_qubits", minimum=0)
        if self.ancilla_qubits >= self.num_qubits:
            raise ConfigurationError(
                "CircuitIR ancilla_qubits cannot be negative and must be smaller than the "
                "total qubit count."
            )
        self.validate()

    @property
    def num_data_qubits(self) -> int:
        """Return logical data/ansatz qubits, excluding reserved ancillas."""

        return self.num_qubits - self.ancilla_qubits

    @property
    def data_wires(self) -> tuple[int, ...]:
        return tuple(range(self.num_data_qubits))

    @property
    def ancilla_wires(self) -> tuple[int, ...]:
        return tuple(range(self.num_data_qubits, self.num_qubits))

    def add(
        self,
        name: str,
        wires: int | Iterable[int],
        *params: ParameterValue,
        condition: tuple[int, int] | None = None,
        **metadata: Any,
    ) -> CircuitIR:
        normalized = (wires,) if isinstance(wires, int) else tuple(wires)
        self._validate_wires(normalized)
        self.operations.append(
            Operation(
                name.upper(),
                normalized,
                tuple(params),
                condition=condition,
                metadata=metadata,
            )
        )
        return self

    def add_operation(self, operation: Operation) -> CircuitIR:
        self._validate_wires(operation.wires)
        if operation.condition is not None:
            self._validate_clbit(operation.condition[0])
        self.operations.append(operation)
        return self

    def measure_wire(self, wire: int, clbit: int) -> CircuitIR:
        """Add a true mid-circuit measurement into a classical bit."""

        self._validate_wires((wire,))
        if clbit < 0:
            raise ConfigurationError("A classical bit index cannot be negative.")
        self.num_clbits = max(self.num_clbits, clbit + 1)
        return self.add("MEASURE", wire, clbit=clbit)

    def reset(self, wire: int) -> CircuitIR:
        return self.add("RESET", wire)

    def add_conditional(
        self,
        name: str,
        wires: int | Iterable[int],
        *params: ParameterValue,
        clbit: int,
        value: int = 1,
        **metadata: Any,
    ) -> CircuitIR:
        """Add a gate controlled by a previously measured classical bit."""

        self._validate_clbit(clbit)
        return self.add(name, wires, *params, condition=(clbit, value), **metadata)

    def measure(self, measurement: Measurement) -> CircuitIR:
        self._validate_measurement(measurement)
        self.measurements.append(measurement)
        try:
            self._validate_measurement_contract()
        except ConfigurationError:
            self.measurements.pop()
            raise
        return self

    def validate(self) -> CircuitIR:
        """Revalidate mutable IR collections before compilation or execution."""

        require_int(self.num_clbits, label="CircuitIR num_clbits", minimum=0)
        if not is_finite_real_number(self.global_phase):
            raise ConfigurationError(
                "CircuitIR global_phase must be a finite real number; a complex "
                "global phase stops the e^{i*phase} factor from being unitary."
            )
        for operation in self.operations:
            self._validate_wires(operation.wires)
            if operation.name == "MEASURE":
                self._validate_clbit(int(operation.metadata["clbit"]))
            # Re-check arity here as well: ``operations`` is a plain mutable
            # list, and a signature may have been registered after an
            # operation was first constructed.
            signature = OPERATION_SIGNATURES.get(operation.name)
            if signature is not None:
                signature.validate(operation.name, operation.wires, operation.params)
                signature.validate_params(operation.name, operation.params)
            if operation.condition is not None:
                self._validate_clbit(operation.condition[0])
        for measurement in self.measurements:
            self._validate_measurement(measurement)
        self._validate_measurement_contract()
        return self

    def parameter_count(self, group: str | None = None) -> int:
        refs = {
            (param.group, param.index)
            for operation in self.operations
            for param in operation.params
            if isinstance(param, ParameterRef) and (group is None or param.group == group)
        }
        return len(refs)

    def parameter_groups(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for operation in self.operations:
            for param in operation.params:
                if isinstance(param, ParameterRef):
                    result[param.group] = max(result.get(param.group, 0), param.index + 1)
        return result

    def depth(self) -> int:
        wire_depths = [0] * self.num_qubits
        for op in self.operations:
            layer = max(wire_depths[w] for w in op.wires) + 1
            for wire in op.wires:
                wire_depths[wire] = layer
        return max(wire_depths, default=0)

    def gate_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for op in self.operations:
            counts[op.name] = counts.get(op.name, 0) + 1
        return counts

    def two_qubit_gate_count(self) -> int:
        return sum(op.name in TWO_QUBIT_GATES for op in self.operations)

    def multi_qubit_operation_count(self) -> int:
        """Count all operations touching multiple wires, including state preparation."""

        return sum(len(op.wires) > 1 for op in self.operations)

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "num_qubits": self.num_qubits,
            "num_data_qubits": self.num_data_qubits,
            "ancilla_qubits": self.ancilla_qubits,
            "data_wires": self.data_wires,
            "ancilla_wires": self.ancilla_wires,
            "operations": len(self.operations),
            "depth": self.depth(),
            "two_qubit_gates": self.two_qubit_gate_count(),
            "multi_qubit_operations": self.multi_qubit_operation_count(),
            "parameters": self.parameter_groups(),
            "num_clbits": self.num_clbits,
            "measurements": [m.kind for m in self.measurements],
            "gate_counts": self.gate_counts(),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-safe representation (restored 1.1 public API)."""

        return cast(dict[str, Any], to_primitive(self))

    def copy(self, *, name: str | None = None) -> CircuitIR:
        return CircuitIR(
            num_qubits=self.num_qubits,
            name=name or self.name,
            operations=list(self.operations),
            measurements=list(self.measurements),
            num_clbits=self.num_clbits,
            global_phase=self.global_phase,
            metadata=dict(self.metadata),
            ancilla_qubits=self.ancilla_qubits,
        )

    def _validate_wires(self, wires: Iterable[int]) -> None:
        invalid = [w for w in wires if w < 0 or w >= self.num_qubits]
        if invalid:
            raise ConfigurationError(
                f"Invalid wire values for a {self.num_qubits} qubit circuit: {invalid}"
            )

    def _validate_measurement_contract(self) -> None:
        """Reject terminal-measurement sets that no adapter can honour.

        Both adapters return exactly one terminal measurement, and a second
        entry used to be dropped without a word.  Rather than letting one
        engine return a tuple that the other cannot, the IR now refuses the
        ambiguous circuit and points at the supported multi-output path:
        several observables inside a single expectation/variance measurement.
        """

        if len(self.measurements) <= 1:
            return
        kinds = [measurement.kind for measurement in self.measurements]
        raise ConfigurationError(
            f"CircuitIR can carry only a single terminal measurement; {len(kinds)} were defined "
            f"({kinds}). Adapters execute only the first one, so the extras are rejected "
            "rather than dropped silently. For a multi-output QNN, use more than one "
            "Observable inside a single expectation/variance measurement; if you really "
            "need separate measurements, copy the circuit and run each one separately."
        )

    def _validate_measurement(self, measurement: Measurement) -> None:
        self._validate_wires(measurement.wires)
        for observable in measurement.observables:
            self._validate_wires(observable.wires)
        if measurement.kind == "statevector":
            full_system = tuple(range(self.num_qubits))
            if measurement.wires != full_system:
                raise ConfigurationError(
                    "Subsystem statevector is not defined in the general case. "
                    f"For the full statevector use wires={full_system}; for a subsystem "
                    "choose a reduced_density_matrix/density_matrix measurement."
                )

    def _validate_clbit(self, clbit: int) -> None:
        if clbit < 0 or clbit >= self.num_clbits:
            raise ConfigurationError(
                f"Invalid clbit for a circuit with {self.num_clbits} classical bit(s): {clbit}. "
                "Define the bit first with measure_wire."
            )


def resolve_parameter(
    value: ParameterValue,
    *,
    inputs: Iterable[float] = (),
    weights: Iterable[float] = (),
) -> float | int | complex:
    if not isinstance(value, ParameterRef):
        return value
    if value.group not in {"input", "weight"}:
        raise ConfigurationError(f"Unknown parameter group: {value.group}")
    source = list(inputs if value.group == "input" else weights)
    try:
        raw = source[value.index]
    except IndexError as exc:
        raise ConfigurationError(
            f"Missing {value.group} parameter: index {value.index}, {len(source)} given"
        ) from exc
    return raw * value.scale + value.offset
