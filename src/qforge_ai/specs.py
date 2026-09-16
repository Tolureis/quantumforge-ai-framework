"""Validated, immutable and serializable experiment specifications."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

import yaml

from .errors import ConfigurationError


class FrozenDict(Mapping[str, Any]):
    """Small recursively hashable mapping used by frozen public specifications.

    ``Mapping`` fields used to contain mutable ``dict`` instances.  Frozen
    dataclasses therefore looked hashable but failed at runtime.  FrozenDict
    closes that gap without leaking a third-party immutable collection type.
    """

    __slots__ = ("_data", "_hash")

    def __init__(self, values: Mapping[str, Any] | None = None) -> None:
        self._data = {str(key): freeze_value(value) for key, value in (values or {}).items()}
        self._hash: int | None = None

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash(tuple(sorted(self._data.items())))
        return self._hash

    def __repr__(self) -> str:
        return f"FrozenDict({self._data!r})"


def freeze_value(value: Any) -> Any:
    """Recursively convert mutable containers into hashable equivalents."""

    if isinstance(value, FrozenDict):
        return value
    if isinstance(value, Mapping):
        return FrozenDict(value)
    if isinstance(value, (list, tuple)):
        return tuple(freeze_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(freeze_value(item) for item in value)
    # NumPy arrays are the common unhashable leaf in user metadata; freezing
    # them to nested tuples is what keeps a frozen spec genuinely hashable.
    if _is_numpy_array(value):
        return freeze_value(value.tolist())
    if _is_numpy_scalar(value):
        return value.item()
    return value


def _is_numpy_array(value: Any) -> bool:
    return hasattr(value, "tolist") and hasattr(value, "shape") and hasattr(value, "dtype")


def _is_numpy_scalar(value: Any) -> bool:
    return (
        hasattr(value, "item")
        and hasattr(value, "dtype")
        and getattr(value, "shape", None) == ()
    )


#: Leaves ``to_primitive`` can render directly.
_PRIMITIVE_LEAVES = (str, bool, int, float, type(None))


def to_primitive(value: Any, *, strict: bool = True, path: str = "") -> Any:
    """Return a JSON/YAML-safe tree for specs and framework metadata.

    The serialization contract used to be open-ended: any object could sit in
    a spec and ``to_dict()`` would hand it back untouched, so the failure only
    appeared later as ``Object of type PosixPath is not JSON serializable``
    with no indication of which field caused it.

    ``strict=True`` (the default) raises :class:`ConfigurationError` naming the
    exact field path.  ``strict=False`` renders unsupported leaves as a type
    summary and is meant for best-effort telemetry -- experiment tracking
    should never abort a training run over an unprintable option.
    """

    if isinstance(value, StrEnum):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: to_primitive(
                getattr(value, item.name),
                strict=strict,
                path=f"{path}.{item.name}" if path else item.name,
            )
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): to_primitive(
                item, strict=strict, path=f"{path}.{key}" if path else str(key)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [
            to_primitive(item, strict=strict, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, complex):
        if not is_finite_number(value):
            raise ConfigurationError(f"{path or 'value'} cannot carry a non-finite complex number.")
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, float) and strict and not math.isfinite(value):
        raise ConfigurationError(
            f"{path or 'value'} cannot carry a non-finite number ({value!r}); "
            "NaN and Infinity have no JSON representation."
        )
    if isinstance(value, _PRIMITIVE_LEAVES):
        return value
    if isinstance(value, Path):
        return str(value)
    if _is_numpy_array(value):
        return to_primitive(value.tolist(), strict=strict, path=path)
    if _is_numpy_scalar(value):
        return to_primitive(value.item(), strict=strict, path=path)
    if strict:
        raise ConfigurationError(
            f"{path or 'value'} cannot be converted to JSON: {type(value).__name__}. "
            "Spec metadata must carry only JSON-compatible data; keep engine objects "
            "under options and use to_primitive(strict=False) for serialization."
        )
    return f"<{type(value).__name__}>"


def validate_metadata(value: Any, *, label: str) -> Any:
    """Validate and normalize a metadata tree at construction time.

    Metadata is pure data: it must survive ``to_dict()``/``to_yaml()`` and it
    must not make a frozen spec unhashable.  Rejecting a bad value here points
    at the line that set it, rather than at whichever later call happened to
    serialize the spec.  ``Path`` and NumPy values are accepted and stored in
    their serialized form.
    """

    normalized = _normalize_metadata(value, label=label)
    return freeze_value(normalized)


def _normalize_metadata(value: Any, *, label: str) -> Any:
    if isinstance(value, StrEnum):
        return str(value)
    if isinstance(value, Mapping):
        # Keys are stringified for JSON/YAML, so ``1`` and ``"1"`` would land
        # on the same entry and one of them would vanish without a word.
        normalized: dict[str, Any] = {}
        seen: dict[str, Any] = {}
        for key, item in value.items():
            text = str(key)
            if text in seen:
                raise ConfigurationError(
                    f"{label} keys are converted to text for JSON and {seen[text]!r} and "
                    f"{key!r} both collapse onto the same {text!r} key. One of them is being "
                    "rejected rather than silently lost; make the keys unique."
                )
            seen[text] = key
            normalized[text] = _normalize_metadata(item, label=f"{label}.{text}")
        return normalized
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(
            _normalize_metadata(item, label=f"{label}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, (float, complex)) and not is_finite_number(value):
        # NaN and Infinity have no JSON representation, so they survived
        # ``to_dict()`` only to produce a file no strict JSON reader accepts.
        raise ConfigurationError(
            f"{label} cannot carry a non-finite number ({value!r}). NaN and Infinity "
            "have no JSON representation, so this is rejected here rather than "
            "breaking silently during serialization."
        )
    if isinstance(value, _PRIMITIVE_LEAVES) or isinstance(value, complex):
        return value
    if isinstance(value, Path):
        return str(value)
    if _is_numpy_array(value):
        return _normalize_metadata(value.tolist(), label=label)
    if _is_numpy_scalar(value):
        return value.item()
    raise ConfigurationError(
        f"{label} accepts only JSON-compatible data; {type(value).__name__} was given. "
        "Engine objects (observable, interpret, etc.) must be kept under options."
    )


#: Option keys the core builders actually read, per spec section.
#:
#: ``BackendSpec`` and ``ModelSpec`` have validated their own ``options`` since
#: 1.2.3; the four sections below did not, so a misspelled or misplaced key was
#: accepted, carried through ``to_dict()``/``to_yaml()`` and never read -- the
#: experiment ran with the default the author believed they had overridden.
#: Plugins that introduce a component reading a new key declare it with
#: :func:`register_option_keys`.
SPEC_OPTION_KEYS: dict[str, set[str]] = {
    # No core encoding reads an option: the encoding vocabulary lives in the
    # typed fields (rotation, scaling, repeats, normalize, feature_dimension).
    "encoding": set(),
    "ansatz": {"layer_offset", "seed", "custom_edges"},
    "measurement": set(),
    "training": set(),
}


def register_option_keys(section: str, *keys: str) -> None:
    """Declare option keys a plugin's component reads.

    ``register_option_keys('encoding', 'my_plugin_depth')`` makes that key
    legal in ``EncodingSpec.options``.  Without a declaration an unknown key is
    a typo, and typos in an experiment definition are rejected rather than
    carried.
    """

    if section not in SPEC_OPTION_KEYS:
        raise ConfigurationError(
            f"Unknown options section: {section!r}; "
            f"valid ones: {sorted(SPEC_OPTION_KEYS)}."
        )
    SPEC_OPTION_KEYS[section].update(str(key) for key in keys)


def _validate_option_keys(options: Mapping[str, Any], *, section: str, label: str) -> None:
    allowed = SPEC_OPTION_KEYS[section]
    unknown = sorted(set(options) - allowed)
    if unknown:
        raise ConfigurationError(
            f"{label} does not use these keys: {unknown}. They are rejected rather than "
            f"being carried along silently and never read. Valid ones for this section: "
            f"{sorted(allowed) or '(none)'}. If a plugin reads a key, declare it with "
            f"qforge_ai.specs.register_option_keys({section!r}, ...)."
        )


def _freeze_metadata_field(instance: Any, name: str, *, label: str) -> None:
    object.__setattr__(
        instance, name, validate_metadata(getattr(instance, name), label=label)
    )


def _freeze_field(instance: Any, name: str) -> None:
    object.__setattr__(instance, name, freeze_value(getattr(instance, name)))


class Engine(StrEnum):
    QISKIT = "qiskit"
    PENNYLANE = "pennylane"
    IR = "ir"


class ModelMode(StrEnum):
    QUANTUM_ONLY = "quantum_only"
    HYBRID = "hybrid"
    QUANTUM_KERNEL = "quantum_kernel"
    ENSEMBLE = "ensemble"
    QUANVOLUTION = "quanvolution"
    DISTRIBUTED_QNN = "distributed_qnn"


class ComponentStatus(StrEnum):
    NATIVE = "native"
    # 1.0/1.1 compatibility alias.  ``BUILTIN`` and ``NATIVE`` intentionally
    # serialize to the same stable value.
    BUILTIN = "native"
    COMPOSITE = "composite"
    EXPERIMENTAL = "experimental"
    DESCRIPTOR = "descriptor"


class GradientMethod(StrEnum):
    AUTO = "auto"
    PARAMETER_SHIFT = "parameter_shift"
    ADJOINT = "adjoint"
    BACKPROP = "backprop"
    FINITE_DIFF = "finite_diff"
    SPSA = "spsa"
    NATURAL_GRADIENT = "natural_gradient"


def is_finite_number(value: Any) -> bool:
    """Return whether *value* is a finite real or complex scalar."""

    try:
        number = complex(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(number.real) and math.isfinite(number.imag)


def is_complex_scalar(value: Any) -> bool:
    """Return whether *value* carries a complex *type*, imaginary part or not.

    ``1+0j`` is numerically real but is still a ``complex`` object, and the two
    engines treat it as one: Qiskit refuses it outright ("Invalid param type
    <class 'complex'> for gate rx") while PennyLane builds a non-unitary
    rotation from it.  Type is therefore the honest test rather than the
    imaginary part -- and it also covers NumPy's complex dtypes, which are not
    instances of :class:`complex`.
    """

    if isinstance(value, complex):
        return True
    dtype = getattr(value, "dtype", None)
    return bool(getattr(dtype, "kind", "") == "c")


def is_finite_real_number(value: Any) -> bool:
    """Return whether *value* is a finite scalar with no imaginary component.

    Rotation angles, parameter affine coefficients, global phases and
    observable coefficients are real physical quantities: a complex angle makes
    the generated gate non-unitary and a complex coefficient makes the
    observable non-Hermitian.  Both used to pass IR validation and then mean
    two different things per engine -- Qiskit raised, PennyLane produced a
    statevector with norm != 1 or a complex expectation value.  This is the
    predicate that keeps such a circuit out of the IR in the first place.
    """

    if is_complex_scalar(value):
        return False
    return is_finite_number(value)


def require_int(
    value: Any,
    *,
    label: str,
    minimum: int | None = None,
    allow_none: bool = False,
) -> Any:
    """Validate a genuine integer field and reject ``bool``.

    ``bool`` is a subclass of ``int``, so ``epochs=True`` used to be accepted
    and silently trained for exactly one epoch -- and ``shots=True`` asked for
    a single shot.  A boolean in an integer field is always a mistake, so it is
    named as one instead of being coerced.
    """

    if value is None:
        if allow_none:
            return None
        raise ConfigurationError(f"{label} cannot be None.")
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(
            f"{label} must be an integer; {type(value).__name__} was given ({value!r})."
        )
    if minimum is not None and value < minimum:
        raise ConfigurationError(f"{label} must be at least {minimum}; {value} was given.")
    return value


def require_float(
    value: Any,
    *,
    label: str,
    minimum: float | None = None,
    maximum: float | None = None,
    allow_none: bool = False,
    exclusive_minimum: bool = False,
) -> Any:
    """Validate a finite real field and reject ``bool``."""

    if value is None:
        if allow_none:
            return None
        raise ConfigurationError(f"{label} cannot be None.")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(
            f"{label} must be numeric; {type(value).__name__} was given ({value!r})."
        )
    number = float(value)
    if not math.isfinite(number):
        raise ConfigurationError(f"{label} must be finite; {value!r} was given.")
    if minimum is not None:
        if exclusive_minimum and number <= minimum:
            raise ConfigurationError(f"{label} must be greater than {minimum}.")
        if not exclusive_minimum and number < minimum:
            raise ConfigurationError(f"{label} must be at least {minimum}.")
    if maximum is not None and number > maximum:
        raise ConfigurationError(f"{label} must be at most {maximum}.")
    return number


def validate_hermitian_matrix(
    matrix: tuple[tuple[complex, ...], ...],
    *,
    expected_dimension: int | None = None,
    label: str = "Observable matrix",
) -> None:
    """Validate a finite, square Hermitian matrix without optional dependencies."""

    dimension = len(matrix)
    if dimension == 0 or any(len(row) != dimension for row in matrix):
        raise ConfigurationError(f"{label} must be a non-empty square matrix.")
    if expected_dimension is not None and dimension != expected_dimension:
        raise ConfigurationError(
            f"{label} must be {expected_dimension}x{expected_dimension}; "
            f"got {dimension}x{dimension}."
        )
    if dimension & (dimension - 1):
        raise ConfigurationError(f"{label} dimension must be a power of 2 for a qubit space.")
    for row in matrix:
        if any(not is_finite_number(value) for value in row):
            raise ConfigurationError(f"{label} must contain only finite numbers.")
    tolerance = 1e-10
    for row_index in range(dimension):
        for column_index in range(row_index, dimension):
            left = complex(matrix[row_index][column_index])
            right = complex(matrix[column_index][row_index]).conjugate()
            if not math.isclose(left.real, right.real, rel_tol=tolerance, abs_tol=tolerance) or not math.isclose(
                left.imag, right.imag, rel_tol=tolerance, abs_tol=tolerance
            ):
                raise ConfigurationError(f"{label} must be Hermitian (A = A†).")


def validate_output_shape(value: Any, *, label: str = "output_shape") -> tuple[int, ...]:
    """Normalize a positive QNN output shape and reject booleans/empty tuples."""

    values = value if isinstance(value, tuple) else (value,)
    if not values:
        raise ConfigurationError(f"{label} cannot be empty.")
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in values):
        raise ConfigurationError(f"{label} must be a positive integer or a tuple of positive integers.")
    return tuple(values)


@dataclass(frozen=True, slots=True)
class EncodingSpec:
    name: str = "angle"
    feature_dimension: int | None = None
    rotation: str = "Y"
    scaling: float = 1.0
    repeats: int = 1
    normalize: bool = False
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _freeze_metadata_field(self, "options", label="EncodingSpec.options")
        _validate_option_keys(self.options, section="encoding", label="EncodingSpec.options")
        if not self.name.strip():
            raise ConfigurationError("Encoding name cannot be empty.")
        require_int(
            self.feature_dimension,
            label="feature_dimension",
            minimum=1,
            allow_none=True,
        )
        require_int(self.repeats, label="Encoding repeats", minimum=1)
        require_float(self.scaling, label="Encoding scaling")
        if self.rotation.upper() not in {"X", "Y", "Z"}:
            raise ConfigurationError("rotation must be X, Y or Z.")


#: Entanglement topologies the core builder implements.
ENTANGLEMENT_TOPOLOGIES = frozenset(
    {
        "none",
        "linear",
        "reverse_linear",
        "ring",
        "circular",
        "full",
        "pairwise",
        "alternating",
        "star",
        "grid",
        "random",
        "hardware",
        "user_defined",
    }
)

#: Catalog display keys mapped onto those topologies.  The catalog advertised
#: every entanglement entry as ``native`` while ``AnsatzSpec`` rejected all
#: twelve of the catalog keys, so a name copied straight out of ``qforge
#: catalog`` could not be used.  Accepting both spellings makes the catalog
#: listing directly usable.
_ENTANGLEMENT_ALIASES = {
    "linear_entanglement": "linear",
    "reverse_linear_entanglement": "reverse_linear",
    "circular_entanglement": "circular",
    "ring_entanglement": "ring",
    "full_entanglement": "full",
    "pairwise_entanglement": "pairwise",
    "alternating_entanglement": "alternating",
    "star_entanglement": "star",
    "grid_entanglement": "grid",
    "random_entanglement": "random",
    "hardware_topology_entanglement": "hardware",
    "user_defined_entanglement": "user_defined",
    "no_entanglement": "none",
}


def normalize_entanglement(value: str) -> str | None:
    """Map a topology or catalog name onto a core topology, else ``None``."""

    key = "_".join(str(value).strip().lower().replace("-", "_").split())
    key = _ENTANGLEMENT_ALIASES.get(key, key)
    return key if key in ENTANGLEMENT_TOPOLOGIES else None


#: Catalog observable keys mapped onto the IR's observable names.
_OBSERVABLE_ALIASES = {
    "pauli_x": "X",
    "pauli_y": "Y",
    "pauli_z": "Z",
    "identity": "I",
    "hadamard_observable": "H",
    "hadamard": "H",
    "hermitian_observable": "HERMITIAN",
    "custom_observable": "HERMITIAN",
}

#: Observables both adapters evaluate natively.  ``HERMITIAN`` is the explicit
#: matrix path and requires ``observable_matrix``.
IR_OBSERVABLES = frozenset({"X", "Y", "Z", "I", "H", "HERMITIAN"})


def normalize_observable(value: str) -> str | None:
    """Map a catalog or shorthand observable name onto an IR observable."""

    raw = str(value).strip()
    key = "_".join(raw.lower().replace("-", "_").split())
    resolved = _OBSERVABLE_ALIASES.get(key, raw.upper())
    return resolved if resolved in IR_OBSERVABLES else None


@dataclass(frozen=True, slots=True)
class AnsatzSpec:
    name: str = "hardware_efficient"
    layers: int = 1
    entanglement: str = "linear"
    rotation_blocks: tuple[str, ...] = ("RY", "RZ")
    skip_final_rotation: bool = False
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _freeze_field(self, "rotation_blocks")
        _freeze_metadata_field(self, "options", label="AnsatzSpec.options")
        _validate_option_keys(self.options, section="ansatz", label="AnsatzSpec.options")
        if not self.name.strip():
            raise ConfigurationError("Ansatz name cannot be empty.")
        if isinstance(self.layers, bool) or not isinstance(self.layers, int) or self.layers < 1:
            raise ConfigurationError(
                "Ansatz layers must be an integer of at least 1; for a non-parametric circuit "
                "use an explicit identity/custom builder."
            )
        entanglement = normalize_entanglement(self.entanglement)
        if entanglement is None:
            raise ConfigurationError(
                f"Unknown entanglement: {self.entanglement}. "
                f"Supported: {sorted(ENTANGLEMENT_TOPOLOGIES)} "
                f"(catalog names are also accepted, e.g. 'linear_entanglement')."
            )
        object.__setattr__(self, "entanglement", entanglement)
        if not self.rotation_blocks:
            raise ConfigurationError("rotation_blocks must contain at least one rotation gate.")
        invalid_rotations = [
            block
            for block in self.rotation_blocks
            if not isinstance(block, str) or block.upper() not in {"RX", "RY", "RZ"}
        ]
        if invalid_rotations:
            raise ConfigurationError(
                f"Invalid rotation_blocks: {invalid_rotations}; supported are RX, RY and RZ."
            )


@dataclass(frozen=True, slots=True)
class MeasurementSpec:
    name: str = "expectation"
    wires: tuple[int, ...] | None = None
    observable: str = "Z"
    observable_matrix: tuple[tuple[complex, ...], ...] | None = None
    shots: int | None = None
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.wires is not None:
            _freeze_field(self, "wires")
        if self.observable_matrix is not None:
            _freeze_field(self, "observable_matrix")
        _freeze_metadata_field(self, "options", label="MeasurementSpec.options")
        _validate_option_keys(self.options, section="measurement", label="MeasurementSpec.options")
        if not self.name.strip():
            raise ConfigurationError("Measurement name cannot be empty.")
        if not self.observable.strip():
            raise ConfigurationError("Observable name cannot be empty.")
        observable = normalize_observable(self.observable)
        if observable is None:
            raise ConfigurationError(
                f"The core adapters cannot evaluate this observable directly: "
                f"{self.observable!r}. Supported: {sorted(IR_OBSERVABLES)} "
                "(catalog names are also accepted, e.g. 'pauli_x'). For an arbitrary Hermitian "
                "use observable='hermitian' together with observable_matrix."
            )
        if observable == "HERMITIAN" and self.observable_matrix is None:
            raise ConfigurationError(
                "observable_matrix is required for a hermitian/custom observable."
            )
        object.__setattr__(self, "observable", observable)
        if self.wires is not None:
            if not self.wires:
                raise ConfigurationError("Measurement wires cannot be empty.")
            if any(isinstance(wire, bool) or not isinstance(wire, int) or wire < 0 for wire in self.wires):
                raise ConfigurationError("Measurement wires must be non-negative integers.")
            if len(set(self.wires)) != len(self.wires):
                raise ConfigurationError("Measurement cannot contain the same wire more than once.")
        if self.observable_matrix is not None:
            validate_hermitian_matrix(self.observable_matrix)
        require_int(self.shots, label="Measurement shots", minimum=1, allow_none=True)


@dataclass(frozen=True, slots=True)
class NoiseSpec:
    name: str = "none"
    probability: float = 0.0
    one_qubit_probability: float | None = None
    two_qubit_probability: float | None = None
    readout_probability: float | None = None
    t1: float | None = None
    t2: float | None = None
    gate_time: float | None = None
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _freeze_metadata_field(self, "options", label="NoiseSpec.options")
        if not self.name.strip():
            raise ConfigurationError("Noise name cannot be empty.")
        for name in (
            "probability",
            "one_qubit_probability",
            "two_qubit_probability",
            "readout_probability",
        ):
            require_float(
                getattr(self, name), label=name, minimum=0.0, maximum=1.0, allow_none=True
            )
        require_float(
            self.t1, label="t1", minimum=0.0, exclusive_minimum=True, allow_none=True
        )
        require_float(
            self.t2, label="t2", minimum=0.0, exclusive_minimum=True, allow_none=True
        )
        require_float(
            self.gate_time,
            label="gate_time",
            minimum=0.0,
            exclusive_minimum=True,
            allow_none=True,
        )
        if self.t1 is not None and self.t2 is not None and self.t2 > 2 * self.t1:
            raise ConfigurationError("Physical thermal relaxation requires t2 <= 2*t1.")
        if self.name.replace("_noise", "") == "thermal_relaxation" and any(
            value is None for value in (self.t1, self.t2, self.gate_time)
        ):
            raise ConfigurationError("Thermal relaxation requires t1, t2 and gate_time.")


@dataclass(frozen=True, slots=True)
class CircuitSpec:
    num_qubits: int
    encoding: EncodingSpec = field(default_factory=EncodingSpec)
    ansatz: AnsatzSpec = field(default_factory=AnsatzSpec)
    measurement: MeasurementSpec = field(default_factory=MeasurementSpec)
    name: str = "quantum_circuit"
    ancilla_qubits: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _freeze_metadata_field(self, "metadata", label="CircuitSpec.metadata")
        if not self.name.strip():
            raise ConfigurationError("Circuit name cannot be empty.")
        require_int(self.num_qubits, label="num_qubits", minimum=1)
        require_int(self.ancilla_qubits, label="ancilla_qubits", minimum=0)
        if self.measurement.wires is not None:
            total_qubits = self.num_qubits + self.ancilla_qubits
            invalid = [w for w in self.measurement.wires if w < 0 or w >= total_qubits]
            if invalid:
                raise ConfigurationError(f"Invalid measurement wire values: {invalid}")

    def to_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], to_primitive(self))


#: Aer simulation methods the core adapter understands.  Whether the *installed*
#: build offers each one is a separate, machine-specific question answered by
#: :class:`~qforge_ai.capabilities.CapabilityRegistry`.
QISKIT_SIMULATION_METHODS = frozenset(
    {
        "automatic",
        "statevector",
        "density_matrix",
        "matrix_product_state",
        "stabilizer",
        "extended_stabilizer",
        "unitary",
        "superop",
        "tensor_network",
    }
)

#: PennyLane devices that actually place work on a GPU.
PENNYLANE_GPU_DEVICES = frozenset({"lightning.gpu", "lightning.kokkos"})

#: Option keys each engine reads.  Everything else is a mistake, not a
#: forward-compatible extension: plugins register their own BackendSpec types.
BACKEND_OPTION_KEYS: dict[Engine, frozenset[str]] = {
    Engine.QISKIT: frozenset(
        {"backend_options", "run_options", "primitive_options", "gradient_options"}
    ),
    Engine.PENNYLANE: frozenset({"device_options", "gradient_options"}),
    Engine.IR: frozenset(),
}


@dataclass(frozen=True, slots=True)
class BackendSpec:
    engine: Engine = Engine.IR
    device: str = "statevector"
    compute_device: str = "CPU"
    shots: int | None = None
    method: str = "automatic"
    gradient: GradientMethod = GradientMethod.AUTO
    seed: int = 42
    noise: NoiseSpec = field(default_factory=NoiseSpec)
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _freeze_field(self, "options")
        require_int(self.shots, label="Backend shots", minimum=1, allow_none=True)
        require_int(self.seed, label="Backend seed", minimum=0)
        if not self.device.strip():
            raise ConfigurationError("Backend device cannot be empty.")
        if not self.method.strip():
            raise ConfigurationError("Backend method cannot be empty.")
        compute = self.compute_device.upper()
        if compute not in {"CPU", "GPU"}:
            raise ConfigurationError("compute_device must be CPU or GPU.")
        object.__setattr__(self, "compute_device", compute)
        if self.engine == Engine.QISKIT and self.device not in {
            "aer",
            "aer_simulator",
            "statevector",
        }:
            raise ConfigurationError(
                "The core Qiskit adapter accepts aer, aer_simulator or statevector as the "
                "device; for real hardware use a backend plugin."
            )
        self._validate_options()
        self._validate_method()
        self._validate_compute_device()

    def _validate_options(self) -> None:
        """Reject option keys the selected engine will never read.

        An unrecognised key used to be carried around and dropped at the point
        of use, so a misspelled ``run_options`` (or a Qiskit-only setting on a
        PennyLane backend) silently produced a run with default settings.
        """

        allowed = BACKEND_OPTION_KEYS[self.engine]
        unknown = sorted(set(self.options) - allowed)
        if unknown:
            raise ConfigurationError(
                f"{self.engine} backend does not use these options keys: {unknown}. "
                f"Valid ones: {sorted(allowed)}"
            )
        for key in ("backend_options", "run_options", "primitive_options",
                    "device_options", "gradient_options"):
            value = self.options.get(key)
            if value is not None and not isinstance(value, Mapping):
                raise ConfigurationError(f"backend.options.{key} must be a mapping.")

    def _validate_method(self) -> None:
        """Reject simulation methods the engine cannot honour.

        PennyLane has no simulation-method concept, so an invalid ``method``
        was previously accepted and ignored outright.
        """

        if self.engine == Engine.QISKIT:
            if self.method not in QISKIT_SIMULATION_METHODS:
                raise ConfigurationError(
                    f"Unknown Qiskit simulator method: {self.method}. "
                    f"Supported: {sorted(QISKIT_SIMULATION_METHODS)}"
                )
            return
        if self.method != "automatic":
            raise ConfigurationError(
                f"The {self.engine} engine has no simulator method concept; "
                f"method must be 'automatic' (got {self.method!r}). "
                "In PennyLane acceleration is selected by device name, e.g. device='lightning.qubit'."
            )

    def _validate_compute_device(self) -> None:
        """Make ``compute_device='GPU'`` mean the same thing on both engines.

        Qiskit selects a GPU through Aer's ``device`` option, but PennyLane
        selects it through the *device name*.  Accepting ``compute_device='GPU'``
        on a ``default.qubit`` backend therefore produced a CPU run that
        claimed to be a GPU run.
        """

        if self.compute_device != "GPU":
            return
        if self.engine == Engine.QISKIT:
            return
        if self.engine == Engine.PENNYLANE:
            if self.device not in PENNYLANE_GPU_DEVICES:
                raise ConfigurationError(
                    f"In PennyLane the GPU is selected by device name; compute_device='GPU' is "
                    f"only meaningful with the {sorted(PENNYLANE_GPU_DEVICES)} devices "
                    f"(got device={self.device!r}). Otherwise the run would happen on the CPU."
                )
            return
        raise ConfigurationError(
            "The IR engine does not execute; compute_device='GPU' is meaningless."
        )


@dataclass(frozen=True, slots=True)
class ModelSpec:
    mode: ModelMode = ModelMode.HYBRID
    template: str = "auto"
    input_dimension: int | None = None
    hidden_dimensions: tuple[int, ...] = ()
    output_dimension: int = 2
    dropout: float = 0.0
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _freeze_field(self, "hidden_dimensions")
        _freeze_field(self, "options")
        require_int(self.input_dimension, label="input_dimension", minimum=1, allow_none=True)
        if not self.template.strip():
            raise ConfigurationError("Model template cannot be empty; you can use 'auto'.")
        require_int(self.output_dimension, label="output_dimension", minimum=1)
        invalid_hidden = [
            width
            for width in self.hidden_dimensions
            if isinstance(width, bool) or not isinstance(width, int) or width < 1
        ]
        if invalid_hidden:
            raise ConfigurationError(
                f"hidden_dimensions must contain only positive integers: {invalid_hidden}"
            )
        require_float(self.dropout, label="dropout", minimum=0.0)
        if self.dropout >= 1.0:
            raise ConfigurationError("dropout must be in the range [0, 1).")
        if "sampler_output_shape" in self.options:
            validate_output_shape(
                self.options["sampler_output_shape"], label="sampler_output_shape"
            )


@dataclass(frozen=True, slots=True)
class TrainingSpec:
    epochs: int = 20
    batch_size: int = 32
    optimizer: str = "adam"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    loss: str = "cross_entropy"
    gradient_clip: float | None = None
    early_stopping_patience: int | None = None
    deterministic: bool = True
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _freeze_metadata_field(self, "options", label="TrainingSpec.options")
        _validate_option_keys(self.options, section="training", label="TrainingSpec.options")
        require_int(self.epochs, label="epochs", minimum=1)
        require_int(self.batch_size, label="batch_size", minimum=1)
        require_float(
            self.learning_rate, label="learning_rate", minimum=0.0, exclusive_minimum=True
        )
        require_float(self.weight_decay, label="weight_decay", minimum=0.0)
        require_float(
            self.gradient_clip,
            label="gradient_clip",
            minimum=0.0,
            exclusive_minimum=True,
            allow_none=True,
        )
        require_int(
            self.early_stopping_patience,
            label="early_stopping_patience",
            minimum=1,
            allow_none=True,
        )
        if not isinstance(self.deterministic, bool):
            raise ConfigurationError("deterministic must be a bool.")


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    name: str
    circuit: CircuitSpec
    backend: BackendSpec = field(default_factory=BackendSpec)
    model: ModelSpec = field(default_factory=ModelSpec)
    training: TrainingSpec = field(default_factory=TrainingSpec)
    seed: int = 42
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _freeze_field(self, "tags")
        _freeze_metadata_field(self, "metadata", label="ExperimentSpec.metadata")
        if not self.name.strip():
            raise ConfigurationError("Experiment name cannot be empty.")
        require_int(self.seed, label="Experiment seed", minimum=0)
        if any(not isinstance(tag, str) or not tag.strip() for tag in self.tags):
            raise ConfigurationError("Experiment tags cannot contain an empty value.")
        if len(set(self.tags)) != len(self.tags):
            raise ConfigurationError("Experiment tags cannot contain a duplicate value.")

    def to_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], to_primitive(self))

    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExperimentSpec:
        """Build a spec from a plain mapping, rejecting anything it cannot mean.

        The YAML/dict path used to be *looser* than direct construction, which
        is exactly backwards: a configuration file is written by hand, so it is
        the path where a typo is most likely and least visible.  ``seed=True``
        became 1, ``seed=2.5`` became 2, ``seed='7'`` became 7, an unknown
        section key was ignored, and ``tags='ab'`` became ``('a', 'b')``.  Every
        value now reaches the dataclass unchanged and is validated there, and
        every key is checked against the fields that exist.
        """

        _require_mapping(data, label="ExperimentSpec")
        top = _checked_section(data, cls, label="ExperimentSpec")
        circuit_data = _checked_section(
            _require_mapping(_require_key(top, "circuit"), label="circuit"),
            CircuitSpec,
            label="circuit",
        )
        measurement_data = _sequence_field(
            _checked_section(
                _require_mapping(circuit_data.get("measurement", {}), label="circuit.measurement"),
                MeasurementSpec,
                label="circuit.measurement",
            ),
            "wires",
            label="circuit.measurement.wires",
        )
        if measurement_data.get("observable_matrix") is not None:
            measurement_data["observable_matrix"] = _complex_matrix(
                measurement_data["observable_matrix"]
            )
        encoding_data = _checked_section(
            _require_mapping(circuit_data.get("encoding", {}), label="circuit.encoding"),
            EncodingSpec,
            label="circuit.encoding",
        )
        ansatz_data = _sequence_field(
            _checked_section(
                _require_mapping(circuit_data.get("ansatz", {}), label="circuit.ansatz"),
                AnsatzSpec,
                label="circuit.ansatz",
            ),
            "rotation_blocks",
            label="circuit.ansatz.rotation_blocks",
        )
        circuit = CircuitSpec(
            **{
                **circuit_data,
                "encoding": EncodingSpec(**encoding_data),
                "ansatz": AnsatzSpec(**ansatz_data),
                "measurement": MeasurementSpec(**measurement_data),
            }
        )
        backend_data = _checked_section(
            _require_mapping(top.get("backend", {}), label="backend"),
            BackendSpec,
            label="backend",
        )
        backend_data["engine"] = _enum_value(
            Engine, backend_data.get("engine", Engine.IR), label="backend.engine"
        )
        backend_data["gradient"] = _enum_value(
            GradientMethod,
            backend_data.get("gradient", GradientMethod.AUTO),
            label="backend.gradient",
        )
        backend_data["noise"] = NoiseSpec(
            **_checked_section(
                _require_mapping(backend_data.get("noise", {}), label="backend.noise"),
                NoiseSpec,
                label="backend.noise",
            )
        )
        model_data = _sequence_field(
            _checked_section(
                _require_mapping(top.get("model", {}), label="model"),
                ModelSpec,
                label="model",
            ),
            "hidden_dimensions",
            label="model.hidden_dimensions",
        )
        model_data["mode"] = _enum_value(
            ModelMode, model_data.get("mode", ModelMode.HYBRID), label="model.mode"
        )
        training_data = _checked_section(
            _require_mapping(top.get("training", {}), label="training"),
            TrainingSpec,
            label="training",
        )
        return cls(
            name=_require_key(top, "name"),
            circuit=circuit,
            backend=BackendSpec(**backend_data),
            model=ModelSpec(**model_data),
            training=TrainingSpec(**training_data),
            # No ``int(...)``: a bool, a float or a numeric string in an integer
            # field is a mistake in the configuration, and ``require_int``
            # inside ``__post_init__`` says so by name.
            seed=top.get("seed", 42),
            tags=_sequence(top.get("tags", ()), label="tags"),
            metadata=top.get("metadata", {}),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentSpec:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, Mapping):
            raise ConfigurationError("The YAML root must be a mapping.")
        return cls.from_dict(data)


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(
            f"{label} must be a mapping; {type(value).__name__} was given."
        )
    return value


def _require_key(data: Mapping[str, Any], key: str) -> Any:
    try:
        return data[key]
    except KeyError:
        raise ConfigurationError(f"Required field missing from the experiment definition: {key!r}.") from None


def _checked_section(
    data: Mapping[str, Any], target: Any, *, label: str
) -> dict[str, Any]:
    """Return *data* as a dict, rejecting keys *target* has no field for.

    An unrecognised key used to be dropped in silence, so a ``layer`` written
    for ``layers`` or a ``backend`` block nested one level too deep produced a
    run with defaults and no diagnostic.
    """

    allowed = {item.name for item in fields(target)}
    unknown = sorted(str(key) for key in data if str(key) not in allowed)
    if unknown:
        raise ConfigurationError(
            f"{label} does not recognize these fields: {unknown}. They are rejected "
            f"rather than silently ignored; valid fields: {sorted(allowed)}."
        )
    return {str(key): value for key, value in data.items()}


def _enum_value(enum: Any, value: Any, *, label: str) -> Any:
    try:
        return enum(value)
    except ValueError:
        raise ConfigurationError(
            f"Unknown value for {label}: {value!r}; valid ones: "
            f"{sorted({str(item) for item in enum})}."
        ) from None


def _sequence(value: Any, *, label: str) -> tuple[Any, ...]:
    """Turn a YAML list into a tuple, refusing to iterate a string.

    ``tags: abc`` used to become ``('a', 'b', 'c')`` -- three tags nobody
    wrote -- because ``tuple()`` happily iterates a string.
    """

    if isinstance(value, tuple):
        return value
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, set, frozenset)):
        raise ConfigurationError(
            f"{label} must be a list; {type(value).__name__} was given ({value!r}). "
            "Use a list for a single value as well."
        )
    return tuple(value)


def _sequence_field(data: dict[str, Any], key: str, *, label: str) -> dict[str, Any]:
    if data.get(key) is not None:
        data[key] = _sequence(data[key], label=label)
    return data


def from_primitive_complex(value: Any) -> complex | float | int:
    """Rebuild a scalar that ``to_primitive`` may have rendered as real/imag.

    ``to_primitive`` writes a complex number as ``{"real": .., "imag": ..}``
    so it survives JSON and YAML.  Nothing read it back, so a Hermitian
    observable with any imaginary part -- every non-trivial one -- could be
    written to YAML and never loaded again: ``from_yaml`` failed with
    "matrix must contain finite numbers", pointing at the wrong problem.
    """

    if isinstance(value, Mapping):
        keys = set(value)
        if keys <= {"real", "imag"} and "real" in keys:
            real = value.get("real", 0.0)
            imag = value.get("imag", 0.0)
            if not is_finite_number(real) or not is_finite_number(imag):
                raise ConfigurationError(
                    "The real/imag fields of a complex number must be finite."
                )
            return complex(float(real), float(imag))
        raise ConfigurationError(
            f"A complex number is only read in the form {{'real': .., 'imag': ..}}; "
            f"keys received: {sorted(keys)}"
        )
    return cast("complex | float | int", value)


def _complex_matrix(value: Any) -> tuple[tuple[complex, ...], ...] | None:
    """Normalize a serialized observable matrix back into complex rows."""

    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise ConfigurationError("observable_matrix must be a sequence of rows.")
    rows: list[tuple[complex, ...]] = []
    for row in value:
        if not isinstance(row, (list, tuple)):
            raise ConfigurationError("observable_matrix rows must be sequences.")
        rows.append(tuple(complex(from_primitive_complex(item)) for item in row))
    return tuple(rows)


__all__ = [
    "FrozenDict",
    "freeze_value",
    "to_primitive",
    "Engine",
    "ModelMode",
    "ComponentStatus",
    "GradientMethod",
    "EncodingSpec",
    "AnsatzSpec",
    "MeasurementSpec",
    "NoiseSpec",
    "CircuitSpec",
    "BackendSpec",
    "ModelSpec",
    "TrainingSpec",
    "ExperimentSpec",
    "BACKEND_OPTION_KEYS",
    "SPEC_OPTION_KEYS",
    "register_option_keys",
    "QISKIT_SIMULATION_METHODS",
    "PENNYLANE_GPU_DEVICES",
    "ENTANGLEMENT_TOPOLOGIES",
    "IR_OBSERVABLES",
    "normalize_entanglement",
    "normalize_observable",
    "is_finite_number",
    "is_complex_scalar",
    "is_finite_real_number",
]
