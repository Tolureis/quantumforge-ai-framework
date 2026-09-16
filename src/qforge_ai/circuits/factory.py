"""High-level circuit factory."""

from __future__ import annotations

from dataclasses import replace

from ..catalog import catalog
from ..errors import ConfigurationError, RegistryError
from ..ir import CircuitIR, Measurement, Observable
from ..registry import ComponentRegistry, Factory, normalize_key
from ..specs import CircuitSpec
from .ansatz import ANSATZES, AnsatzBuilder
from .encodings import ENCODINGS, data_reuploading_block

SUPPORTED_MEASUREMENTS = {
    "expectation_measurement": "expectation",
    "probability_measurement": "probability",
    "sample_measurement": "sample",
    "count_measurement": "counts",
    "computational_basis_measurement": "counts",
    "variance_measurement": "variance",
    "statevector_measurement": "statevector",
    "density_matrix_measurement": "density_matrix",
    "reduced_density_matrix_measurement": "density_matrix",
}


class CircuitFactory:
    """Build native IR circuits from declarative specifications."""

    def __init__(self, registry: ComponentRegistry | None = None) -> None:
        self.registry = registry or catalog
        self._register_factories()

    def build(self, spec: CircuitSpec) -> CircuitIR:
        ir = CircuitIR(
            num_qubits=spec.num_qubits + spec.ancilla_qubits,
            name=spec.name,
            ancilla_qubits=spec.ancilla_qubits,
            metadata={
                **dict(spec.metadata),
                "spec": spec.to_dict(),
                "data_qubits": spec.num_qubits,
                "ancilla_qubits": spec.ancilla_qubits,
                "warnings": [],
            },
        )
        encoding_key = self.registry.resolve_key(spec.encoding.name)
        ansatz_key = self.registry.resolve_key(spec.ansatz.name)
        encoding_builder = self._builder(encoding_key, category="encoding")
        ansatz_builder = self._builder(ansatz_key, category="ansatz")
        if encoding_key == "data_reuploading_feature_map":
            self._build_data_reuploading(
                ir,
                spec,
                ansatz_key=ansatz_key,
                ansatz_builder=ansatz_builder,
            )
        else:
            start = len(ir.operations)
            encoding_builder(ir, spec.encoding)
            self._mark_role(ir, start, "encoding")
            start = len(ir.operations)
            ansatz_builder(ir, spec.ansatz)
            self._mark_role(ir, start, "ansatz")
        self._add_measurement(ir, spec)
        return ir.validate()

    def build_feature_map(self, spec: CircuitSpec) -> CircuitIR:
        """Build only the data-encoding circuit, without ansatz leakage."""

        ir = CircuitIR(
            num_qubits=spec.num_qubits + spec.ancilla_qubits,
            name=f"{spec.name}_feature_map",
            ancilla_qubits=spec.ancilla_qubits,
            metadata={
                **dict(spec.metadata),
                "spec": spec.to_dict(),
                "data_qubits": spec.num_qubits,
                "ancilla_qubits": spec.ancilla_qubits,
                "feature_map_only": True,
                "warnings": [],
            },
        )
        key = self.registry.resolve_key(spec.encoding.name)
        builder = self._builder(key, category="encoding")
        start = len(ir.operations)
        builder(ir, spec.encoding)
        self._mark_role(ir, start, "encoding")
        return ir

    @staticmethod
    def _build_data_reuploading(
        ir: CircuitIR,
        spec: CircuitSpec,
        *,
        ansatz_key: str,
        ansatz_builder: AnsatzBuilder,
    ) -> None:
        if spec.ansatz.layers != spec.encoding.repeats:
            raise ConfigurationError(
                "For real data re-uploading, encoding.repeats and ansatz.layers must be equal; "
                "each cycle produces one encoding block and one ansatz layer."
            )
        if spec.encoding.repeats < 2:
            raise ConfigurationError(
                "Data re-uploading must load the same feature vector at least twice; "
                "use repeats >= 2. For a single load choose angle encoding."
            )
        if ansatz_key in {"qcnn", "quantum_convolution", "quantum_pooling"}:
            raise ConfigurationError(
                "Data re-uploading cannot be combined with a QCNN/pooling ansatz that narrows "
                "the qubit set."
            )
        for cycle in range(spec.encoding.repeats):
            start = len(ir.operations)
            data_reuploading_block(ir, spec.encoding, cycle)
            CircuitFactory._mark_role(ir, start, "encoding", cycle=cycle)
            layer_spec = replace(
                spec.ansatz,
                layers=1,
                options={**dict(spec.ansatz.options), "layer_offset": cycle},
                skip_final_rotation=(
                    spec.ansatz.skip_final_rotation or cycle < spec.encoding.repeats - 1
                ),
            )
            start = len(ir.operations)
            ansatz_builder(ir, layer_spec)
            CircuitFactory._mark_role(ir, start, "ansatz", cycle=cycle)
        ir.metadata["data_reuploading_cycles"] = spec.encoding.repeats

    @staticmethod
    def _mark_role(ir: CircuitIR, start: int, role: str, *, cycle: int | None = None) -> None:
        for index in range(start, len(ir.operations)):
            operation = ir.operations[index]
            metadata = {**dict(operation.metadata), "role": role}
            if cycle is not None:
                metadata["cycle"] = cycle
            ir.operations[index] = replace(operation, metadata=metadata)

    def _add_measurement(self, ir: CircuitIR, spec: CircuitSpec) -> None:
        key = self.registry.resolve_key(spec.measurement.name)
        try:
            kind = SUPPORTED_MEASUREMENTS[key]
        except KeyError as exc:
            raise RegistryError(
                f"'{key}' is registered in the catalog but has no core measurement implementation. "
                f"Supported measurements: {sorted(SUPPORTED_MEASUREMENTS)}"
            ) from exc
        active = tuple(ir.metadata.get("active_qcnn_wires", ()))
        if spec.measurement.wires is None:
            # A pure statevector belongs to the complete system.  Other default
            # measurements intentionally exclude reserved ancillas.
            wires = (
                tuple(range(ir.num_qubits))
                if kind == "statevector"
                else active or tuple(range(spec.num_qubits))
            )
        else:
            wires = spec.measurement.wires
            discarded = sorted(set(wires) - set(active)) if active else []
            if discarded:
                ir.metadata.setdefault("warnings", []).append(
                    "Wires that are inactive after QCNN pooling are being measured explicitly: "
                    f"{discarded}; active wires: {list(active)}."
                )
        if kind == "statevector" and tuple(wires) != tuple(range(ir.num_qubits)):
            raise ConfigurationError(
                "Subsystem statevector is not supported. For a subsystem use "
                "MeasurementSpec(name='reduced_density_matrix', wires=...) or "
                "density_matrix."
            )
        observables: tuple[Observable, ...] = ()
        if kind in {"expectation", "variance"}:
            if spec.measurement.observable_matrix is not None:
                observables = (
                    Observable(
                        spec.measurement.observable.upper(),
                        tuple(wires),
                        matrix=spec.measurement.observable_matrix,
                    ),
                )
            else:
                observables = tuple(
                    Observable(spec.measurement.observable.upper(), (wire,)) for wire in wires
                )
        ir.measure(
            Measurement(
                kind=kind,
                wires=tuple(wires),
                observables=observables,
                shots=spec.measurement.shots,
                options=spec.measurement.options,
            )
        )
        ir.metadata["measurement_wires"] = list(wires)
        ir.metadata["measurement_component"] = key

    def _builder(self, key: str, *, category: str) -> Factory:
        descriptor = self.registry.get(key)
        accepted_categories = (
            {"ansatz", "tensor_network"} if category == "ansatz" else {category}
        )
        if descriptor.category not in accepted_categories:
            raise RegistryError(
                f"'{key}' is a {descriptor.category} component; a {category} factory was expected."
            )
        try:
            return self.registry.get_factory(key)
        except RegistryError as exc:
            raise RegistryError(
                f"'{key}' exists in the catalog but has no executable IR producer. "
                "Register a plugin factory."
            ) from exc

    def _register_factories(self) -> None:
        builders: dict[str, object] = {**ENCODINGS, **ANSATZES}
        for key, builder in builders.items():
            assert callable(builder)
            self.registry.register_factory(key, builder, ignore_existing=True)
        for key, alias_of in _APPROXIMATION_ALIASES.items():
            self.registry.annotate(
                key,
                approximation=True,
                alias_of=alias_of,
                scientific_contract=(
                    "The core factory implements this name not as a separate published algorithm, "
                    f"but as an approximate/compatibility template based on {alias_of}."
                ),
            )


_APPROXIMATION_ALIASES = {
    "two_local": "hardware_efficient_ansatz",
    "n_local": "hardware_efficient_ansatz",
    "alternating_layer_ansatz": "hardware_efficient_ansatz",
    "strongly_entangling_layers": "xyz_rotation_ansatz",
    "basic_entangler_layers": "real_amplitudes",
    "random_layers": "xyz_rotation_ansatz",
    "euler_rotation_ansatz": "xyz_rotation_ansatz",
    "ising_ansatz": "efficient_su2",
    "pauli_rotation_ansatz": "xyz_rotation_ansatz",
    "quantum_convolution": "qcnn",
    "quantum_pooling": "qcnn",
}

# Register executable core paths and their scientific alias metadata as soon
# as the public package is imported. CircuitFactory remains idempotent, and
# third-party registrations can still replace individual paths explicitly.
_BUILTIN_FACTORY_REGISTRATION = CircuitFactory()


def canonical_component_name(name: str) -> str:
    return catalog.resolve_key(normalize_key(name))


__all__ = ["CircuitFactory", "canonical_component_name"]
