"""PennyLane QNode adapter with PyTorch/JAX-compatible differentiation."""

from __future__ import annotations

from collections.abc import Iterable
from time import perf_counter
from typing import Any

import numpy as np

from ..errors import CapabilityError, ConfigurationError, ExecutionError, TranslationError
from ..ir import CircuitIR, Operation, ParameterRef
from ..noise import (
    NOISELESS_OPERATIONS,
    apply_pennylane_channel,
    apply_pennylane_readout,
    canonical_noise_name,
    is_ideal,
    is_terminal_only,
)
from ..specs import BackendSpec, Engine, NoiseSpec
from ._optional import require
from .base import BackendAdapter, ExecutionResult
from .gradients import normalize_pennylane_gradient_options


class PennyLaneAdapter(BackendAdapter):
    engine = "pennylane"

    def __init__(
        self,
        *,
        device: str = "default.qubit",
        shots: int | None = None,
        seed: int = 42,
        noise: NoiseSpec | None = None,
        device_options: dict[str, Any] | None = None,
        gradient_options: dict[str, Any] | None = None,
    ) -> None:
        if shots is not None and shots < 1:
            raise ConfigurationError("PennyLane shots must be None or a positive integer.")
        self.device_name = device
        self.shots = shots
        self.seed = seed
        self.noise = noise or NoiseSpec()
        self.device_options = device_options or {}
        self.gradient_options = dict(gradient_options or {})

    @classmethod
    def from_spec(cls, spec: BackendSpec) -> PennyLaneAdapter:
        if spec.engine != Engine.PENNYLANE:
            raise ConfigurationError(
                "PennyLaneAdapter.from_spec requires a PennyLane BackendSpec."
            )
        device = spec.device if spec.device != "statevector" else "default.qubit"
        return cls(
            device=device,
            shots=spec.shots,
            seed=spec.seed,
            noise=spec.noise,
            device_options=dict(spec.options.get("device_options", {})),
            gradient_options=dict(spec.options.get("gradient_options", {})),
        )

    def resolved_device_name(self) -> str:
        """Return the device this adapter will actually construct.

        A noisy run on ``default.qubit``/``lightning.qubit`` is silently moved
        to ``default.mixed`` because those devices carry no channels.  The
        provenance record used to report the *requested* device, so a result
        produced on ``default.mixed`` claimed to come from ``default.qubit``.
        """

        if is_ideal(self.noise):
            return self.device_name
        if self.device_name in {"default.qubit", "lightning.qubit"}:
            return "default.mixed"
        return self.device_name

    def compile(
        self,
        circuit: CircuitIR,
        *,
        interface: str = "auto",
        diff_method: str = "best",
        shots: int | None = None,
    ) -> Any:
        circuit.validate()
        qml = require(
            "pennylane", package="pennylane", extra="pennylane", feature="PennyLane adapter"
        )
        effective_shots = shots if shots is not None else self.shots
        kind = circuit.measurements[0].kind if circuit.measurements else "statevector"
        if is_terminal_only(self.noise) and kind not in {
            "counts",
            "sample",
        }:
            raise CapabilityError(
                "PennyLane readout error is supported only with a counts or sample measurement."
            )
        if effective_shots is None and circuit.measurements:
            effective_shots = circuit.measurements[0].shots
        if effective_shots is None and circuit.measurements and circuit.measurements[0].kind in {
            "counts",
            "sample",
        }:
            effective_shots = 1024
        device_name = self.resolved_device_name()
        helper_wires = sum(
            operation.name in {"MEASURE", "RESET"} for operation in circuit.operations
        )
        noisy = not is_ideal(self.noise)
        if noisy:
            if any(
                measurement.kind == "statevector" for measurement in circuit.measurements
            ):
                raise CapabilityError(
                    "A noisy PennyLane run cannot produce a pure statevector; "
                    "choose density_matrix."
                )
            if device_name != "default.mixed":
                raise CapabilityError(
                    f"Noise-channel support for {device_name} cannot be verified by the core; "
                    "use default.mixed or a device plugin."
                )
        gradient_kwargs = normalize_pennylane_gradient_options(
            self.gradient_options, diff_method=diff_method
        )
        device_options = dict(self.device_options)
        # BackendSpec.seed is the canonical experiment seed; a duplicate
        # device_options value must not silently diverge from it. Capability
        # preflight limits core devices to implementations that accept seed.
        device_options["seed"] = self.seed
        dev = qml.device(
            device_name,
            # PennyLane's deferred-measurement transform allocates one helper
            # wire per mid-circuit measurement when a measured wire may feed
            # a condition/reset.  These wires are an engine implementation
            # detail and never appear in the IR/terminal measurement contract.
            wires=circuit.num_qubits + helper_wires,
            **device_options,
        )

        # ``gradient_kwargs=`` -- not ``**gradient_kwargs``.  QNode forwards
        # this mapping to the gradient transform; splatting it put ``h``,
        # ``strategy`` and ``num_directions`` into QNode's own signature, so
        # *every* finite-diff or SPSA model raised
        # ``TypeError: QNode.__init__() got an unexpected keyword argument 'h'``
        # before it could run a single forward pass.
        @qml.qnode(
            dev,
            interface=interface,
            diff_method=diff_method,
            gradient_kwargs=gradient_kwargs or None,
        )
        def qnode(inputs: Any, weights: Any) -> Any:
            def value(param: Any) -> Any:
                if not isinstance(param, ParameterRef):
                    return param
                source = inputs if param.group == "input" else weights
                return source[..., param.index] * param.scale + param.offset

            classical: dict[int, Any] = {}
            for operation in circuit.operations:
                if operation.name == "MEASURE":
                    clbit = int(operation.metadata["clbit"])
                    classical[clbit] = qml.measure(operation.wires[0])
                    continue
                if operation.condition is not None:
                    clbit, expected = operation.condition
                    if clbit not in classical:
                        raise ConfigurationError(
                            f"A conditional gate depends on the unmeasured clbit {clbit}."
                        )

                    def conditional_gate(operation_to_apply: Operation = operation) -> None:
                        self._apply(qml, operation_to_apply, value)
                        self._apply_channel(qml, operation_to_apply, noisy=noisy)

                    qml.cond(classical[clbit] == expected, conditional_gate)()
                else:
                    self._apply(qml, operation, value)
                    self._apply_channel(qml, operation, noisy=noisy)
            measured_wires = (
                circuit.measurements[0].wires
                if circuit.measurements
                else tuple(range(circuit.num_qubits))
            )
            apply_pennylane_readout(qml, self.noise, measured_wires)
            return self._measure(qml, circuit)

        executable = (
            qml.set_shots(qnode, shots=effective_shots)
            if effective_shots is not None
            else qnode
        )
        executable.qforge_ir = circuit
        return executable

    def _apply_channel(self, qml: Any, op: Operation, *, noisy: bool) -> None:
        """Apply the canonical gate channel for *op*, matching the Qiskit side.

        ``BASIS`` and ``STATE_PREP`` define the initial state and stay ideal on
        both engines: PennyLane emitted one channel after ``AmplitudeEmbedding``
        while Qiskit's ``initialize`` decomposed into a state-dependent gate
        count, so ``bit_flip(p=1)`` over ``BASIS(0)`` read ``-1`` here and
        ``+1`` there.
        """

        if not noisy or op.name in NOISELESS_OPERATIONS:
            return
        if len(op.wires) > 2:
            raise CapabilityError(
                f"{op.name} is defined on {len(op.wires)} wires; the core noise layer "
                "supports only 1 or 2 wires. This operation is rejected rather than "
                "being run without noise silently."
            )
        apply_pennylane_channel(qml, self.noise, op.wires)

    def _apply(self, qml: Any, op: Operation, value: Any) -> None:
        params = [value(param) for param in op.params]
        wires = list(op.wires)
        mapping = {
            "X": qml.PauliX,
            "Y": qml.PauliY,
            "Z": qml.PauliZ,
            "H": qml.Hadamard,
            "S": qml.S,
            "T": qml.T,
            "SX": qml.SX,
            "RX": qml.RX,
            "RY": qml.RY,
            "RZ": qml.RZ,
            "CX": qml.CNOT,
            "CNOT": qml.CNOT,
            "CZ": qml.CZ,
            "CY": qml.CY,
            "SWAP": qml.SWAP,
            "CRX": qml.CRX,
            "CRY": qml.CRY,
            "CRZ": qml.CRZ,
            "RXX": qml.IsingXX,
            "RYY": qml.IsingYY,
            "RZZ": qml.IsingZZ,
        }
        if op.name in mapping:
            mapping[op.name](*params, wires=wires if len(wires) > 1 else wires[0])
        elif op.name == "U":
            qml.U3(*params, wires=wires[0])
        elif op.name == "BASIS":
            _validate_basis_value(qml, params[0])
            bit = qml.math.expand_dims(params[0], axis=-1)
            qml.BasisState(bit, wires=[wires[0]])
        elif op.name == "STATE_PREP":
            target_dimension = 2 ** len(wires)
            if len(params) < target_dimension:
                zero = qml.math.zeros_like(params[0])
                params = [*params, *([zero] * (target_dimension - len(params)))]
            amplitudes = qml.math.stack(params, axis=-1)
            normalize = bool(op.metadata.get("normalize", False))
            if normalize:
                # Normalize here rather than delegating to PennyLane.  Its
                # ``normalize=True`` leaves a vector alone when the norm is
                # already within its internal tolerance, while Qiskit always
                # divides -- so a vector with norm 1 + 2.6e-7 produced two
                # different prepared states from one IR.  Doing the division in
                # the adapter makes ``normalize`` mean one thing on both
                # engines, and ``qml.math`` keeps it differentiable.
                norm = qml.math.sqrt(
                    qml.math.sum(qml.math.abs(amplitudes) ** 2, axis=-1)
                )
                amplitudes = amplitudes / qml.math.expand_dims(norm, axis=-1)
            qml.AmplitudeEmbedding(
                amplitudes,
                wires=wires,
                normalize=False,
                # Padding is performed above because PennyLane intentionally
                # forces normalization whenever pad_with is not None.
                pad_with=None,
                validate_norm=True,
            )
        elif op.name == "RZ_PRODUCT":
            qml.RZ(
                float(op.metadata.get("coefficient", 1.0)) * params[0] * params[1], wires=wires[0]
            )
        elif op.name == "RZZ_PRODUCT":
            qml.IsingZZ(params[0] * params[1], wires=wires)
        elif op.name == "RESET":
            qml.measure(wires[0], reset=True)
        elif op.name == "BARRIER":
            qml.Barrier(wires=wires)
        else:
            raise TranslationError(f"Unsupported operation in the PennyLane translation: {op.name}")

    @staticmethod
    def _measure(qml: Any, circuit: CircuitIR) -> Any:
        if not circuit.measurements:
            return qml.state()
        measurement = circuit.measurements[0]
        if measurement.kind == "expectation":
            return tuple(
                qml.expval(_observable(qml, obs)) for obs in measurement.observables
            )
        if measurement.kind == "variance":
            return tuple(qml.var(_observable(qml, obs)) for obs in measurement.observables)
        if measurement.kind == "probability":
            return qml.probs(wires=measurement.wires)
        if measurement.kind == "sample":
            return qml.sample(wires=measurement.wires)
        if measurement.kind == "counts":
            return qml.counts(wires=measurement.wires)
        if measurement.kind == "statevector":
            return qml.state()
        if measurement.kind == "density_matrix":
            return qml.density_matrix(wires=measurement.wires)
        raise TranslationError(f"PennyLane measurement not supported: {measurement.kind}")

    def run(
        self,
        circuit: CircuitIR,
        *,
        inputs: Iterable[float] = (),
        weights: Iterable[float] = (),
        interface: str = "auto",
        diff_method: str = "best",
        shots: int | None = None,
        **kwargs: Any,
    ) -> ExecutionResult:
        started = perf_counter()
        if shots is not None and shots < 1:
            raise ConfigurationError("PennyLane shots must be None or a positive integer.")
        input_array = np.asarray(list(inputs), dtype=float)
        weight_array = np.asarray(list(weights), dtype=float)
        try:
            qnode = self.compile(circuit, interface=interface, diff_method=diff_method, shots=shots)
            values = qnode(input_array, weight_array)
        except Exception as exc:
            raise ExecutionError(f"PennyLane execution failed: {exc}") from exc
        kind = circuit.measurements[0].kind if circuit.measurements else "statevector"
        if kind == "counts":
            values = _normalize_counts(values)
        effective_shots = shots if shots is not None else self.shots
        if effective_shots is None and circuit.measurements:
            effective_shots = circuit.measurements[0].shots
        if (
            effective_shots is None
            and circuit.measurements
            and circuit.measurements[0].kind in {"counts", "sample"}
        ):
            effective_shots = 1024
        return ExecutionResult(
            engine=self.engine,
            values=values,
            runtime_seconds=perf_counter() - started,
            shots=effective_shots,
            metadata={
                # ``device`` is the device that ran; ``requested_device`` is
                # what the spec asked for.  Noise moves default.qubit onto
                # default.mixed, and reporting the request made the provenance
                # record name a device the run never touched.
                "device": self.resolved_device_name(),
                "requested_device": self.device_name,
                "diff_method": diff_method,
                "gradient_options": dict(self.gradient_options),
                "noise_channel": canonical_noise_name(self.noise.name),
                "noise_placement": "circuit" if not is_ideal(self.noise) else "none",
                "seed": self.seed,
                "requested_shots": shots if shots is not None else self.shots,
                "actual_shots": effective_shots,
                "device_options": dict(self.device_options),
                **kwargs,
            },
        )


def _normalize_counts(values: Any) -> Any:
    """Return counts as a plain ``dict[str, int]`` in wire-major bit order.

    ``qml.counts`` already keys on ``wires[0]``-first bitstrings -- the
    framework's contract -- but returns NumPy string keys and NumPy integer
    values.  Both engines now hand back the same builtin types, so a counts
    dictionary can be compared, serialized to JSON or used as a lookup key
    without caring which engine produced it.
    """

    if not isinstance(values, dict):
        return values
    return {str(key): int(value) for key, value in values.items()}


def _observable(qml: Any, observable: Any) -> Any:
    if observable.matrix is not None:
        result = qml.Hermitian(np.asarray(observable.matrix, dtype=complex), wires=observable.wires)
    elif observable.name == "X":
        result = qml.PauliX(observable.wires[0])
    elif observable.name == "Y":
        result = qml.PauliY(observable.wires[0])
    elif observable.name == "Z":
        result = qml.PauliZ(observable.wires[0])
    elif observable.name == "H":
        result = qml.Hadamard(observable.wires[0])
    elif observable.name == "I":
        result = qml.Identity(observable.wires[0])
    else:
        raise TranslationError(f"PennyLane observable not supported: {observable.name}")
    return observable.coefficient * result


def _validate_basis_value(qml: Any, value: Any) -> None:
    if qml.math.is_abstract(value):
        return
    try:
        concrete = np.asarray(qml.math.toarray(value), dtype=float)
    except (TypeError, ValueError):
        return
    if not np.all(np.logical_or(concrete == 0.0, concrete == 1.0)):
        raise ConfigurationError(
            f"BASIS encoding accepts only concrete 0/1 bits; got {concrete}."
        )


__all__ = ["PennyLaneAdapter"]
