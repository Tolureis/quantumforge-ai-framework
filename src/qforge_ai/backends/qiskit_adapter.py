"""Qiskit 2.x / Aer adapter."""

from __future__ import annotations

from collections.abc import Iterable
from math import sqrt
from time import perf_counter
from typing import Any

import numpy as np

from ..errors import CapabilityError, ConfigurationError, ExecutionError, TranslationError
from ..ir import CircuitIR, Operation, ParameterRef, resolve_parameter
from ..noise import (
    NOISELESS_OPERATIONS,
    build_qiskit_noise_model,
    canonical_noise_name,
    is_ideal,
    is_terminal_only,
    qiskit_channel_instruction,
    qiskit_readout_instruction,
)
from ..runtime import qiskit_gpu_available
from ..specs import BackendSpec, Engine, NoiseSpec
from ._optional import require
from .base import BackendAdapter, ExecutionResult
from .gates import QISKIT_SINGLE_QUBIT_GATES, QISKIT_TWO_QUBIT_GATES


class QiskitAdapter(BackendAdapter):
    engine = "qiskit"

    def __init__(
        self,
        *,
        method: str = "automatic",
        backend_name: str = "aer",
        compute_device: str = "CPU",
        device: str | None = None,
        shots: int | None = None,
        seed: int = 42,
        noise: NoiseSpec | None = None,
        backend_options: dict[str, Any] | None = None,
        run_options: dict[str, Any] | None = None,
        primitive_options: dict[str, Any] | None = None,
    ) -> None:
        # ``device`` is accepted for 1.0 compatibility.  New code uses the
        # unambiguous backend_name/compute_device pair.
        if device is not None:
            if device.upper() in {"CPU", "GPU"}:
                compute_device = device
            else:
                backend_name = device
        if backend_name not in {"aer", "aer_simulator", "statevector"}:
            raise ConfigurationError(f"Unsupported Qiskit backend: {backend_name}")
        if compute_device.upper() not in {"CPU", "GPU"}:
            raise ConfigurationError("Qiskit compute_device must be CPU or GPU.")
        if shots is not None and shots < 1:
            raise ConfigurationError("Qiskit shots must be None or a positive integer.")
        self.method = method
        self.backend_name = backend_name
        self.compute_device = compute_device.upper()
        self.shots = shots
        self.seed = seed
        self.noise = noise or NoiseSpec()
        self.backend_options = backend_options or {}
        self.run_options = run_options or {}
        self.primitive_options = primitive_options or {}
        if self.compute_device == "GPU" and not qiskit_gpu_available():
            raise CapabilityError(
                "The installed Qiskit Aer build offers no GPU device on this machine. "
                "Verify qiskit-aer-gpu, a compatible CUDA and the NVIDIA driver."
            )

    @classmethod
    def from_spec(cls, spec: BackendSpec) -> QiskitAdapter:
        if spec.engine != Engine.QISKIT:
            raise ConfigurationError("QiskitAdapter.from_spec requires a Qiskit BackendSpec.")
        return cls(
            method=spec.method,
            backend_name=spec.device,
            compute_device=spec.compute_device,
            shots=spec.shots,
            seed=spec.seed,
            noise=spec.noise,
            backend_options=dict(spec.options.get("backend_options", {})),
            run_options=dict(spec.options.get("run_options", {})),
            primitive_options=dict(spec.options.get("primitive_options", {})),
        )

    def compile(
        self,
        circuit: CircuitIR,
        *,
        inputs: Iterable[float] | None = None,
        weights: Iterable[float] | None = None,
        add_measurements: bool = False,
        measurement_wires: Iterable[int] | None = None,
        transpile_backend: Any | None = None,
        optimization_level: int = 1,
        include_noise: bool = True,
    ) -> Any:
        circuit.validate()
        qiskit = require("qiskit", package="qiskit", extra="qiskit", feature="Qiskit adapter")
        circuit_mod = require(
            "qiskit.circuit", package="qiskit", extra="qiskit", feature="Qiskit adapter"
        )
        groups = circuit.parameter_groups()
        vectors = {
            group: circuit_mod.ParameterVector(group, length) for group, length in groups.items()
        }
        if add_measurements and measurement_wires is not None:
            terminal_wires = tuple(measurement_wires)
            circuit._validate_wires(terminal_wires)
        elif add_measurements and circuit.measurements:
            terminal_wires = tuple(circuit.measurements[0].wires)
        else:
            terminal_wires = tuple(range(circuit.num_qubits)) if add_measurements else ()
        terminal_offset = circuit.num_clbits
        num_clbits = max(circuit.num_clbits, terminal_offset + len(terminal_wires))
        qc = qiskit.QuantumCircuit(circuit.num_qubits, num_clbits)
        qc.name = circuit.name
        concrete_inputs = None if inputs is None else list(inputs)
        concrete_weights = None if weights is None else list(weights)

        def value(param: Any) -> Any:
            if not isinstance(param, ParameterRef):
                return param
            concrete = concrete_inputs if param.group == "input" else concrete_weights
            if concrete is not None:
                return resolve_parameter(
                    param,
                    inputs=concrete_inputs or (),
                    weights=concrete_weights or (),
                )
            return vectors[param.group][param.index] * param.scale + param.offset

        # Gate noise is inserted into the circuit rather than configured on the
        # backend, so it lands on exactly the operations the IR names -- the
        # same positions the PennyLane adapter uses.  ``include_noise=False``
        # is for callers that need an invertible circuit (the fidelity kernel's
        # compute-uncompute step) and apply the channel at backend level.
        noisy_gates = include_noise and not is_ideal(self.noise) and not is_terminal_only(
            self.noise
        )
        for operation in circuit.operations:
            if operation.name == "MEASURE":
                qc.measure(operation.wires[0], int(operation.metadata["clbit"]))
            elif operation.condition is not None:
                clbit, expected = operation.condition
                with qc.if_test((qc.clbits[clbit], expected)):
                    self._apply(qc, operation, value)
                    if noisy_gates:
                        self._apply_channel(qc, operation)
            else:
                self._apply(qc, operation, value)
                if noisy_gates:
                    self._apply_channel(qc, operation)
        qc.global_phase = circuit.global_phase
        if add_measurements:
            if include_noise and is_terminal_only(self.noise):
                readout = qiskit_readout_instruction(self.noise)
                if readout is not None:
                    for wire in terminal_wires:
                        qc.append(readout, [wire])
            qc.measure(terminal_wires, range(terminal_offset, terminal_offset + len(terminal_wires)))
        if transpile_backend is not None:
            qc = qiskit.transpile(
                qc,
                transpile_backend,
                optimization_level=optimization_level,
                seed_transpiler=self.seed,
            )
        return qc

    def _apply_channel(self, qc: Any, op: Operation) -> None:
        """Append the canonical gate channel for *op* onto its own wires."""

        if op.name in NOISELESS_OPERATIONS:
            return
        if len(op.wires) > 2:
            raise CapabilityError(
                f"{op.name} is defined on {len(op.wires)} wires; the core noise layer "
                "supports only 1 or 2 wires. This operation is rejected rather than "
                "being run without noise silently."
            )
        instruction = qiskit_channel_instruction(self.noise, len(op.wires))
        if instruction is None:
            return
        # Kraus operators are wire-major; Qiskit qargs are little-endian.
        qc.append(instruction, list(reversed(op.wires)))

    def _apply(self, qc: Any, op: Operation, value: Any) -> None:
        params = [value(param) for param in op.params]
        wires = list(op.wires)
        single = QISKIT_SINGLE_QUBIT_GATES
        double = QISKIT_TWO_QUBIT_GATES
        if op.name in single:
            getattr(qc, single[op.name])(*params, wires[0])
        elif op.name in double:
            getattr(qc, double[op.name])(*params, *wires)
        elif op.name == "BASIS":
            if isinstance(params[0], (int, float, np.number)):
                bit = float(params[0])
                if bit not in {0.0, 1.0}:
                    raise ConfigurationError(
                        f"BASIS encoding accepts only concrete 0/1 bits; got {bit}."
                    )
                if bit == 1.0:
                    qc.x(wires[0])
            else:
                raise TranslationError("Qiskit BASIS encoding requires a concrete input at compile time.")
        elif op.name == "STATE_PREP":
            if any(hasattr(p, "parameters") and p.parameters for p in params):
                raise TranslationError("Qiskit StatePreparation requires a concrete and normalized input.")
            vector = np.asarray(params, dtype=complex)
            target = 2 ** len(wires)
            vector = np.pad(vector, (0, target - len(vector)))
            norm = np.linalg.norm(vector)
            if norm == 0:
                raise ConfigurationError("Amplitude encoding does not accept a zero vector.")
            normalize = bool(op.metadata.get("normalize", False))
            if normalize:
                vector = vector / norm
            elif not np.isclose(norm, 1.0, rtol=1e-7, atol=1e-10):
                raise ConfigurationError(
                    "With normalize=False the amplitude encoding input must have an L2 norm of 1; "
                    f"got norm={norm:.12g}."
                )
            # The IR amplitude vector is wire-major: index bit ``j`` belongs to
            # ``wires[j]``, so ``wires[0]`` is the most significant bit.
            # ``QuantumCircuit.initialize`` reads its qubit list little-endian
            # (``qubits[0]`` is least significant), so the list is reversed.
            # Without this the two engines prepared *orthogonal* states from the
            # same IR: ``[0,1,0,0]`` gave ``|10>`` on Qiskit and ``|01>`` on
            # PennyLane, a fidelity of exactly 0.
            qc.initialize(vector, list(reversed(wires)))
        elif op.name == "RZ_PRODUCT":
            qc.rz(float(op.metadata.get("coefficient", 1.0)) * params[0] * params[1], wires[0])
        elif op.name == "RZZ_PRODUCT":
            qc.rzz(params[0] * params[1], *wires)
        elif op.name == "RESET":
            qc.reset(wires[0])
        elif op.name == "BARRIER":
            qc.barrier(*wires)
        else:
            raise TranslationError(f"Unsupported operation in the Qiskit translation: {op.name}")

    def run(
        self,
        circuit: CircuitIR,
        *,
        inputs: Iterable[float] = (),
        weights: Iterable[float] = (),
        shots: int | None = None,
        **kwargs: Any,
    ) -> ExecutionResult:
        started = perf_counter()
        if shots is not None and shots < 1:
            raise ConfigurationError("Qiskit shots must be None or a positive integer.")
        effective_shots = shots if shots is not None else self.shots
        measurement = circuit.measurements[0] if circuit.measurements else None
        kind = measurement.kind if measurement else "statevector"
        if is_terminal_only(self.noise) and kind not in {
            "counts",
            "sample",
        }:
            raise CapabilityError(
                "Qiskit readout error is supported only with a counts or sample measurement."
            )
        if effective_shots is None and measurement is not None:
            effective_shots = measurement.shots
        noisy = not is_ideal(self.noise)
        finite_sampled_kind = kind in {"probability", "expectation", "variance"}
        terminal_sampled_kind = kind in {"counts", "sample"}
        if kind not in {"expectation", "variance"} and self.primitive_options:
            raise ConfigurationError(
                "primitive_options can be used only on the primitive-based expectation/variance "
                "paths; for direct simulator execution use backend_options "
                "and run_options."
            )
        actual_shots = (
            (effective_shots or 1024)
            if terminal_sampled_kind
            else effective_shots if finite_sampled_kind else None
        )
        qc = self.compile(
            circuit,
            inputs=inputs,
            weights=weights,
            add_measurements=(
                terminal_sampled_kind or (kind == "probability" and actual_shots is not None)
            ),
            measurement_wires=measurement.wires if measurement is not None else None,
        )
        dynamic = any(
            operation.name in {"MEASURE", "RESET"} or operation.condition is not None
            for operation in circuit.operations
        )
        if dynamic and kind not in {"counts", "sample"}:
            raise ExecutionError(
                "Dynamic circuit execution requires a terminal counts or sample measurement."
            )
        try:
            use_aer = (
                self.backend_name in {"aer", "aer_simulator"}
                or noisy
                or dynamic
                or terminal_sampled_kind
                or actual_shots is not None
                or self.method != "automatic"
                or self.compute_device == "GPU"
                or bool(self.backend_options)
                or bool(self.run_options)
                or bool(self.primitive_options)
            )
            execution_metadata: dict[str, Any] = {}
            if use_aer:
                values, execution_metadata = self._run_aer(
                    qc,
                    circuit,
                    kind=kind,
                    shots=actual_shots,
                    noisy=noisy,
                )
                if execution_metadata.get("actual_shots") is not None:
                    actual_shots = int(execution_metadata["actual_shots"])
            else:
                values = self._run_statevector(qc, circuit, kind)
        except Exception as exc:
            raise ExecutionError(f"Qiskit execution failed: {exc}") from exc
        return ExecutionResult(
            engine=self.engine,
            values=values,
            runtime_seconds=perf_counter() - started,
            shots=actual_shots,
            metadata={
                # ``method`` is what was asked for; ``simulation_method`` is
                # what actually ran.  A noisy circuit is forced onto
                # density_matrix, and reporting only the request made the
                # provenance record describe a run that never happened.
                "method": self.method,
                "simulation_method": execution_metadata.pop(
                    "simulation_method", self.method
                ),
                "backend_name": self.backend_name,
                "compute_device": self.compute_device,
                "noise_channel": canonical_noise_name(self.noise.name),
                "noise_placement": "circuit" if noisy else "none",
                "requested_shots": effective_shots,
                "actual_shots": actual_shots,
                **execution_metadata,
                **kwargs,
            },
        )

    def _run_statevector(self, qc: Any, ir: CircuitIR, kind: str) -> Any:
        quantum_info = require(
            "qiskit.quantum_info",
            package="qiskit",
            extra="qiskit",
            feature="statevector simulation",
        )
        state = quantum_info.Statevector.from_instruction(qc)
        if kind == "statevector":
            return _qiskit_state_to_wire_major(np.asarray(state.data), ir.num_qubits)
        if kind == "density_matrix":
            density = np.outer(state.data, np.conjugate(state.data))
            return _reduced_wire_major_density(
                density,
                num_qubits=ir.num_qubits,
                wires=ir.measurements[0].wires,
            )
        if kind == "probability":
            wires = ir.measurements[0].wires
            probabilities = np.asarray(state.probabilities(qargs=list(wires)))
            return _qiskit_probabilities_to_wire_major(probabilities, len(wires))
        if kind in {"expectation", "variance"}:
            outputs = []
            for observable in ir.measurements[0].observables:
                operator, qargs = _qiskit_observable(quantum_info, observable, ir.num_qubits)
                mean = state.expectation_value(operator, qargs=qargs)
                if kind == "expectation":
                    outputs.append(float(np.real(mean)))
                else:
                    squared = state.expectation_value(operator @ operator, qargs=qargs)
                    outputs.append(float(np.real(squared - mean * mean)))
            return np.asarray(outputs)
        return _qiskit_state_to_wire_major(np.asarray(state.data), ir.num_qubits)

    def _simulation_method(self, *, sampled: bool, noisy: bool) -> str:
        """Return the Aer method this run will actually use.

        A noisy non-sampled circuit cannot be carried by a pure state, so it is
        forced onto ``density_matrix`` regardless of what was requested.  The
        result metadata reports this value rather than the request.
        """

        return "density_matrix" if noisy and not sampled else self.method

    def analytic_method(self) -> str:
        """Return the Aer method an *analytic* (precision-0) primitive must use.

        ``method='automatic'`` lets Aer pick its own strategy for a circuit
        carrying Kraus instructions, and its choice is not the exact
        density-matrix evolution the direct execution path performs.  On one
        ``bit_flip(p=0.13)`` circuit the default EstimatorQNN read
        ``[0.2726, 0.3480]`` while ``QiskitAdapter.run`` and PennyLane both read
        ``[0.2838, 0.3264]``: an error of 2.2e-2 on a quantity the two engines
        otherwise agree on to 1e-16.  A noisy analytic expectation value is
        exactly ``Tr[rho O]``, so the primitive is pinned to the same
        ``density_matrix`` method ``_simulation_method`` already forces for
        direct execution.
        """

        return self._simulation_method(sampled=False, noisy=not is_ideal(self.noise))

    def _run_aer(
        self,
        qc: Any,
        ir: CircuitIR,
        *,
        kind: str,
        shots: int | None,
        noisy: bool,
    ) -> tuple[Any, dict[str, Any]]:
        sampled = kind in {"counts", "sample"} or (kind == "probability" and shots is not None)
        method = self._simulation_method(sampled=sampled, noisy=noisy)
        values, metadata = self._run_aer_dispatch(
            qc, ir, kind=kind, shots=shots, noisy=noisy, sampled=sampled, method=method
        )
        return values, {"simulation_method": method, **metadata}

    def _run_aer_dispatch(
        self,
        qc: Any,
        ir: CircuitIR,
        *,
        kind: str,
        shots: int | None,
        noisy: bool,
        sampled: bool,
        method: str,
    ) -> tuple[Any, dict[str, Any]]:
        qiskit = require("qiskit", package="qiskit", extra="qiskit", feature="Qiskit Aer")
        if shots is not None and kind in {"expectation", "variance"}:
            return self._run_aer_estimator(qc, ir, kind=kind, shots=shots, method=method)
        if shots is None and kind in {"expectation", "variance"}:
            return self._run_aer_analytic_estimator(qc, ir, kind=kind, method=method)

        backend = self.aer_backend(method=method)
        if not sampled:
            if kind == "statevector" and noisy:
                raise ExecutionError(
                    "A noisy channel does not produce a pure statevector; use density_matrix."
                )
            if kind == "statevector":
                qc.save_statevector()
            elif kind == "probability" and not noisy:
                qc.save_probabilities(ir.measurements[0].wires)
            else:
                qc.save_density_matrix()
        compiled = qiskit.transpile(qc, backend, seed_transpiler=self.seed)
        run_options = {**self.run_options, "seed_simulator": self.seed}
        if shots is not None:
            run_options["shots"] = shots
        result = backend.run(compiled, **run_options).result()
        if not sampled:
            quantum_info = require(
                "qiskit.quantum_info",
                package="qiskit",
                extra="qiskit",
                feature="noisy density-matrix simulation",
            )
            if kind == "statevector":
                state = quantum_info.Statevector(result.data(0)["statevector"])
                return _qiskit_state_to_wire_major(
                    np.asarray(state.data), ir.num_qubits
                ), {}
            if kind == "probability" and not noisy:
                wires = ir.measurements[0].wires
                probabilities = np.asarray(result.data(0)["probabilities"])
                return _qiskit_probabilities_to_wire_major(
                    probabilities, len(wires)
                ), {}
            density = quantum_info.DensityMatrix(result.data(0)["density_matrix"])
            if kind == "density_matrix":
                return (
                    _reduced_wire_major_density(
                        np.asarray(density.data),
                        num_qubits=ir.num_qubits,
                        wires=ir.measurements[0].wires,
                    ),
                    {},
                )
            if kind == "probability":
                wires = ir.measurements[0].wires
                probabilities = np.asarray(density.probabilities(qargs=list(wires)))
                return _qiskit_probabilities_to_wire_major(probabilities, len(wires)), {}
            if kind in {"expectation", "variance"}:
                outputs = []
                for observable in ir.measurements[0].observables:
                    operator, qargs = _qiskit_observable(
                        quantum_info, observable, ir.num_qubits
                    )
                    mean = density.expectation_value(operator, qargs=qargs)
                    if kind == "expectation":
                        outputs.append(float(np.real(mean)))
                    else:
                        squared = density.expectation_value(operator @ operator, qargs=qargs)
                        outputs.append(float(np.real(squared - mean * mean)))
                return np.asarray(outputs), {}
            return np.asarray(density.data), {}
        terminal_width = len(ir.measurements[0].wires)
        # Every sampled result leaves this method in the framework's wire-major
        # bit order: key character ``j`` belongs to ``measurement.wires[j]``.
        # The raw Aer dictionary is little-endian and used to be handed to the
        # caller unconverted while ``sample`` and ``probability`` were
        # converted, so one IR produced ``{'10': n}`` here and ``{'01': n}`` on
        # PennyLane -- silently transposing class labels and any bitstring
        # analysis built on top of the counts.
        counts = _terminal_counts(result.get_counts(), terminal_width)
        if kind == "sample":
            samples = np.asarray(
                [
                    list(map(int, bitstring))
                    for bitstring, count in counts.items()
                    for _ in range(count)
                ]
            )
            return samples, {}
        if kind == "probability":
            probabilities = _counts_to_wire_major_probabilities(counts, terminal_width)
            standard_errors = np.sqrt(
                probabilities * (1.0 - probabilities) / float(shots or 1)
            )
            return probabilities, {"standard_errors": standard_errors.tolist()}
        return counts, {}

    def aer_backend(
        self,
        *,
        method: str | None = None,
        include_noise_model: bool = False,
    ) -> Any:
        """Create the configured AerSimulator shared by execution and QNN primitives.

        Gate noise normally travels *in the circuit* (see ``compile``), so the
        backend carries no noise model by default -- attaching one as well
        would apply every channel twice.  ``include_noise_model=True`` is for
        the fidelity-kernel path, whose compute-uncompute step needs an
        invertible feature map and therefore cannot hold Kraus instructions.
        """

        aer = require("qiskit_aer", package="qiskit-aer", extra="qiskit", feature="Qiskit Aer")
        options = {
            **self.backend_options,
            "method": method or self.method,
            "device": self.compute_device,
        }
        if include_noise_model:
            noise_model = build_qiskit_noise_model(self.noise)
            if noise_model is not None:
                options["noise_model"] = noise_model
        return aer.AerSimulator(**options)

    def estimator_primitive(
        self, *, include_noise_model: bool = False
    ) -> tuple[Any, float, str]:
        """Create the EstimatorV2 dictated by this adapter's backend contract."""

        primitives = require(
            "qiskit.primitives",
            package="qiskit",
            extra="qiskit",
            feature="EstimatorQNN",
        )
        if self.shots is not None:
            if self.run_options:
                raise ConfigurationError(
                    "A finite-shot BackendEstimatorV2 does not accept BackendSpec.options.run_options "
                    "values. Move the backend run settings to "
                    "backend_options, or use the analytic Aer EstimatorV2."
                )
            self._validate_primitive_options(
                allowed={"abelian_grouping"},
                path="finite-shot BackendEstimatorV2",
            )
            precision = _precision_for_shots(self.shots)
            estimator = primitives.BackendEstimatorV2(
                backend=self.aer_backend(include_noise_model=include_noise_model),
                options={
                    "default_precision": precision,
                    "seed_simulator": self.seed,
                    "abelian_grouping": bool(
                        self.primitive_options.get("abelian_grouping", True)
                    ),
                },
            )
            return estimator, precision, "backend_estimator_v2_finite_shot"
        # ``is_ideal`` rather than an inline name set: ``NoiseSpec(name='ideal')``
        # is a documented alias for "no noise", and spelling the check by hand
        # here made the alias behave like a real channel.
        noisy = not is_ideal(self.noise)
        uses_aer = (
            self.backend_name in {"aer", "aer_simulator"}
            or noisy
            or self.method != "automatic"
            or self.compute_device == "GPU"
            or bool(self.backend_options)
            or bool(self.run_options)
        )
        if uses_aer:
            self._validate_primitive_options(allowed=set(), path="analytic Aer EstimatorV2")
            aer_primitives = require(
                "qiskit_aer.primitives",
                package="qiskit-aer",
                extra="qiskit",
                feature="Aer EstimatorQNN",
            )
            method = self.analytic_method()
            estimator = aer_primitives.EstimatorV2.from_backend(
                self.aer_backend(
                    method=method, include_noise_model=include_noise_model
                ),
                options={
                    "default_precision": 0.0,
                    "run_options": {**self.run_options, "seed_simulator": self.seed},
                },
            )
            return estimator, 0.0, f"aer_estimator_v2_analytic[{method}]"
        self._validate_primitive_options(allowed=set(), path="StatevectorEstimator")
        return (
            primitives.StatevectorEstimator(default_precision=0.0, seed=self.seed),
            0.0,
            "statevector_estimator",
        )

    def sampler_primitive(
        self, *, include_noise_model: bool = False, shots: int | None = None
    ) -> tuple[Any, str]:
        """Create the SamplerV2 dictated by this adapter's backend contract.

        *shots* must be resolved by the caller.  It used to silently fall back
        to 1024 whenever nothing was configured, so a "no shots" experiment ran
        finite-shot on Qiskit while running analytically on PennyLane.
        """

        shots = self.shots if shots is None else shots
        if shots is None:
            raise ConfigurationError(
                "The Qiskit sampler-based path cannot run analytically and does not fall back "
                "to 1024 shots silently; give an explicit shot count with BackendSpec.shots "
                "or Measurement.shots."
            )
        if shots < 1:
            raise ConfigurationError("Qiskit shots must be a positive integer.")
        noisy = not is_ideal(self.noise)
        uses_aer = (
            self.backend_name in {"aer", "aer_simulator"}
            or noisy
            or self.method != "automatic"
            or self.compute_device == "GPU"
            or bool(self.backend_options)
            or bool(self.run_options)
        )
        if uses_aer:
            self._validate_primitive_options(allowed=set(), path="Aer SamplerV2")
            aer_primitives = require(
                "qiskit_aer.primitives",
                package="qiskit-aer",
                extra="qiskit",
                feature="Aer SamplerQNN",
            )
            sampler = aer_primitives.SamplerV2.from_backend(
                self.aer_backend(include_noise_model=include_noise_model),
                default_shots=shots,
                seed=self.seed,
                options={"run_options": self.run_options},
            )
            return sampler, "aer_sampler_v2"
        self._validate_primitive_options(allowed=set(), path="StatevectorSampler")
        primitives = require(
            "qiskit.primitives",
            package="qiskit",
            extra="qiskit",
            feature="SamplerQNN",
        )
        return (
            primitives.StatevectorSampler(default_shots=shots, seed=self.seed),
            "statevector_sampler",
        )

    def _for_primitive(self, qc: Any, backend: Any) -> Any:
        """Lower *qc* to instructions the target simulator method accepts.

        Aer's ``EstimatorV2`` runs the circuit as given, so a high-level
        instruction such as ``initialize`` reaches a ``density_matrix``
        simulation that cannot execute it.  AerSimulator has no coupling map,
        so this decomposition keeps the trivial qubit layout and the
        full-width observables stay aligned.
        """

        qiskit = require("qiskit", package="qiskit", extra="qiskit", feature="Qiskit Aer")
        return qiskit.transpile(
            qc, backend, optimization_level=0, seed_transpiler=self.seed
        )

    def _validate_primitive_options(self, *, allowed: set[str], path: str) -> None:
        unsupported = sorted(set(self.primitive_options) - allowed)
        if unsupported:
            raise ConfigurationError(
                f"{path} does not support these primitive_options values: {unsupported}. "
                f"Allowed for this path: {sorted(allowed)}"
            )

    def _run_aer_estimator(
        self,
        qc: Any,
        ir: CircuitIR,
        *,
        kind: str,
        shots: int,
        method: str | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if self.run_options:
            raise ConfigurationError(
                "A finite-shot BackendEstimatorV2 does not accept run_options; "
                "use backend_options."
            )
        self._validate_primitive_options(
            allowed={"abelian_grouping"}, path="finite-shot BackendEstimatorV2"
        )
        primitives = require(
            "qiskit.primitives",
            package="qiskit",
            extra="qiskit",
            feature="finite-shot expectation",
        )
        quantum_info = require(
            "qiskit.quantum_info",
            package="qiskit",
            extra="qiskit",
            feature="finite-shot expectation",
        )
        observables = [
            _full_qiskit_observable(quantum_info, observable, ir.num_qubits)
            for observable in ir.measurements[0].observables
        ]
        squared = [observable @ observable for observable in observables]
        requested = observables if kind == "expectation" else [*observables, *squared]
        precision = _precision_for_shots(shots)
        primitive_options = {
            "default_precision": precision,
            "seed_simulator": self.seed,
            "abelian_grouping": bool(
                self.primitive_options.get("abelian_grouping", True)
            ),
        }
        backend = self.aer_backend(method=method)
        estimator = primitives.BackendEstimatorV2(
            backend=backend,
            options=primitive_options,
        )
        pub = estimator.run(
            [(self._for_primitive(qc, backend), requested)], precision=precision
        ).result()[0]
        actual_shots = int(pub.metadata.get("shots", shots))
        values = np.asarray(pub.data.evs, dtype=float).reshape(-1)
        standard_errors = np.asarray(pub.data.stds, dtype=float).reshape(-1)
        if kind == "variance":
            count = len(observables)
            means = values[:count]
            values = values[count:] - means**2
            standard_errors = np.sqrt(
                standard_errors[count:] ** 2
                + (2.0 * np.abs(means) * standard_errors[:count]) ** 2
            )
        return values, {
            "standard_errors": standard_errors.tolist(),
            "estimator_precision": precision,
            "estimator_shots": actual_shots,
            "actual_shots": actual_shots,
        }

    def _run_aer_analytic_estimator(
        self,
        qc: Any,
        ir: CircuitIR,
        *,
        kind: str,
        method: str | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        self._validate_primitive_options(allowed=set(), path="analytic Aer EstimatorV2")
        aer_primitives = require(
            "qiskit_aer.primitives",
            package="qiskit-aer",
            extra="qiskit",
            feature="analytic Aer expectation",
        )
        quantum_info = require(
            "qiskit.quantum_info",
            package="qiskit",
            extra="qiskit",
            feature="analytic Aer expectation",
        )
        observables = [
            _full_qiskit_observable(quantum_info, observable, ir.num_qubits)
            for observable in ir.measurements[0].observables
        ]
        squared = [observable @ observable for observable in observables]
        requested = observables if kind == "expectation" else [*observables, *squared]
        backend = self.aer_backend(method=method)
        estimator = aer_primitives.EstimatorV2.from_backend(
            backend,
            options={
                "default_precision": 0.0,
                "run_options": {**self.run_options, "seed_simulator": self.seed},
            },
        )
        pub = estimator.run([(self._for_primitive(qc, backend), requested)], precision=0.0).result()[0]
        values = np.asarray(pub.data.evs, dtype=float).reshape(-1)
        if kind == "variance":
            count = len(observables)
            values = values[count:] - values[:count] ** 2
        return values, {
            "standard_errors": [0.0] * len(observables),
            "estimator_precision": 0.0,
            "estimator_shots": None,
        }


def _qiskit_state_to_wire_major(state: np.ndarray, num_qubits: int) -> np.ndarray:
    if state.size != 2**num_qubits:
        return state
    return state.reshape((2,) * num_qubits).transpose(tuple(reversed(range(num_qubits)))).reshape(-1)


def _qiskit_probabilities_to_wire_major(values: np.ndarray, num_wires: int) -> np.ndarray:
    if values.size != 2**num_wires:
        return values
    return values.reshape((2,) * num_wires).transpose(tuple(reversed(range(num_wires)))).reshape(-1)


def _reduced_wire_major_density(
    density: np.ndarray,
    *,
    num_qubits: int,
    wires: tuple[int, ...],
) -> np.ndarray:
    """Normalize Qiskit matrix axes and trace to requested wire order."""

    full_dimension = 2**num_qubits
    if density.shape != (full_dimension, full_dimension):
        return density
    tensor = density.reshape((2,) * (2 * num_qubits))
    qiskit_to_wire_major = tuple(reversed(range(num_qubits))) + tuple(
        reversed(range(num_qubits, 2 * num_qubits))
    )
    tensor = tensor.transpose(qiskit_to_wire_major)
    keep = list(wires)
    traced = [wire for wire in range(num_qubits) if wire not in keep]
    permutation = keep + traced + [num_qubits + wire for wire in keep] + [
        num_qubits + wire for wire in traced
    ]
    tensor = tensor.transpose(permutation)
    keep_dimension = 2 ** len(keep)
    traced_dimension = 2 ** len(traced)
    tensor = tensor.reshape(
        keep_dimension,
        traced_dimension,
        keep_dimension,
        traced_dimension,
    )
    return np.asarray(np.trace(tensor, axis1=1, axis2=3))


def _observable_qargs(observable: Any) -> list[int]:
    """Return little-endian ``qargs`` for a wire-major observable matrix.

    An IR observable matrix is indexed wire-major: index bit ``j`` belongs to
    ``wires[j]``, so ``wires[0]`` is the most significant bit.  Qiskit's
    ``qargs`` are little-endian (``qargs[0]`` is least significant), so the
    wire tuple must be reversed.  Passing it straight through made
    ``diag(0,1,2,3)`` read the two qubits in the opposite order, and the same
    circuit measured 1.0 on Qiskit against 2.0 on PennyLane.
    """

    return list(reversed(observable.wires))


def _qiskit_observable(quantum_info: Any, observable: Any, num_qubits: int) -> tuple[Any, Any]:
    if observable.matrix is not None:
        operator = quantum_info.Operator(np.asarray(observable.matrix, dtype=complex))
        if observable.coefficient != 1.0:
            operator = observable.coefficient * operator
        return operator, _observable_qargs(observable)
    if observable.name == "H":
        # ``H`` is a valid observable but not a Pauli label, so it cannot go
        # through ``SparsePauliOp.from_sparse_list``.  Without this branch the
        # catalog's ``hadamard_observable`` failed on the plain statevector
        # path while working on the Aer estimator path.
        operator = quantum_info.Operator(
            np.asarray([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / sqrt(2.0)
        )
        if observable.coefficient != 1.0:
            operator = observable.coefficient * operator
        return operator, _observable_qargs(observable)
    if observable.name not in {"I", "X", "Y", "Z"}:
        raise TranslationError(
            f"Qiskit observable not supported: {observable.name}. "
            "For an arbitrary Hermitian use Observable(matrix=...)."
        )
    pauli = quantum_info.SparsePauliOp.from_sparse_list(
        [(observable.name, list(observable.wires), observable.coefficient)],
        num_qubits=num_qubits,
    )
    return pauli, None


def _full_qiskit_observable(quantum_info: Any, observable: Any, num_qubits: int) -> Any:
    """Embed an IR observable into a full-system SparsePauliOp."""

    if observable.matrix is not None:
        local = quantum_info.Operator(np.asarray(observable.matrix, dtype=complex))
        if observable.coefficient != 1.0:
            local = observable.coefficient * local
    elif observable.name == "H":
        local = quantum_info.Operator(
            np.asarray([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / sqrt(2.0)
        )
        if observable.coefficient != 1.0:
            local = observable.coefficient * local
    elif observable.name in {"I", "X", "Y", "Z"}:
        return quantum_info.SparsePauliOp.from_sparse_list(
            [(observable.name, list(observable.wires), observable.coefficient)],
            num_qubits=num_qubits,
        )
    else:
        raise TranslationError(
            f"Finite-shot Qiskit observable not supported: {observable.name}"
        )
    full = quantum_info.Operator(np.eye(2**num_qubits, dtype=complex)).compose(
        local,
        qargs=_observable_qargs(observable),
    )
    return quantum_info.SparsePauliOp.from_operator(full).simplify()


def _counts_to_wire_major_probabilities(counts: dict[str, int], width: int) -> np.ndarray:
    """Turn *wire-major* counts into the wire-major probability vector.

    Index bit ``j`` of the returned vector belongs to ``wires[j]``, matching
    ``qml.probs(wires=...)`` and ``_qiskit_probabilities_to_wire_major``.
    """

    probabilities = np.zeros(2**width, dtype=float)
    total = sum(int(count) for count in counts.values())
    if total == 0:
        return probabilities
    for bitstring, count in counts.items():
        probabilities[int(bitstring, 2)] += int(count) / total
    return probabilities


def _precision_for_shots(shots: int) -> float:
    """Map exact shot counts to BackendEstimatorV2's ceil(1/precision²) contract."""

    return float(np.nextafter(1.0 / sqrt(shots), np.inf))


def _terminal_counts(counts: dict[str, int], width: int) -> dict[str, int]:
    """Return terminal counts in wire-major order.

    Two conversions happen here, and both are part of the framework's public
    result contract:

    * mid-circuit classical bits are dropped.  Terminal measurements occupy the
      highest classical indices, which Qiskit prints leftmost, so the first
      ``width`` characters are exactly the terminal register.
    * the surviving register is reversed.  Qiskit prints a register
      little-endian (leftmost character = highest clbit = last measured wire)
      while the framework -- and PennyLane's ``qml.counts`` -- put ``wires[0]``
      first.  Without this the ``counts`` dictionary was the only result in the
      framework still carrying Qiskit's bit order.
    """

    normalized: dict[str, int] = {}
    for bitstring, count in counts.items():
        terminal = bitstring.replace(" ", "")[:width][::-1]
        normalized[terminal] = normalized.get(terminal, 0) + int(count)
    return normalized


__all__ = ["QiskitAdapter"]
