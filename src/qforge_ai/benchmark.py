"""Resource, latency and cross-framework consistency benchmarks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, replace
from statistics import NormalDist
from typing import Any

import numpy as np

from .backends import BackendAdapter, BackendRouter, PennyLaneAdapter, QiskitAdapter
from .errors import ConfigurationError
from .ir import CircuitIR, Measurement, Observable
from .specs import BackendSpec, Engine


@dataclass(frozen=True, slots=True)
class ResourceReport:
    num_qubits: int
    operation_count: int
    depth: int
    two_qubit_gate_count: int
    multi_qubit_operation_count: int
    parameter_count: int
    gate_counts: dict[str, int]
    estimated_statevector_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ConsistencyReport:
    comparable: bool
    fidelity: float | None
    maximum_absolute_error: float | None
    qiskit_runtime_seconds: float
    pennylane_runtime_seconds: float
    passed: bool
    tolerance: float
    notes: tuple[str, ...] = ()
    total_variation_distance: float | None = None
    hellinger_distance: float | None = None
    confidence_level: float | None = None
    statistical_tolerance: float | None = None
    qiskit_shots: int | None = None
    pennylane_shots: int | None = None
    total_variation_confidence_interval: tuple[float, float] | None = None


def resource_report(circuit: CircuitIR) -> ResourceReport:
    return ResourceReport(
        num_qubits=circuit.num_qubits,
        operation_count=len(circuit.operations),
        depth=circuit.depth(),
        two_qubit_gate_count=circuit.two_qubit_gate_count(),
        multi_qubit_operation_count=circuit.multi_qubit_operation_count(),
        parameter_count=circuit.parameter_count(),
        gate_counts=circuit.gate_counts(),
        estimated_statevector_bytes=16 * (2**circuit.num_qubits),
    )


def compare_engines(
    circuit: CircuitIR,
    *,
    inputs: Iterable[float],
    weights: Iterable[float],
    tolerance: float = 1e-7,
    confidence_level: float = 0.95,
    measurement: str = "statevector",
    qiskit_adapter: BackendAdapter | None = None,
    pennylane_adapter: BackendAdapter | None = None,
    qiskit_backend: BackendSpec | None = None,
    pennylane_backend: BackendSpec | None = None,
) -> ConsistencyReport:
    if tolerance < 0:
        raise ConfigurationError("tolerance cannot be negative.")
    if not 0.0 < confidence_level < 1.0:
        raise ConfigurationError("confidence_level must be between 0 and 1.")
    comparison_ir = _comparison_ir(circuit, measurement)
    qiskit_engine = _adapter(
        qiskit_adapter,
        qiskit_backend,
        expected=Engine.QISKIT,
        fallback=QiskitAdapter(),
    )
    pennylane_engine = _adapter(
        pennylane_adapter,
        pennylane_backend,
        expected=Engine.PENNYLANE,
        fallback=PennyLaneAdapter(),
    )
    qiskit = qiskit_engine.run(comparison_ir, inputs=inputs, weights=weights)
    pennylane = pennylane_engine.run(comparison_ir, inputs=inputs, weights=weights)
    if measurement in {"counts", "sample"}:
        left, right = _empirical_probabilities(qiskit.values, pennylane.values)
    else:
        dtype = float if measurement == "probability" else complex
        left = np.asarray(qiskit.values, dtype=dtype)
        right = np.asarray(pennylane.values, dtype=dtype)
    if left.shape != right.shape:
        return ConsistencyReport(
            comparable=False,
            fidelity=None,
            maximum_absolute_error=None,
            qiskit_runtime_seconds=qiskit.runtime_seconds,
            pennylane_runtime_seconds=pennylane.runtime_seconds,
            passed=False,
            tolerance=tolerance,
            notes=(f"Shape difference: {left.shape} != {right.shape}",),
            qiskit_shots=qiskit.shots,
            pennylane_shots=pennylane.shots,
        )
    notes: tuple[str, ...]
    fidelity: float | None
    if measurement == "statevector":
        flat_left = left.reshape(-1)
        flat_right = right.reshape(-1)
        overlap = np.vdot(flat_left, flat_right)
        fidelity = float(abs(overlap) ** 2)
        phase = np.angle(overlap) if abs(overlap) else 0.0
        aligned = flat_right * np.exp(-1j * phase)
        max_error = float(np.max(np.abs(flat_left - aligned)))
        notes = ("Global phase was normalized; wire-major order was applied in every adapter.",)
    else:
        fidelity = None
        max_error = float(np.max(np.abs(left - right))) if left.size else 0.0
        notes = (f"{measurement} results were compared under the wire-major convention.",)
    distribution = measurement in {"probability", "counts", "sample"}
    total_variation = None
    hellinger = None
    statistical_tolerance = None
    tvd_confidence_interval = None
    passed = max_error <= tolerance
    report_confidence = None
    if distribution:
        left_probability = np.asarray(left, dtype=float).reshape(-1)
        right_probability = np.asarray(right, dtype=float).reshape(-1)
        total_variation = float(0.5 * np.sum(np.abs(left_probability - right_probability)))
        hellinger = float(
            np.sqrt(
                0.5
                * np.sum(
                    (np.sqrt(np.clip(left_probability, 0.0, None))
                    - np.sqrt(np.clip(right_probability, 0.0, None)))
                    ** 2
                )
            )
        )
        if qiskit.shots is not None or pennylane.shots is not None:
            report_confidence = confidence_level
            z_score = NormalDist().inv_cdf((1.0 + confidence_level) / 2.0)
            variance = np.zeros_like(left_probability)
            if qiskit.shots is not None:
                variance += left_probability * (1.0 - left_probability) / qiskit.shots
            if pennylane.shots is not None:
                variance += right_probability * (1.0 - right_probability) / pennylane.shots
            sampling_margin = min(
                1.0,
                float(0.5 * z_score * np.sum(np.sqrt(variance))),
            )
            statistical_tolerance = max(tolerance, sampling_margin)
            tvd_confidence_interval = (
                max(0.0, total_variation - sampling_margin),
                min(1.0, total_variation + sampling_margin),
            )
            passed = total_variation <= statistical_tolerance
            notes += (
                f"The finite-shot distribution was evaluated at a {confidence_level * 100:.1f}% "
                "confidence level with TVD/Hellinger.",
            )
    return ConsistencyReport(
        comparable=True,
        fidelity=fidelity,
        maximum_absolute_error=max_error,
        qiskit_runtime_seconds=qiskit.runtime_seconds,
        pennylane_runtime_seconds=pennylane.runtime_seconds,
        passed=passed,
        tolerance=tolerance,
        notes=notes,
        total_variation_distance=total_variation,
        hellinger_distance=hellinger,
        confidence_level=report_confidence,
        statistical_tolerance=statistical_tolerance,
        qiskit_shots=qiskit.shots,
        pennylane_shots=pennylane.shots,
        total_variation_confidence_interval=tvd_confidence_interval,
    )


def _comparison_ir(circuit: CircuitIR, measurement: str) -> CircuitIR:
    supported = {
        "statevector",
        "density_matrix",
        "probability",
        "expectation",
        "variance",
        "counts",
        "sample",
    }
    if measurement not in supported:
        raise ConfigurationError(
            f"Comparison measurement not supported: {measurement}; {sorted(supported)}"
        )
    result = circuit.copy(name=f"{circuit.name}_{measurement}_consistency")
    if measurement == "statevector":
        result.measurements.clear()
        return result
    if result.measurements:
        original = result.measurements[0]
        observables = original.observables
        if measurement in {"expectation", "variance"} and not observables:
            observables = tuple(Observable("Z", (wire,)) for wire in original.wires)
        result.measurements = [
            replace(original, kind=measurement, observables=observables)
        ]
    else:
        wires = result.data_wires
        observables = (
            tuple(Observable("Z", (wire,)) for wire in wires)
            if measurement in {"expectation", "variance"}
            else ()
        )
        result.measurements = [Measurement(measurement, wires, observables)]
    return result


def _adapter(
    explicit: BackendAdapter | None,
    spec: BackendSpec | None,
    *,
    expected: Engine,
    fallback: BackendAdapter,
) -> BackendAdapter:
    if explicit is not None and spec is not None:
        raise ConfigurationError("adapter and BackendSpec cannot be given together for the same engine.")
    adapter = explicit or (BackendRouter().create(spec) if spec is not None else fallback)
    if adapter.engine != str(expected):
        raise ConfigurationError(f"A {adapter.engine} adapter was given for the {expected} comparison slot.")
    return adapter


def _empirical_probabilities(left: Any, right: Any) -> tuple[np.ndarray, np.ndarray]:
    """Bin two sampled results onto the same wire-major bitstring support.

    Neither side is re-ordered here.  Both adapters return counts and samples
    in the framework's wire-major bit order, so a benchmark that flipped the
    Qiskit keys would now *introduce* the disagreement it used to hide -- and
    hiding it was the reason the raw ``counts`` dictionary stayed
    Qiskit-endian for so long.
    """

    left_counts = _as_counts(left)
    right_counts = _as_counts(right)
    keys = sorted(set(left_counts) | set(right_counts))

    def normalize(counts: dict[str, int]) -> np.ndarray:
        total = sum(counts.values())
        return np.asarray([counts.get(key, 0) / total for key in keys], dtype=float)

    return normalize(left_counts), normalize(right_counts)


def _as_counts(values: Any) -> dict[str, int]:
    if isinstance(values, Mapping):
        return {str(key).replace(" ", ""): int(value) for key, value in values.items()}
    samples = np.asarray(values, dtype=int)
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)
    result: dict[str, int] = {}
    for row in samples:
        key = "".join(map(str, row.tolist()))
        result[key] = result.get(key, 0) + 1
    return result


__all__ = ["ResourceReport", "ConsistencyReport", "resource_report", "compare_engines"]
