import numpy as np
import pytest

from qforge_ai import AnsatzSpec, CircuitSpec, EncodingSpec
from qforge_ai.backends.base import BackendAdapter, ExecutionResult
from qforge_ai.benchmark import compare_engines, resource_report
from qforge_ai.circuits import CircuitFactory


def test_resource_report() -> None:
    ir = CircuitFactory().build(
        CircuitSpec(3, EncodingSpec("angle", 3), AnsatzSpec("real_amplitudes", 1))
    )
    report = resource_report(ir)
    assert report.num_qubits == 3
    assert report.estimated_statevector_bytes == 128
    assert report.parameter_count == 9


def test_state_preparation_is_not_a_two_qubit_gate() -> None:
    ir = CircuitFactory().build(
        CircuitSpec(2, EncodingSpec("amplitude", 4), AnsatzSpec("real_amplitudes", 1))
    )
    assert ir.multi_qubit_operation_count() > ir.two_qubit_gate_count()


class _ProbabilityAdapter(BackendAdapter):
    def __init__(self, engine: str, values: list[float], shots: int) -> None:
        self.engine = engine
        self.values = np.asarray(values)
        self.shots = shots

    def compile(self, circuit, **kwargs):
        return circuit

    def run(self, circuit, *, inputs=(), weights=(), **kwargs) -> ExecutionResult:
        return ExecutionResult(self.engine, self.values, 0.001, self.shots)


def test_finite_shot_benchmark_reports_statistical_distances() -> None:
    ir = CircuitFactory().build(
        CircuitSpec(1, EncodingSpec("angle", 1), AnsatzSpec("real_amplitudes", 1))
    )
    report = compare_engines(
        ir,
        inputs=[0.0],
        weights=[0.0, 0.0],
        measurement="probability",
        qiskit_adapter=_ProbabilityAdapter("qiskit", [0.51, 0.49], 1000),
        pennylane_adapter=_ProbabilityAdapter("pennylane", [0.5, 0.5], 1000),
    )
    assert report.total_variation_distance == pytest.approx(0.01)
    assert report.hellinger_distance is not None
    assert report.hellinger_distance > 0.0
    assert report.confidence_level == 0.95
    assert report.statistical_tolerance is not None
    assert report.total_variation_confidence_interval is not None
    assert report.qiskit_shots == report.pennylane_shots == 1000
    assert report.passed
