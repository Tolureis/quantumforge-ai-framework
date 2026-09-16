"""Verify an ideal circuit on Qiskit and PennyLane."""

import numpy as np

from qforge_ai import AnsatzSpec, CircuitSpec, EncodingSpec, MeasurementSpec
from qforge_ai.benchmark import compare_engines
from qforge_ai.circuits import CircuitFactory

spec = CircuitSpec(
    num_qubits=3,
    encoding=EncodingSpec("angle", 3),
    ansatz=AnsatzSpec("real_amplitudes", layers=1, entanglement="ring"),
    measurement=MeasurementSpec("statevector"),
)
ir = CircuitFactory().build(spec)
rng = np.random.default_rng(42)
report = compare_engines(
    ir,
    inputs=rng.normal(size=ir.parameter_groups()["input"]),
    weights=rng.normal(size=ir.parameter_groups()["weight"]),
)
print(report)
