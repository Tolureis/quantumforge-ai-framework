"""Build and inspect a backend-independent circuit."""

from qforge_ai import AnsatzSpec, CircuitSpec, EncodingSpec, MeasurementSpec
from qforge_ai.circuits import CircuitFactory

spec = CircuitSpec(
    num_qubits=4,
    encoding=EncodingSpec(name="angle", feature_dimension=4),
    ansatz=AnsatzSpec(name="hardware_efficient", layers=2, entanglement="ring"),
    measurement=MeasurementSpec(name="expectation", wires=(0, 1, 2, 3)),
)

circuit = CircuitFactory().build(spec)
print(circuit.summary())
