"""Create a differentiable PennyLane-PyTorch hybrid model."""

import torch

from qforge_ai import (
    AnsatzSpec,
    BackendSpec,
    CircuitSpec,
    EncodingSpec,
    Engine,
    MeasurementSpec,
    ModelMode,
    ModelSpec,
)
from qforge_ai.circuits import CircuitFactory
from qforge_ai.models import ModelFactory

ir = CircuitFactory().build(
    CircuitSpec(
        num_qubits=4,
        encoding=EncodingSpec("angle", 4),
        ansatz=AnsatzSpec("hardware_efficient", layers=2, entanglement="ring"),
        measurement=MeasurementSpec("expectation", wires=(0, 1, 2, 3)),
    )
)
model = ModelFactory().build(
    ir,
    ModelSpec(
        mode=ModelMode.HYBRID,
        input_dimension=10,
        hidden_dimensions=(16,),
        output_dimension=3,
    ),
    BackendSpec(engine=Engine.PENNYLANE, device="default.qubit"),
)
print(model(torch.randn(5, 10)).shape)
