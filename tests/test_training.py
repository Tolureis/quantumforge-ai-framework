import importlib.util

import pytest

from qforge_ai import (
    AnsatzSpec,
    BackendSpec,
    CircuitSpec,
    EncodingSpec,
    Engine,
    ExperimentRunner,
    ExperimentSpec,
    MeasurementSpec,
    ModelSpec,
    TrainingSpec,
)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
def test_experiment_runner_train_api_consumes_training_spec() -> None:
    import torch

    spec = ExperimentSpec(
        name="train-api",
        circuit=CircuitSpec(
            2,
            EncodingSpec("angle", 2),
            AnsatzSpec("real_amplitudes", 1),
            MeasurementSpec("expectation", wires=(0, 1)),
        ),
        backend=BackendSpec(engine=Engine.IR),
        model=ModelSpec(input_dimension=2, output_dimension=2),
        training=TrainingSpec(epochs=2, batch_size=2),
    )
    runner = ExperimentRunner()
    prepared = runner.prepare(spec)
    model = torch.nn.Linear(2, 2)
    data = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.randn(6, 2), torch.randint(0, 2, (6,))),
        batch_size=2,
    )
    result = runner.train(prepared, data, model=model)
    assert len(result.history) == 2
    assert result.metadata["experiment"] == "train-api"
