import importlib.util

import pytest

from qforge_ai import AnsatzSpec, BackendSpec, CircuitSpec, EncodingSpec, MeasurementSpec
from qforge_ai.circuits import CircuitFactory
from qforge_ai.errors import CapabilityError
from qforge_ai.models import ModelFactory
from qforge_ai.specs import Engine, ModelMode, ModelSpec


def _ir():
    return CircuitFactory().build(
        CircuitSpec(
            2,
            EncodingSpec("angle", 2),
            AnsatzSpec("real_amplitudes", 1),
            MeasurementSpec("expectation", wires=(0, 1)),
        )
    )


def test_unknown_template_never_falls_through() -> None:
    with pytest.raises(CapabilityError, match="not implemented by the core"):
        ModelFactory().build(
            _ir(),
            ModelSpec(mode=ModelMode.HYBRID, template="not_a_template"),
            BackendSpec(engine=Engine.PENNYLANE, device="default.qubit"),
        )


@pytest.mark.skipif(
    importlib.util.find_spec("pennylane") is None or importlib.util.find_spec("torch") is None,
    reason="pennylane and torch required",
)
@pytest.mark.parametrize(
    "template,mode,options",
    [
        ("classical_quantum_classical", ModelMode.HYBRID, {}),
        ("quantum_sequential", ModelMode.HYBRID, {"classical_blocks": 2}),
        ("quantum_residual", ModelMode.HYBRID, {}),
        ("ids_quantum_bottleneck", ModelMode.HYBRID, {}),
        ("quantum_parallel", ModelMode.HYBRID, {"branches": 2}),
        ("quantum_ensemble", ModelMode.ENSEMBLE, {"branches": 2}),
        # Catalog spellings must route to the same architectures.
        ("classical_quantum_classical_network", ModelMode.HYBRID, {}),
        ("residual_hybrid_qnn", ModelMode.HYBRID, {}),
        ("quantum_bottleneck_network", ModelMode.HYBRID, {}),
        ("parallel_quantum_layer_network", ModelMode.HYBRID, {"branches": 2}),
    ],
)
def test_model_template_routes_are_runnable(
    template: str, mode: ModelMode, options: dict
) -> None:
    import torch

    model = ModelFactory().build(
        _ir(),
        ModelSpec(
            mode=mode,
            template=template,
            input_dimension=3,
            output_dimension=2,
            options=options,
        ),
        BackendSpec(engine=Engine.PENNYLANE, device="default.qubit"),
    )
    output = model(torch.randn(2, 3))
    assert output.shape == (2, 2)


def test_model_factory_passes_complete_backend_contract_to_qnn_factories(monkeypatch) -> None:
    import qforge_ai.models.factory as factory_module

    qiskit_capture = {}
    pennylane_capture = {}

    def estimator_factory(ir, **kwargs):
        qiskit_capture.update(kwargs)
        return "qiskit-qnn"

    def pennylane_factory(ir, **kwargs):
        pennylane_capture.update(kwargs)
        return "pennylane-qnn"

    monkeypatch.setattr(factory_module, "build_estimator_qnn", estimator_factory)
    monkeypatch.setattr(factory_module, "qiskit_torch_layer", lambda qnn: qnn)
    monkeypatch.setattr(factory_module, "pennylane_torch_layer", pennylane_factory)

    qiskit_backend = BackendSpec(
        engine=Engine.QISKIT,
        device="aer",
        shots=77,
        method="density_matrix",
        seed=91,
        options={"backend_options": {"max_parallel_threads": 2}},
    )
    result = ModelFactory().build(
        _ir(),
        ModelSpec(mode=ModelMode.QUANTUM_ONLY, template="quantum_classifier"),
        qiskit_backend,
    )
    assert result == "qiskit-qnn"
    assert qiskit_capture["backend"] == qiskit_backend

    pennylane_backend = BackendSpec(
        engine=Engine.PENNYLANE,
        device="default.qubit",
        shots=55,
        seed=37,
        options={"device_options": {"max_workers": 3}},
    )
    result = ModelFactory().build(
        _ir(),
        ModelSpec(mode=ModelMode.QUANTUM_ONLY, template="quantum_classifier"),
        pennylane_backend,
    )
    assert result == "pennylane-qnn"
    assert pennylane_capture["shots"] == 55
    assert pennylane_capture["seed"] == 37
    assert pennylane_capture["device_options"] == {"max_workers": 3}
