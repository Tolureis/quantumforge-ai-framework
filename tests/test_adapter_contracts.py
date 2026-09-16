import importlib.util
from types import SimpleNamespace

import pytest

from qforge_ai import (
    AnsatzSpec,
    BackendSpec,
    CircuitSpec,
    EncodingSpec,
    MeasurementSpec,
)
from qforge_ai.backends import QiskitAdapter
from qforge_ai.circuits import CircuitFactory
from qforge_ai.errors import ConfigurationError
from qforge_ai.models import ModelFactory
from qforge_ai.specs import Engine, GradientMethod, ModelMode, ModelSpec
from qforge_ai.training import TorchTrainer


def _ir(measurement: str = "expectation"):
    return CircuitFactory().build(
        CircuitSpec(
            2,
            EncodingSpec("angle", 2),
            AnsatzSpec("real_amplitudes"),
            MeasurementSpec(measurement, wires=(0, 1)),
        )
    )


def test_qiskit_qnn_auto_routing_follows_measurement_semantics(monkeypatch) -> None:
    import qforge_ai.models.factory as factory_module

    calls: list[str] = []
    monkeypatch.setattr(
        factory_module,
        "build_estimator_qnn",
        lambda ir, **kwargs: calls.append("estimator") or "estimator-qnn",
    )
    monkeypatch.setattr(
        factory_module,
        "build_sampler_qnn",
        lambda ir, **kwargs: calls.append("sampler") or "sampler-qnn",
    )
    monkeypatch.setattr(factory_module, "qiskit_torch_layer", lambda qnn: qnn)
    backend = BackendSpec(engine=Engine.QISKIT, shots=256)
    # A quantum-only model cannot reshape its output, so output_dimension has
    # to match what the measurement actually produces: two observables for the
    # expectation circuit, four basis outcomes for the probability circuit.
    expectation_model = ModelSpec(
        mode=ModelMode.QUANTUM_ONLY, template="quantum_classifier", output_dimension=2
    )
    probability_model = ModelSpec(
        mode=ModelMode.QUANTUM_ONLY, template="quantum_classifier", output_dimension=4
    )

    assert (
        ModelFactory().build(_ir("expectation"), expectation_model, backend)
        == "estimator-qnn"
    )
    assert (
        ModelFactory().build(_ir("probability"), probability_model, backend)
        == "sampler-qnn"
    )
    assert calls == ["estimator", "sampler"]


@pytest.mark.parametrize(
    "measurement,qnn",
    [("probability", "estimator"), ("expectation", "sampler")],
)
def test_qiskit_qnn_rejects_semantic_mismatch(measurement: str, qnn: str) -> None:
    model = ModelSpec(
        mode=ModelMode.QUANTUM_ONLY,
        template="quantum_classifier",
        options={"qnn": qnn},
    )
    with pytest.raises(ConfigurationError, match="requires"):
        ModelFactory().build(
            _ir(measurement), model, BackendSpec(engine=Engine.QISKIT)
        )


def test_sampler_output_shape_drives_classical_head_dimension() -> None:
    model = ModelSpec(
        input_dimension=2,
        options={"interpret": lambda value: value % 2, "sampler_output_shape": 2},
    )
    assert ModelFactory._dimensions(_ir("probability"), model) == (2, 2, 2)
    matrix_output = ModelSpec(
        input_dimension=2,
        options={"interpret": lambda value: (value // 2, value % 2), "sampler_output_shape": (2, 2)},
    )
    assert ModelFactory._dimensions(_ir("probability"), matrix_output) == (2, 2, 4)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
def test_sampler_tuple_output_is_flattened_before_classical_head() -> None:
    import torch

    class TupleQuantumLayer(torch.nn.Module):
        def forward(self, inputs):
            return torch.ones((inputs.shape[0], 2, 2), dtype=inputs.dtype)

    model_spec = ModelSpec(
        input_dimension=2,
        output_dimension=3,
        options={"interpret": lambda value: (value // 2, value % 2), "sampler_output_shape": (2, 2)},
    )
    model = ModelFactory._hybrid_torch(
        _ir("probability"), TupleQuantumLayer(), model_spec, architecture="standard"
    )
    assert model.head.in_features == 4
    assert model(torch.randn(5, 2)).shape == (5, 3)
    assert model(torch.randn(2)).shape == (3,)


def test_qiskit_estimator_qnn_receives_backend_gradient(monkeypatch) -> None:
    import qforge_ai.models.qiskit_qnn as qnn_module

    class FakeParameter:
        def __init__(self, group: str, index: int) -> None:
            self.group = group
            self.index = index

        def __str__(self) -> str:
            return f"{self.group}[{self.index}]"

    class FakeCircuit:
        parameters = (FakeParameter("input", 0), FakeParameter("weight", 0))

    class FakeAdapter:
        def compile(self, ir):
            return FakeCircuit()

        def estimator_primitive(self):
            return "estimator", 0.0, "fake_estimator"

    class ParamShiftEstimatorGradient:
        def __init__(self, *, estimator) -> None:
            self.estimator = estimator

    class FakeEstimatorQNN:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    def fake_require(module: str, **kwargs):
        del kwargs
        if module == "qiskit_machine_learning.neural_networks":
            return SimpleNamespace(EstimatorQNN=FakeEstimatorQNN)
        if module == "qiskit_machine_learning.gradients":
            return SimpleNamespace(
                ParamShiftEstimatorGradient=ParamShiftEstimatorGradient,
            )
        raise AssertionError(module)

    monkeypatch.setattr(qnn_module, "require", fake_require)
    monkeypatch.setattr(
        qnn_module.QiskitAdapter,
        "from_spec",
        classmethod(lambda cls, backend: FakeAdapter()),
    )
    backend = BackendSpec(
        engine=Engine.QISKIT,
        gradient=GradientMethod.PARAMETER_SHIFT,
    )
    qnn = qnn_module.build_estimator_qnn(
        _ir("expectation"), backend=backend, observable="observable"
    )
    assert isinstance(qnn.kwargs["gradient"], ParamShiftEstimatorGradient)
    assert qnn.kwargs["gradient"].estimator == "estimator"
    assert qnn.qforge_gradient_method == "parameter_shift"


def test_spsa_gradient_consumes_backend_options_and_seed(monkeypatch) -> None:
    import qforge_ai.models.qiskit_qnn as qnn_module

    captured = {}

    class SPSAEstimatorGradient:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    gradients = SimpleNamespace(SPSAEstimatorGradient=SPSAEstimatorGradient)
    monkeypatch.setattr(qnn_module, "require", lambda *args, **kwargs: gradients)
    backend = BackendSpec(
        engine=Engine.QISKIT,
        gradient=GradientMethod.SPSA,
        seed=73,
        options={"gradient_options": {"epsilon": 0.02, "batch_size": 3}},
    )
    qnn_module._build_estimator_gradient("primitive", backend)
    assert captured == {
        "estimator": "primitive",
        "epsilon": 0.02,
        "batch_size": 3,
        "seed": 73,
    }


def test_qiskit_primitive_option_contract_rejects_silent_ignores(monkeypatch) -> None:
    import qforge_ai.backends.qiskit_adapter as adapter_module

    fake_primitives = SimpleNamespace()
    monkeypatch.setattr(adapter_module, "require", lambda *args, **kwargs: fake_primitives)

    with pytest.raises(ConfigurationError, match="run_options"):
        QiskitAdapter(shots=32, run_options={"memory": True}).estimator_primitive()
    with pytest.raises(ConfigurationError, match="primitive_options"):
        QiskitAdapter(
            shots=32, primitive_options={"unsupported": True}
        ).estimator_primitive()


def test_empty_validation_loader_is_rejected_without_nan() -> None:
    class NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    class FakeTorch:
        @staticmethod
        def no_grad():
            return NoGrad()

    class Model:
        @staticmethod
        def eval():
            return None

    with pytest.raises(ConfigurationError, match="validation_loader"):
        TorchTrainer._validate(Model(), [], object(), "cpu", FakeTorch())


def test_sampler_custom_interpret_requires_output_shape(monkeypatch) -> None:
    import qforge_ai.models.qiskit_qnn as qnn_module

    class FakeAdapter:
        def compile(self, ir, **kwargs):
            return SimpleNamespace(parameters=())

        def sampler_primitive(self, *, include_noise_model=False, shots=None):
            return "sampler", "fake_sampler"

    monkeypatch.setattr(
        qnn_module.QiskitAdapter,
        "from_spec",
        classmethod(lambda cls, backend: FakeAdapter()),
    )
    monkeypatch.setattr(
        qnn_module,
        "_build_sampler_gradient",
        lambda sampler, backend: "gradient",
    )
    monkeypatch.setattr(
        qnn_module,
        "require",
        lambda *args, **kwargs: SimpleNamespace(SamplerQNN=lambda **values: values),
    )
    with pytest.raises(ConfigurationError, match="output_shape"):
        qnn_module.build_sampler_qnn(
            _ir("probability"),
            backend=BackendSpec(engine=Engine.QISKIT, shots=128),
            interpret=lambda value: value % 2,
        )


def test_sampler_qnn_requires_explicit_shots() -> None:
    """SamplerQNN samples; it must not quietly substitute 1024 shots.

    PennyLane runs the same probability circuit analytically when no shots are
    given, so the silent default made one experiment definition mean two
    different experiments depending on the engine.
    """

    import qforge_ai.models.qiskit_qnn as qnn_module

    with pytest.raises(ConfigurationError, match="shot"):
        qnn_module.build_sampler_qnn(
            _ir("probability"),
            backend=BackendSpec(engine=Engine.QISKIT),
        )


def test_mlflow_tracker_supports_context_and_explicit_lifecycle(monkeypatch) -> None:
    import qforge_ai.tracking as tracking_module

    events = []

    class FakeMLflow:
        @staticmethod
        def set_experiment(name):
            events.append(("experiment", name))

        @staticmethod
        def start_run(**kwargs):
            events.append(("start", kwargs))
            return object()

        @staticmethod
        def log_params(values):
            events.append(("params", values))

        @staticmethod
        def log_metrics(values, **kwargs):
            events.append(("metrics", values, kwargs))

        @staticmethod
        def end_run(**kwargs):
            events.append(("end", kwargs))

    monkeypatch.setattr(tracking_module, "require", lambda *args, **kwargs: FakeMLflow)
    tracker = tracking_module.MLflowTracker("compat")
    tracker.start_run("manual")
    assert tracker.active_run is not None
    tracker.log_params({"nested": {"value": 2}})
    tracker.log_metrics({"loss": 0.1}, step=3)
    tracker.end_run()
    assert tracker.active_run is None
    assert events[1][0] == "start"
    assert ("nested.value", "2") in events[2][1].items()
