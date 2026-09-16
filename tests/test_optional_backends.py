import importlib.util

import numpy as np
import pytest

from qforge_ai import AnsatzSpec, CircuitSpec, EncodingSpec, MeasurementSpec, Observable
from qforge_ai.circuits import CircuitFactory
from qforge_ai.specs import NoiseSpec


def _ir():
    return CircuitFactory().build(
        CircuitSpec(
            2,
            EncodingSpec("angle", 2),
            AnsatzSpec("real_amplitudes", layers=1),
            MeasurementSpec("expectation"),
        )
    )


@pytest.mark.skipif(importlib.util.find_spec("qiskit") is None, reason="qiskit not installed")
def test_qiskit_adapter() -> None:
    from qforge_ai.backends import QiskitAdapter

    ir = _ir()
    result = QiskitAdapter().run(
        ir,
        inputs=np.zeros(ir.parameter_groups()["input"]),
        weights=np.zeros(ir.parameter_groups()["weight"]),
    )
    assert result.as_array().shape == (2,)


@pytest.mark.skipif(importlib.util.find_spec("pennylane") is None, reason="pennylane not installed")
def test_pennylane_adapter() -> None:
    from qforge_ai.backends import PennyLaneAdapter

    ir = _ir()
    result = PennyLaneAdapter().run(
        ir,
        inputs=np.zeros(ir.parameter_groups()["input"]),
        weights=np.zeros(ir.parameter_groups()["weight"]),
    )
    assert result.as_array().shape == (2,)


@pytest.mark.skipif(
    importlib.util.find_spec("qiskit") is None or importlib.util.find_spec("pennylane") is None,
    reason="both engines are required",
)
def test_cross_engine_statevector_normalizes_endianness() -> None:
    from qforge_ai.benchmark import compare_engines

    ir = _ir()
    report = compare_engines(
        ir,
        inputs=np.asarray([0.2, -0.4]),
        weights=np.asarray([0.1, 0.2, 0.3, 0.4]),
    )
    assert report.passed
    assert report.fidelity == pytest.approx(1.0)


@pytest.mark.parametrize("measurement", ["probability", "density_matrix", "expectation"])
@pytest.mark.skipif(
    importlib.util.find_spec("qiskit") is None or importlib.util.find_spec("pennylane") is None,
    reason="both engines are required",
)
def test_cross_engine_measurement_contract(measurement: str) -> None:
    from qforge_ai.benchmark import compare_engines

    ir = _ir()
    report = compare_engines(
        ir,
        inputs=np.asarray([0.27, -0.41]),
        weights=np.asarray([0.13, -0.2, 0.37, 0.04]),
        measurement=measurement,
    )
    assert report.passed, report


@pytest.mark.skipif(
    importlib.util.find_spec("qiskit") is None or importlib.util.find_spec("pennylane") is None,
    reason="both engines are required",
)
@pytest.mark.parametrize("measurement", ["probability", "density_matrix"])
def test_cross_engine_partial_wire_order_contract(measurement: str) -> None:
    from qforge_ai.benchmark import compare_engines

    ir = CircuitFactory().build(
        CircuitSpec(
            3,
            EncodingSpec("angle", 3),
            AnsatzSpec("real_amplitudes", layers=1, entanglement="ring"),
            MeasurementSpec("probability", wires=(2, 0)),
        )
    )
    report = compare_engines(
        ir,
        inputs=np.asarray([0.2, -0.4, 0.7]),
        weights=np.arange(ir.parameter_groups()["weight"]) * 0.07,
        measurement=measurement,
    )
    assert report.passed, report


@pytest.mark.skipif(
    importlib.util.find_spec("qiskit") is None or importlib.util.find_spec("qiskit_aer") is None,
    reason="qiskit aer not installed",
)
def test_qiskit_noisy_expectation() -> None:
    from qforge_ai.backends import QiskitAdapter

    ir = _ir()
    result = QiskitAdapter(noise=NoiseSpec(name="depolarizing_noise", probability=0.01)).run(
        ir,
        inputs=np.zeros(ir.parameter_groups()["input"]),
        weights=np.zeros(ir.parameter_groups()["weight"]),
    )
    assert result.as_array().shape == (2,)
    assert result.shots is None


@pytest.mark.skipif(
    importlib.util.find_spec("qiskit_machine_learning") is None,
    reason="qiskit machine learning not installed",
)
def test_qiskit_qnn_factories() -> None:
    from qforge_ai.models import build_estimator_qnn, build_sampler_qnn
    from qforge_ai.specs import BackendSpec, Engine

    ir = _ir()
    inputs = np.zeros(ir.parameter_groups()["input"])
    weights = np.zeros(ir.parameter_groups()["weight"])
    estimator = build_estimator_qnn(ir)
    sampler_ir = CircuitFactory().build(
        CircuitSpec(
            2,
            EncodingSpec("angle", 2),
            AnsatzSpec("real_amplitudes", layers=1),
            MeasurementSpec("probability"),
        )
    )
    # SamplerQNN samples, so the shot count has to be stated rather than
    # defaulted to 1024 behind the caller's back.
    sampler = build_sampler_qnn(
        sampler_ir, backend=BackendSpec(engine=Engine.QISKIT, shots=1024)
    )
    assert estimator.forward(inputs, weights).shape == (1, 2)
    assert sampler.forward(inputs, weights).shape == (1, 4)


@pytest.mark.skipif(
    importlib.util.find_spec("qiskit_machine_learning") is None,
    reason="qiskit machine learning not installed",
)
def test_qiskit_estimator_qnn_finite_diff_gradient_is_runnable() -> None:
    from qforge_ai.models import build_estimator_qnn
    from qforge_ai.specs import BackendSpec, Engine, GradientMethod

    ir = CircuitFactory().build(
        CircuitSpec(
            1,
            EncodingSpec("angle", 1),
            AnsatzSpec("real_amplitudes"),
            MeasurementSpec("expectation", wires=(0,)),
        )
    )
    qnn = build_estimator_qnn(
        ir,
        backend=BackendSpec(
            engine=Engine.QISKIT,
            gradient=GradientMethod.FINITE_DIFF,
            options={"gradient_options": {"epsilon": 1e-5, "method": "central"}},
        ),
    )
    input_grad, weight_grad = qnn.backward(np.asarray([0.2]), np.asarray([0.1, -0.3]))
    assert input_grad.shape == (1, 1, 1)
    assert weight_grad.shape == (1, 1, 2)
    assert qnn.qforge_gradient_method == "finite_diff"


@pytest.mark.skipif(
    importlib.util.find_spec("qiskit_machine_learning") is None,
    reason="qiskit machine learning not installed",
)
@pytest.mark.parametrize("measurement", ["expectation", "probability"])
def test_qiskit_qnn_spsa_gradient_is_runnable(measurement: str) -> None:
    from qforge_ai.models import build_estimator_qnn, build_sampler_qnn
    from qforge_ai.specs import BackendSpec, Engine, GradientMethod

    ir = CircuitFactory().build(
        CircuitSpec(
            1,
            EncodingSpec("angle", 1),
            AnsatzSpec("real_amplitudes"),
            MeasurementSpec(measurement, wires=(0,)),
        )
    )
    backend = BackendSpec(
        engine=Engine.QISKIT,
        gradient=GradientMethod.SPSA,
        seed=7,
        shots=None if measurement == "expectation" else 256,
        options={"gradient_options": {"epsilon": 1e-3, "batch_size": 2}},
    )
    builder = build_estimator_qnn if measurement == "expectation" else build_sampler_qnn
    qnn = builder(ir, backend=backend)
    input_grad, weight_grad = qnn.backward(np.asarray([0.2]), np.asarray([0.1, -0.3]))
    assert input_grad.shape[-1] == 1
    assert weight_grad.shape[-1] == 2
    assert qnn.qforge_gradient_method == "spsa"


@pytest.mark.skipif(importlib.util.find_spec("pennylane") is None, reason="pennylane not installed")
def test_pennylane_noisy_expectation_switches_to_mixed_device() -> None:
    from qforge_ai.backends import PennyLaneAdapter

    ir = _ir()
    result = PennyLaneAdapter(noise=NoiseSpec(name="depolarizing_noise", probability=0.01)).run(
        ir,
        inputs=np.zeros(ir.parameter_groups()["input"]),
        weights=np.zeros(ir.parameter_groups()["weight"]),
    )
    assert result.as_array().shape == (2,)


@pytest.mark.skipif(
    importlib.util.find_spec("qiskit") is None or importlib.util.find_spec("pennylane") is None,
    reason="both engines are required",
)
def test_basis_and_u_gate_semantics_match() -> None:
    from qforge_ai.backends import PennyLaneAdapter, QiskitAdapter
    from qforge_ai.ir import CircuitIR, Measurement

    ir = CircuitIR(2)
    ir.add("BASIS", 0, 1).add("BASIS", 1, 0)
    ir.add("U", 1, 0.31, -0.22, 0.73)
    ir.measure(Measurement("statevector", (0, 1)))
    left = QiskitAdapter().run(ir).as_array()
    right = PennyLaneAdapter().run(ir).as_array()
    overlap = np.vdot(left, right)
    assert abs(overlap) ** 2 == pytest.approx(1.0)


@pytest.mark.skipif(importlib.util.find_spec("qiskit") is None, reason="qiskit not installed")
def test_qiskit_dynamic_ir_translation() -> None:
    from qforge_ai.backends import QiskitAdapter
    from qforge_ai.ir import CircuitIR

    ir = CircuitIR(2)
    ir.add("H", 0).measure_wire(0, 0).add_conditional("X", 1, clbit=0)
    circuit = QiskitAdapter().compile(ir)
    assert circuit.num_clbits == 1
    assert any(instruction.operation.name == "if_else" for instruction in circuit.data)


@pytest.mark.skipif(
    importlib.util.find_spec("qiskit") is None or importlib.util.find_spec("qiskit_aer") is None,
    reason="qiskit aer not installed",
)
def test_qiskit_dynamic_ir_execution() -> None:
    from qforge_ai.backends import QiskitAdapter
    from qforge_ai.ir import CircuitIR, Measurement

    ir = CircuitIR(2)
    ir.add("X", 0).measure_wire(0, 0).add_conditional("X", 1, clbit=0)
    ir.measure(Measurement("counts", (0, 1), shots=64))
    result = QiskitAdapter(shots=64).run(ir)
    assert result.values == {"11": 64}


@pytest.mark.skipif(importlib.util.find_spec("pennylane") is None, reason="pennylane required")
def test_pennylane_dynamic_ir_execution() -> None:
    from qforge_ai.backends import PennyLaneAdapter
    from qforge_ai.ir import CircuitIR, Measurement

    ir = CircuitIR(2)
    ir.add("X", 0).measure_wire(0, 0).add_conditional("X", 1, clbit=0)
    ir.measure(Measurement("probability", (0, 1)))
    result = PennyLaneAdapter().run(ir)
    assert result.as_array() == pytest.approx(np.asarray([0.0, 0.0, 0.0, 1.0]))


@pytest.mark.skipif(importlib.util.find_spec("pennylane") is None, reason="pennylane required")
def test_pennylane_reset_and_reuse() -> None:
    from qforge_ai.backends import PennyLaneAdapter
    from qforge_ai.ir import CircuitIR, Measurement

    ir = CircuitIR(1)
    ir.add("X", 0).measure_wire(0, 0).reset(0)
    ir.measure(Measurement("probability", (0,)))
    result = PennyLaneAdapter().run(ir)
    assert result.as_array() == pytest.approx(np.asarray([1.0, 0.0]))


@pytest.mark.skipif(
    any(importlib.util.find_spec(module) is None for module in ("qiskit", "qiskit_aer")),
    reason="qiskit aer is required",
)
def test_qiskit_readout_error_is_terminal_and_deterministic() -> None:
    from qforge_ai.backends import QiskitAdapter
    from qforge_ai.ir import CircuitIR, Measurement

    ir = CircuitIR(1)
    ir.add("X", 0).measure(Measurement("counts", (0,), shots=32))
    result = QiskitAdapter(
        shots=32,
        noise=NoiseSpec(name="readout_error", readout_probability=1.0),
    ).run(ir)
    assert result.values == {"0": 32}


@pytest.mark.skipif(importlib.util.find_spec("pennylane") is None, reason="pennylane required")
def test_pennylane_readout_error_is_terminal_and_deterministic() -> None:
    from qforge_ai.backends import PennyLaneAdapter
    from qforge_ai.ir import CircuitIR, Measurement

    ir = CircuitIR(1)
    ir.add("X", 0).measure(Measurement("counts", (0,), shots=32))
    result = PennyLaneAdapter(
        shots=32,
        noise=NoiseSpec(name="readout_error", readout_probability=1.0),
    ).run(ir)
    assert result.values == {"0": 32}


@pytest.mark.skipif(
    importlib.util.find_spec("qiskit") is None or importlib.util.find_spec("pennylane") is None,
    reason="both engines are required",
)
def test_custom_hermitian_observable_matches() -> None:
    from qforge_ai.backends import PennyLaneAdapter, QiskitAdapter
    from qforge_ai.ir import CircuitIR, Measurement, Observable

    matrix = ((1.0 + 0j, 0j), (0j, -1.0 + 0j))
    ir = CircuitIR(1)
    ir.add("RY", 0, 0.37)
    ir.measure(Measurement("expectation", (0,), (Observable("CUSTOM", (0,), matrix=matrix),)))
    left = QiskitAdapter().run(ir).as_array()
    right = PennyLaneAdapter().run(ir).as_array()
    assert left == pytest.approx(right)


@pytest.mark.skipif(
    importlib.util.find_spec("pennylane") is None or importlib.util.find_spec("torch") is None,
    reason="pennylane and torch are required",
)
def test_pennylane_torch_layer_batch_forward_backward() -> None:
    import torch

    from qforge_ai.models.pennylane_qnn import as_torch_layer

    ir = _ir()
    layer = as_torch_layer(ir, diff_method="parameter-shift")
    inputs = torch.randn(5, ir.parameter_groups()["input"], requires_grad=True)
    output = layer(inputs)
    assert output.shape == (5, 2)
    output.sum().backward()
    assert inputs.grad is not None
    assert inputs.grad.shape == inputs.shape


@pytest.mark.skipif(
    importlib.util.find_spec("pennylane") is None,
    reason="pennylane is required",
)
def test_pennylane_two_qubit_noise_consumes_p2() -> None:
    from qforge_ai.backends import PennyLaneAdapter

    ir = _ir()
    inputs = np.asarray([0.4, -0.2])
    weights = np.asarray([0.3, -0.1, 0.2, 0.5])
    low = PennyLaneAdapter(
        noise=NoiseSpec(
            name="depolarizing_noise",
            one_qubit_probability=0.0,
            two_qubit_probability=0.0,
        )
    ).run(ir, inputs=inputs, weights=weights).as_array()
    high = PennyLaneAdapter(
        noise=NoiseSpec(
            name="depolarizing_noise",
            one_qubit_probability=0.0,
            two_qubit_probability=0.2,
        )
    ).run(ir, inputs=inputs, weights=weights).as_array()
    assert not np.allclose(low, high)


@pytest.mark.skipif(
    any(
        importlib.util.find_spec(module) is None
        for module in ("qiskit", "qiskit_aer", "pennylane")
    ),
    reason="qiskit aer and pennylane are required",
)
def test_noisy_cross_engine_expectation_is_within_model_tolerance() -> None:
    from qforge_ai.backends import PennyLaneAdapter, QiskitAdapter
    from qforge_ai.benchmark import compare_engines

    ir = _ir()
    noise = NoiseSpec(
        name="depolarizing_noise",
        one_qubit_probability=0.0,
        two_qubit_probability=0.02,
    )
    report = compare_engines(
        ir,
        inputs=np.asarray([0.2, -0.4]),
        weights=np.asarray([0.1, 0.2, 0.3, 0.4]),
        measurement="expectation",
        qiskit_adapter=QiskitAdapter(noise=noise),
        pennylane_adapter=PennyLaneAdapter(noise=noise),
        tolerance=2e-3,
    )
    assert report.passed, report


@pytest.mark.skipif(
    any(
        importlib.util.find_spec(module) is None
        for module in ("qiskit", "qiskit_aer", "pennylane")
    ),
    reason="qiskit aer and pennylane are required",
)
def test_real_finite_shot_benchmark_reports_distribution_statistics() -> None:
    from qforge_ai.backends import PennyLaneAdapter, QiskitAdapter
    from qforge_ai.benchmark import compare_engines

    ir = _ir()
    report = compare_engines(
        ir,
        inputs=np.asarray([0.27, -0.41]),
        weights=np.asarray([0.13, -0.2, 0.37, 0.04]),
        measurement="probability",
        qiskit_adapter=QiskitAdapter(shots=512, seed=17),
        pennylane_adapter=PennyLaneAdapter(shots=512, seed=17),
    )
    assert report.qiskit_shots == report.pennylane_shots == 512
    assert report.total_variation_distance is not None
    assert report.hellinger_distance is not None
    assert report.statistical_tolerance is not None
    assert report.total_variation_confidence_interval is not None
    assert report.confidence_level == 0.95
    assert report.passed, report


@pytest.mark.skipif(
    importlib.util.find_spec("qiskit_machine_learning") is None,
    reason="qiskit machine learning not installed",
)
def test_qsvc_uses_encoding_operations_only() -> None:
    from qforge_ai.models import build_qsvc

    ir = _ir()
    model = build_qsvc(ir)
    # ``feature_map`` is the framework's own CircuitIR on both engines: it used
    # to be a Qiskit QuantumCircuit on one route and unavailable on the other.
    feature_map = model.quantum_kernel.feature_map
    assert "weight" not in feature_map.parameter_groups()
    assert all(op.metadata.get("role") == "encoding" for op in feature_map.operations)
    compiled = model.quantum_kernel.engine_feature_map()
    assert all(not str(parameter).startswith("weight") for parameter in compiled.parameters)


@pytest.mark.skipif(
    any(importlib.util.find_spec(module) is None for module in ("qiskit", "qiskit_aer")),
    reason="qiskit aer is required",
)
@pytest.mark.parametrize("measurement", ["probability", "expectation"])
def test_qiskit_finite_shot_probability_and_expectation_are_sampled(
    measurement: str,
) -> None:
    from qforge_ai.backends import QiskitAdapter

    ir = CircuitFactory().build(
        CircuitSpec(
            2,
            EncodingSpec("angle", 2),
            AnsatzSpec("real_amplitudes", layers=1),
            MeasurementSpec(measurement, wires=(0,)),
        )
    )
    result = QiskitAdapter(shots=128, seed=19).run(
        ir,
        inputs=np.asarray([0.27, -0.11]),
        weights=np.zeros(ir.parameter_groups()["weight"]),
    )
    assert result.shots == 128
    assert result.metadata["requested_shots"] == 128
    assert result.metadata["actual_shots"] == 128
    assert result.metadata["standard_errors"]
    if measurement == "probability":
        assert result.as_array().sum() == pytest.approx(1.0)
        assert np.allclose(result.as_array() * 128, np.round(result.as_array() * 128))
    else:
        assert result.metadata["estimator_shots"] == 128
        assert result.metadata["estimator_precision"] == pytest.approx(1 / np.sqrt(128))


@pytest.mark.skipif(
    importlib.util.find_spec("qiskit_machine_learning") is None,
    reason="qiskit machine learning not installed",
)
def test_qiskit_qnn_consumes_backend_spec_and_sampler_measurement_wires() -> None:
    from qforge_ai.models import build_estimator_qnn, build_sampler_qnn
    from qforge_ai.specs import BackendSpec, Engine

    ir = CircuitFactory().build(
        CircuitSpec(
            3,
            EncodingSpec("angle", 3),
            AnsatzSpec("real_amplitudes", layers=1),
            MeasurementSpec("probability", wires=(2, 0)),
        )
    )
    backend = BackendSpec(
        engine=Engine.QISKIT,
        device="aer",
        shots=64,
        method="matrix_product_state",
        seed=23,
        noise=NoiseSpec(name="depolarizing_noise", probability=0.01),
        options={"backend_options": {"max_parallel_threads": 1}},
    )
    sampler = build_sampler_qnn(ir, backend=backend)
    assert sampler.qforge_backend_spec == backend
    assert sampler.qforge_measurement_wires == (2, 0)
    assert sampler.output_shape == (4,)
    assert sampler.circuit.num_clbits == 2
    measured = [item for item in sampler.circuit.data if item.operation.name == "measure"]
    assert [sampler.circuit.find_bit(item.qubits[0]).index for item in measured] == [2, 0]
    assert sampler.forward(
        np.zeros(ir.parameter_groups()["input"]),
        np.zeros(ir.parameter_groups()["weight"]),
    ).shape == (1, 4)

    expectation_ir = ir.copy()
    expectation_ir.measurements = [
        expectation_ir.measurements[0].__class__(
            "expectation",
            (0,),
            (Observable("Z", (0,)),),
        )
    ]
    estimator = build_estimator_qnn(expectation_ir, backend=backend)
    assert estimator.qforge_backend_spec == backend
    assert estimator.qforge_shots == 64
    assert estimator.qforge_primitive_kind == "backend_estimator_v2_finite_shot"
    assert estimator.estimator.options.seed_simulator == 23
    assert estimator.estimator.backend.options.method == "matrix_product_state"
    assert estimator.estimator.backend.options.max_parallel_threads == 1
    # Gate noise now travels inside the circuit, applied at exactly the IR
    # operations PennyLane also applies it to, so the backend carries no
    # noise model -- attaching one as well would apply every channel twice.
    assert estimator.estimator.backend.options.noise_model is None
    assert any(
        item.operation.name == "kraus" for item in estimator.circuit.data
    ), "noise channel must be present in the compiled QNN circuit"

    measurement_shot_ir = CircuitFactory().build(
        CircuitSpec(
            1,
            EncodingSpec("angle", 1),
            AnsatzSpec("real_amplitudes", layers=1),
            MeasurementSpec("expectation", wires=(0,), shots=48),
        )
    )
    measurement_shot_qnn = build_estimator_qnn(
        measurement_shot_ir,
        backend=BackendSpec(engine=Engine.QISKIT, device="aer", seed=11),
    )
    assert measurement_shot_qnn.qforge_shots == 48
    assert measurement_shot_qnn.qforge_backend_spec.shots == 48


@pytest.mark.skipif(
    any(
        importlib.util.find_spec(module) is None
        for module in ("qiskit_machine_learning", "qiskit_aer", "torch")
    ),
    reason="qiskit machine learning, aer and torch are required",
)
def test_qiskit_finite_shot_qnn_forward_and_backward() -> None:
    import torch

    from qforge_ai.models import ModelFactory
    from qforge_ai.specs import BackendSpec, Engine, ModelMode, ModelSpec

    ir = CircuitFactory().build(
        CircuitSpec(
            1,
            EncodingSpec("angle", 1),
            AnsatzSpec("real_amplitudes", layers=1),
            MeasurementSpec("expectation", wires=(0,)),
        )
    )
    model = ModelFactory().build(
        ir,
        ModelSpec(
            mode=ModelMode.QUANTUM_ONLY,
            template="quantum_classifier",
            # One observable on one wire: a quantum-only layer emits exactly
            # one value and cannot be asked for two.
            output_dimension=1,
        ),
        BackendSpec(engine=Engine.QISKIT, device="aer", shots=32, seed=5),
    )
    inputs = torch.tensor([[0.2]], requires_grad=True)
    output = model(inputs)
    output.sum().backward()
    assert output.shape == (1, 1)
    assert inputs.grad is not None
    assert model.weight.grad is not None


@pytest.mark.skipif(
    any(
        importlib.util.find_spec(module) is None
        for module in ("qiskit_machine_learning", "torch")
    ),
    reason="qiskit machine learning and torch are required",
)
def test_qiskit_custom_sampler_shape_matches_hybrid_head() -> None:
    import torch

    from qforge_ai.models import ModelFactory
    from qforge_ai.specs import BackendSpec, Engine, ModelMode, ModelSpec

    ir = CircuitFactory().build(
        CircuitSpec(
            2,
            EncodingSpec("angle", 2),
            AnsatzSpec("real_amplitudes", layers=1),
            MeasurementSpec("probability", wires=(0, 1)),
        )
    )
    model = ModelFactory().build(
        ir,
        ModelSpec(
            mode=ModelMode.HYBRID,
            template="classical_quantum_classical",
            input_dimension=2,
            output_dimension=3,
            options={
                "interpret": lambda value: value.bit_count() % 2,
                "sampler_output_shape": 2,
            },
        ),
        BackendSpec(engine=Engine.QISKIT, shots=512),
    )
    assert model.head.in_features == 2
    assert model(torch.randn(4, 2)).shape == (4, 3)


@pytest.mark.skipif(importlib.util.find_spec("qiskit") is None, reason="qiskit required")
@pytest.mark.parametrize("dimension", [2, 4])
def test_qiskit_amplitude_normalize_flag_is_enforced(dimension: int) -> None:
    from qforge_ai.backends import QiskitAdapter
    from qforge_ai.errors import ConfigurationError

    invalid = np.asarray([1.0, 1.0, 0.0, 0.0])[:dimension]
    strict_ir = CircuitFactory().build(
        CircuitSpec(
            2,
            EncodingSpec("amplitude", dimension, normalize=False),
            AnsatzSpec("real_amplitudes", layers=1),
            MeasurementSpec("probability"),
        )
    )
    with pytest.raises(ConfigurationError, match="normalize=False"):
        QiskitAdapter().run(
            strict_ir,
            inputs=invalid,
            weights=np.zeros(strict_ir.parameter_groups()["weight"]),
        )
    normalized_ir = CircuitFactory().build(
        CircuitSpec(
            2,
            EncodingSpec("amplitude", dimension, normalize=True),
            AnsatzSpec("real_amplitudes", layers=1),
            MeasurementSpec("probability"),
        )
    )
    result = QiskitAdapter().run(
        normalized_ir,
        inputs=invalid,
        weights=np.zeros(normalized_ir.parameter_groups()["weight"]),
    )
    assert result.as_array().sum() == pytest.approx(1.0)


@pytest.mark.skipif(importlib.util.find_spec("pennylane") is None, reason="pennylane required")
@pytest.mark.parametrize("dimension", [2, 4])
def test_pennylane_amplitude_normalize_flag_and_seed_are_enforced(dimension: int) -> None:
    from qforge_ai.backends import PennyLaneAdapter
    from qforge_ai.errors import ExecutionError

    invalid = np.asarray([1.0, 1.0, 0.0, 0.0])[:dimension]
    strict_ir = CircuitFactory().build(
        CircuitSpec(
            2,
            EncodingSpec("amplitude", dimension, normalize=False),
            AnsatzSpec("real_amplitudes", layers=1),
            MeasurementSpec("probability", shots=64),
        )
    )
    with pytest.raises(ExecutionError, match="norm"):
        PennyLaneAdapter(shots=64, seed=31).run(
            strict_ir,
            inputs=invalid,
            weights=np.zeros(strict_ir.parameter_groups()["weight"]),
        )
    normalized_ir = CircuitFactory().build(
        CircuitSpec(
            2,
            EncodingSpec("amplitude", dimension, normalize=True),
            AnsatzSpec("real_amplitudes", layers=1),
            MeasurementSpec("probability", shots=64),
        )
    )
    kwargs = {
        "inputs": invalid,
        "weights": np.zeros(normalized_ir.parameter_groups()["weight"]),
    }
    first = PennyLaneAdapter(shots=64, seed=31).run(normalized_ir, **kwargs)
    second = PennyLaneAdapter(shots=64, seed=31).run(normalized_ir, **kwargs)
    assert first.shots == second.shots == 64
    assert np.array_equal(first.as_array(), second.as_array())
