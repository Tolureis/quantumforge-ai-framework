"""Regression tests for the defects reported against 1.2.3.

Every test here fails on 1.2.3 and passes on 1.2.4.  They are grouped by the
numbered finding so a future change that reintroduces one is attributed
immediately.

The findings, in the reporter's numbering:

1. ``counts`` bit order differed between engines (critical)
2. the noisy Qiskit EstimatorQNN default disagreed with direct execution (critical)
3. PennyLane gradient options broke every real model (major)
4. the PennyLane kernel contract disagreed with the Qiskit one (major)
5. the PennyLane QSVC was not a scikit-learn estimator
6. the IR accepted complex rotation parameters and observable coefficients
7. the core CI job failed on a dependency-light install
8. the catalog verified components but never their composition
"""

from __future__ import annotations

import importlib.util
import pickle

import numpy as np
import pytest

from qforge_ai import (
    AnsatzSpec,
    BackendSpec,
    CircuitSpec,
    EncodingSpec,
    MeasurementSpec,
    ModelSpec,
    NoiseSpec,
)
from qforge_ai.circuits import CircuitFactory
from qforge_ai.errors import ConfigurationError
from qforge_ai.ir import CircuitIR, Measurement, Observable, ParameterRef
from qforge_ai.specs import Engine, GradientMethod, ModelMode

HAS_QISKIT = importlib.util.find_spec("qiskit") is not None
HAS_AER = importlib.util.find_spec("qiskit_aer") is not None
HAS_QML = importlib.util.find_spec("qiskit_machine_learning") is not None
HAS_PENNYLANE = importlib.util.find_spec("pennylane") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None

requires_aer = pytest.mark.skipif(not (HAS_QISKIT and HAS_AER), reason="qiskit-aer not installed")
requires_qml = pytest.mark.skipif(
    not (HAS_QISKIT and HAS_AER and HAS_QML), reason="qiskit-machine-learning not installed"
)
requires_pennylane = pytest.mark.skipif(not HAS_PENNYLANE, reason="pennylane not installed")
requires_both = pytest.mark.skipif(
    not (HAS_QISKIT and HAS_AER and HAS_PENNYLANE), reason="both engines required"
)
requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
requires_sklearn = pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not installed")


def _model_ir(measurement: str = "expectation_measurement") -> CircuitIR:
    return CircuitFactory().build(
        CircuitSpec(
            num_qubits=2,
            encoding=EncodingSpec(name="angle_encoding", feature_dimension=2),
            ansatz=AnsatzSpec(name="real_amplitudes", layers=1),
            measurement=MeasurementSpec(name=measurement, wires=(0, 1)),
        )
    )


def _kernel_ir() -> CircuitIR:
    return CircuitFactory().build(
        CircuitSpec(
            num_qubits=2,
            encoding=EncodingSpec(name="zz_feature_map", feature_dimension=2),
            ansatz=AnsatzSpec(name="real_amplitudes", layers=1),
            measurement=MeasurementSpec(name="expectation_measurement", wires=(0, 1)),
        )
    )


# --------------------------------------------------------------------------
# 1. counts must use the framework's wire-major bit order on both engines
# --------------------------------------------------------------------------


@requires_both
@pytest.mark.parametrize("wires", [(0, 1), (1, 2), (0, 1, 2)])
def test_counts_use_the_same_bit_order_on_both_engines(wires) -> None:
    """``X`` on wire 1 read ``{'10': n}`` on Qiskit and ``{'01': n}`` here.

    ``sample`` and ``probability`` were both converted to wire-major order;
    only the raw ``counts`` dictionary kept Qiskit's little-endian spelling, so
    any bitstring-keyed analysis -- class labels above all -- silently
    transposed when the engine changed.
    """

    from qforge_ai.backends import PennyLaneAdapter, QiskitAdapter

    ir = CircuitIR(3, name="bit_order")
    ir.add("X", 1)
    ir.measure(Measurement("counts", wires=wires, shots=64))
    qiskit_counts = QiskitAdapter(shots=64, seed=5).run(ir).values
    pennylane_counts = PennyLaneAdapter(shots=64, seed=5).run(ir).values
    expected = "".join("1" if wire == 1 else "0" for wire in wires)
    assert set(qiskit_counts) == {expected}
    assert set(pennylane_counts) == {expected}


@requires_both
def test_counts_bit_order_survives_mid_circuit_measurement() -> None:
    """Terminal counts must not leak the internal classical bits either."""

    from qforge_ai.backends import PennyLaneAdapter, QiskitAdapter

    ir = CircuitIR(2, name="dynamic_counts", num_clbits=1)
    ir.add("X", 0)
    ir.measure_wire(0, 0)
    ir.add_conditional("X", 1, clbit=0)
    ir.measure(Measurement("counts", wires=(0, 1), shots=64))
    qiskit_counts = QiskitAdapter(shots=64, seed=5).run(ir).values
    pennylane_counts = PennyLaneAdapter(shots=64, seed=5).run(ir).values
    assert set(qiskit_counts) == {"11"}
    assert set(pennylane_counts) == {"11"}


@requires_both
def test_counts_are_plain_python_types_on_both_engines() -> None:
    from qforge_ai.backends import PennyLaneAdapter, QiskitAdapter

    ir = CircuitIR(2, name="counts_types")
    ir.add("H", 0)
    ir.measure(Measurement("counts", wires=(0, 1), shots=32))
    for adapter in (QiskitAdapter(shots=32), PennyLaneAdapter(shots=32)):
        counts = adapter.run(ir).values
        assert all(type(key) is str for key in counts)
        assert all(type(value) is int for value in counts.values())


@requires_both
def test_benchmark_no_longer_compensates_for_a_bit_order_difference() -> None:
    """The comparison used to flip Qiskit's keys, hiding finding 1."""

    from qforge_ai.benchmark import compare_engines

    ir = CircuitIR(2, name="benchmark_counts")
    ir.add("X", 1)
    report = compare_engines(
        ir, inputs=(), weights=(), measurement="counts", tolerance=1e-9
    )
    assert report.comparable
    assert report.total_variation_distance == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------
# 2. a noisy analytic EstimatorQNN must reproduce direct execution
# --------------------------------------------------------------------------


@requires_qml
@requires_pennylane
@pytest.mark.parametrize(
    "noise",
    [
        NoiseSpec(name="bit_flip", probability=0.13),
        NoiseSpec(name="depolarizing", probability=0.2),
        NoiseSpec(name="amplitude_damping", probability=0.3),
        NoiseSpec(name="thermal_relaxation", t1=60.0, t2=40.0, gate_time=5.0),
    ],
)
def test_noisy_estimator_qnn_matches_direct_execution(noise) -> None:
    """``method='automatic'`` let Aer pick its own strategy for a Kraus circuit.

    With ``bit_flip(p=0.13)`` the default QNN returned ``[0.2792, 0.2767]``
    against ``[0.2907, 0.2615]`` from both direct adapters -- a 1.5e-2 error on
    a quantity the two engines otherwise agree on to 1e-16.  A noisy analytic
    expectation value is ``Tr[rho O]``, so the primitive is now pinned to the
    same density-matrix method direct execution already forced.
    """

    from qforge_ai.backends import PennyLaneAdapter, QiskitAdapter
    from qforge_ai.models.qiskit_qnn import build_estimator_qnn

    ir = _model_ir()
    inputs = [0.3, -0.7]
    weights = list(np.linspace(0.1, 0.9, ir.parameter_groups().get("weight", 0)))
    direct_qiskit = np.asarray(
        QiskitAdapter(noise=noise).run(ir, inputs=inputs, weights=weights).values, dtype=float
    )
    direct_pennylane = np.asarray(
        PennyLaneAdapter(noise=noise).run(ir, inputs=inputs, weights=weights).values, dtype=float
    )
    assert np.allclose(direct_qiskit, direct_pennylane, atol=1e-12)

    qnn = build_estimator_qnn(
        ir, backend=BackendSpec(engine=Engine.QISKIT, device="aer", noise=noise)
    )
    values = np.asarray(
        qnn.forward(np.asarray([inputs]), np.asarray(weights)), dtype=float
    ).reshape(-1)
    assert np.allclose(values, direct_qiskit, atol=1e-12)
    assert qnn.qforge_primitive_kind == "aer_estimator_v2_analytic[density_matrix]"


@requires_aer
def test_analytic_method_is_reported_and_only_forced_when_noisy() -> None:
    from qforge_ai.backends import QiskitAdapter

    ideal = QiskitAdapter(backend_name="aer")
    noisy = QiskitAdapter(backend_name="aer", noise=NoiseSpec(name="bit_flip", probability=0.1))
    assert ideal.analytic_method() == "automatic"
    assert noisy.analytic_method() == "density_matrix"
    # 'ideal' is a documented alias for "no channel" and must not force it.
    alias = QiskitAdapter(backend_name="aer", noise=NoiseSpec(name="ideal"))
    assert alias.analytic_method() == "automatic"


@requires_qml
def test_estimator_qnn_refuses_readout_noise_it_cannot_apply() -> None:
    """An EstimatorQNN measures nothing, so a readout channel was dropped."""

    from qforge_ai.models.qiskit_qnn import build_estimator_qnn

    with pytest.raises(ConfigurationError, match="readout"):
        build_estimator_qnn(
            _model_ir(),
            backend=BackendSpec(
                engine=Engine.QISKIT,
                device="aer",
                noise=NoiseSpec(name="readout_error", readout_probability=0.1),
            ),
        )


# --------------------------------------------------------------------------
# 3. PennyLane gradient options must reach the gradient transform
# --------------------------------------------------------------------------


@requires_pennylane
@requires_torch
@pytest.mark.parametrize(
    "gradient,options",
    [
        (GradientMethod.FINITE_DIFF, {"epsilon": 0.01}),
        (GradientMethod.FINITE_DIFF, {"epsilon": 0.01, "method": "forward"}),
        (GradientMethod.FINITE_DIFF, {"h": 0.02, "approx_order": 2}),
        (GradientMethod.SPSA, {"epsilon": 0.01, "batch_size": 2}),
    ],
)
def test_pennylane_gradient_options_build_and_backpropagate(gradient, options) -> None:
    """Every finite-diff/SPSA model raised before it could run a forward pass.

    The normalized options were splatted into ``QNode.__init__`` instead of
    being passed as ``gradient_kwargs``, so building the model died with
    ``TypeError: QNode.__init__() got an unexpected keyword argument 'h'``.
    The existing test only checked the option *dictionary* conversion, which
    is why the defect survived.
    """

    import torch

    from qforge_ai.models import ModelFactory

    ir = _model_ir()
    backend = BackendSpec(
        engine=Engine.PENNYLANE,
        device="default.qubit",
        gradient=gradient,
        options={"gradient_options": options},
    )
    model = ModelFactory().build(
        ir,
        ModelSpec(mode=ModelMode.HYBRID, input_dimension=2, output_dimension=2),
        backend,
    )
    inputs = torch.rand(3, 2, requires_grad=True)
    output = model(inputs)
    assert output.shape == (3, 2)
    output.sum().backward()
    assert inputs.grad is not None
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


@requires_pennylane
def test_gradient_options_actually_change_the_computed_gradient() -> None:
    """A step size that reaches the transform must change the answer."""

    import pennylane as qml

    from qforge_ai.backends import PennyLaneAdapter

    ir = CircuitIR(1, name="gradient_options")
    ir.add("RX", 0, ParameterRef("weight", 0))
    ir.measure(Measurement("expectation", wires=(0,), observables=(Observable("Z", (0,)),)))

    def slope(epsilon: float) -> float:
        adapter = PennyLaneAdapter(gradient_options={"epsilon": epsilon, "method": "forward"})
        qnode = adapter.compile(ir, interface="autograd", diff_method="finite-diff")
        weights = qml.numpy.array([0.7], requires_grad=True)
        return float(qml.jacobian(lambda w: qml.math.stack(qnode(np.zeros(0), w)))(weights)[0][0])

    coarse = slope(0.5)
    fine = slope(1e-4)
    exact = -np.sin(0.7)
    assert abs(fine - exact) < 1e-3
    # A 0.5 forward step cannot land on the exact derivative; if the option had
    # been dropped again both calls would return the same default-step value.
    assert abs(coarse - fine) > 1e-3


# --------------------------------------------------------------------------
# 4. one kernel contract, and a Gram matrix that is a Mercer kernel
# --------------------------------------------------------------------------


@requires_qml
@requires_pennylane
@requires_sklearn
@pytest.mark.parametrize(
    "template,options,backend_kwargs",
    [
        # Sampler fidelity with no shots: Qiskit raised, PennyLane silently
        # built an analytic kernel instead.
        ("quantum_kernel_model", {"fidelity": "sampler"}, {}),
        # A statevector kernel cannot carry noise: Qiskit raised, PennyLane
        # silently built a noisy kernel and called it analytic.
        (
            "statevector_quantum_kernel",
            {},
            {"noise": NoiseSpec(name="amplitude_damping", probability=0.2)},
        ),
        # Nor a shot budget.
        ("statevector_quantum_kernel", {}, {"shots": 128}),
    ],
)
def test_kernel_contract_is_identical_on_both_engines(
    template, options, backend_kwargs
) -> None:
    from qforge_ai.models import ModelFactory

    ir = _kernel_ir()
    spec = ModelSpec(
        mode=ModelMode.QUANTUM_KERNEL,
        template=template,
        input_dimension=2,
        output_dimension=2,
        options=options,
    )
    for engine, device in ((Engine.QISKIT, "aer"), (Engine.PENNYLANE, "default.qubit")):
        with pytest.raises(ConfigurationError):
            ModelFactory().build(
                ir, spec, BackendSpec(engine=engine, device=device, **backend_kwargs)
            )


@requires_qml
@requires_sklearn
def test_ideal_noise_alias_is_not_treated_as_a_channel_by_the_qiskit_kernel() -> None:
    """``NoiseSpec(name='ideal')`` is documented as "no noise"."""

    from qforge_ai.models.qiskit_qnn import build_quantum_kernel

    _, kind = build_quantum_kernel(
        _kernel_ir(),
        backend=BackendSpec(engine=Engine.QISKIT, device="aer", noise=NoiseSpec(name="ideal")),
        fidelity="statevector",
    )
    assert kind == "fidelity_statevector_kernel"


@requires_pennylane
def test_fidelity_option_actually_reaches_the_pennylane_builder() -> None:
    """It was resolved by the factory and then never passed on."""

    from qforge_ai.models.pennylane_qnn import build_fidelity_kernel

    backend = BackendSpec(engine=Engine.PENNYLANE, device="default.qubit", shots=256)
    _, kind = build_fidelity_kernel(_kernel_ir(), backend=backend, fidelity="sampler")
    assert kind == "pennylane_fidelity_kernel[sampler]"
    with pytest.raises(ConfigurationError, match="statevector"):
        build_fidelity_kernel(_kernel_ir(), backend=backend, fidelity="statevector")


@requires_pennylane
@pytest.mark.parametrize(
    "channel", ["amplitude_damping", "phase_damping", "depolarizing", "bit_flip"]
)
def test_noisy_gram_matrix_is_a_valid_mercer_kernel(channel) -> None:
    """Noisy compute-uncompute produced an asymmetric, non-PSD "kernel".

    Under amplitude damping the reported Gram matrix was asymmetric by up to
    0.137.  ``Tr[rho(x) rho(y)]`` is an inner product of the two encoded
    states, so it is symmetric and positive semi-definite by construction.
    """

    from qforge_ai.models.pennylane_qnn import build_fidelity_kernel

    backend = BackendSpec(
        engine=Engine.PENNYLANE,
        device="default.qubit",
        noise=NoiseSpec(name=channel, probability=0.25),
    )
    kernel, kind = build_fidelity_kernel(_kernel_ir(), backend=backend)
    # Renamed in 1.2.5: ``Tr[rho(x) rho(y)]`` is the Hilbert-Schmidt inner
    # product, not the mixed-state fidelity.  See the 1.2.5 defect suite.
    assert kind == "hilbert_schmidt_kernel[pennylane_density_matrix]"
    samples = np.array([[0.1, 0.9], [1.2, -0.4], [-0.8, 0.3], [0.5, 0.5]])
    gram = kernel(samples)
    assert np.allclose(gram, gram.T, atol=1e-12)
    assert np.min(np.linalg.eigvalsh(gram)) > -1e-10
    # The diagonal is the purity of the encoded state: below 1 under a
    # state-changing channel, which is physics rather than an error.
    assert np.all(np.diag(gram) <= 1.0 + 1e-12)


@requires_pennylane
def test_sampled_gram_matrix_is_symmetrized_and_projected() -> None:
    from qforge_ai.models.pennylane_qnn import build_fidelity_kernel

    backend = BackendSpec(engine=Engine.PENNYLANE, device="default.qubit", shots=256, seed=11)
    kernel, kind = build_fidelity_kernel(_kernel_ir(), backend=backend, fidelity="sampler")
    assert kind == "pennylane_fidelity_kernel[sampler]"
    gram = kernel(np.array([[0.1, 0.9], [1.2, -0.4], [-0.8, 0.3]]))
    assert np.allclose(gram, gram.T, atol=1e-12)
    assert np.min(np.linalg.eigvalsh(gram)) > -1e-10


@requires_pennylane
def test_enforce_psd_repairs_a_sampled_estimate() -> None:
    from qforge_ai.models.pennylane_qnn import enforce_psd

    noisy = np.array([[1.0, 0.9, 0.1], [0.4, 1.0, 0.2], [0.1, 0.7, 1.0]])
    repaired = enforce_psd(noisy)
    assert np.allclose(repaired, repaired.T, atol=1e-12)
    assert np.min(np.linalg.eigvalsh(repaired)) > -1e-10


@requires_both
@requires_sklearn
@requires_qml
def test_analytic_kernels_still_agree_across_engines() -> None:
    """The rewrite must not move the analytic answer."""

    from qforge_ai.models.pennylane_qnn import build_fidelity_kernel
    from qforge_ai.models.qiskit_qnn import build_quantum_kernel

    ir = _kernel_ir()
    samples = np.array([[0.1, 0.2], [1.3, 1.4], [0.7, -0.5]])
    qiskit_kernel, _ = build_quantum_kernel(
        ir, backend=BackendSpec(engine=Engine.QISKIT, device="statevector")
    )
    pennylane_kernel, _ = build_fidelity_kernel(
        ir, backend=BackendSpec(engine=Engine.PENNYLANE, device="default.qubit")
    )
    left = np.asarray(qiskit_kernel.evaluate(samples))
    right = pennylane_kernel(samples)
    assert np.allclose(left, right, atol=1e-8)
    assert np.allclose(np.diag(right), 1.0, atol=1e-8)


# --------------------------------------------------------------------------
# 5. the PennyLane QSVC must be a real scikit-learn estimator
# --------------------------------------------------------------------------


@requires_pennylane
@requires_sklearn
def test_pennylane_qsvc_satisfies_the_sklearn_estimator_api() -> None:
    from sklearn.base import clone

    from qforge_ai.models.pennylane_qnn import build_pennylane_qsvc

    backend = BackendSpec(engine=Engine.PENNYLANE, device="default.qubit")
    estimator = build_pennylane_qsvc(_kernel_ir(), backend=backend, C=3.0)

    params = estimator.get_params()
    assert params, "get_params() returned an empty dict"
    assert params["C"] == 3.0
    assert "quantum_kernel" in params

    estimator.set_params(C=2.5)
    assert estimator.C == 2.5
    assert clone(estimator).C == 2.5


@requires_pennylane
@requires_sklearn
def test_pennylane_qsvc_pickles_after_fitting() -> None:
    from qforge_ai.models.pennylane_qnn import build_pennylane_qsvc

    estimator = build_pennylane_qsvc(
        _kernel_ir(), backend=BackendSpec(engine=Engine.PENNYLANE, device="default.qubit")
    )
    samples = np.array([[0.1, 0.2], [1.3, 1.4], [0.15, 0.25], [1.35, 1.45]])
    labels = np.array([0, 1, 0, 1])
    estimator.fit(samples, labels)
    restored = pickle.loads(pickle.dumps(estimator))
    assert np.array_equal(estimator.predict(samples), restored.predict(samples))


@requires_pennylane
@requires_sklearn
def test_pennylane_qsvc_works_inside_gridsearch_and_pipeline() -> None:
    from sklearn.model_selection import GridSearchCV, cross_val_score
    from sklearn.pipeline import Pipeline

    from qforge_ai.models.pennylane_qnn import build_pennylane_qsvc

    backend = BackendSpec(engine=Engine.PENNYLANE, device="default.qubit")
    samples = np.array(
        [[0.1, 0.2], [1.3, 1.4], [0.15, 0.25], [1.35, 1.45], [0.2, 0.1], [1.4, 1.3]]
    )
    labels = np.array([0, 1, 0, 1, 0, 1])

    search = GridSearchCV(
        build_pennylane_qsvc(_kernel_ir(), backend=backend), {"C": [0.5, 4.0]}, cv=2
    )
    search.fit(samples, labels)
    assert search.best_params_["C"] in {0.5, 4.0}

    scores = cross_val_score(
        build_pennylane_qsvc(_kernel_ir(), backend=backend), samples, labels, cv=3
    )
    assert len(scores) == 3

    pipeline = Pipeline([("qsvc", build_pennylane_qsvc(_kernel_ir(), backend=backend))])
    pipeline.fit(samples, labels)
    assert pipeline.predict(samples).shape == (6,)


def test_pennylane_extra_installs_scikit_learn() -> None:
    """The error told the user to install an extra that did not supply it."""

    import tomllib
    from pathlib import Path

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    if not pyproject.exists():  # installed without the sdist layout
        pytest.skip("pyproject.toml not available")
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    extras = data["project"]["optional-dependencies"]
    for name in ("pennylane", "all"):
        assert any(item.startswith("scikit-learn") for item in extras[name]), name


# --------------------------------------------------------------------------
# 6. rotation parameters and observable coefficients must be real
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,wires,params",
    [
        ("RX", 0, (1j,)),
        ("RY", 0, (0.5 + 0.5j,)),
        ("RZ", 0, (np.complex128(1.0),)),
        ("U", 0, (1j, 0.0, 0.0)),
        ("RXX", (0, 1), (1j,)),
        ("CRZ", (0, 1), (1j,)),
        ("RZ_PRODUCT", 0, (1j, 1.0)),
        ("RZZ_PRODUCT", (0, 1), (1j, 1.0)),
        ("BASIS", 0, (1j,)),
    ],
)
def test_ir_rejects_complex_rotation_parameters(name, wires, params) -> None:
    """A complex angle makes the gate non-unitary.

    Qiskit refused it outright; PennyLane built the gate and returned a
    statevector of norm 1.24 -- an unphysical state presented as a result.
    """

    with pytest.raises(ConfigurationError, match="real number"):
        CircuitIR(2, name="complex_parameter").add(name, wires, *params)


def test_state_preparation_still_accepts_complex_amplitudes() -> None:
    """Amplitudes are the one legitimately complex parameter family."""

    ir = CircuitIR(1, name="amplitudes")
    ir.add("STATE_PREP", (0,), 0.6 + 0j, 0.8j)
    assert len(ir.operations) == 1


def test_observable_coefficient_must_be_real() -> None:
    """``c * H`` is Hermitian only for real ``c``."""

    with pytest.raises(ConfigurationError, match="real number"):
        Observable("Z", wires=(0,), coefficient=1j)


def test_parameter_reference_affine_map_must_be_real() -> None:
    for kwargs in ({"scale": 1j}, {"offset": 1j}):
        with pytest.raises(ConfigurationError, match="real number"):
            ParameterRef("input", 0, **kwargs)


def test_global_phase_must_be_real() -> None:
    with pytest.raises(ConfigurationError, match="real number"):
        CircuitIR(1, name="phase", global_phase=1j)


@requires_pennylane
def test_no_unphysical_state_can_be_produced_from_a_validated_ir() -> None:
    """The end the validation exists to serve: norm-1 states only."""

    from qforge_ai.backends import PennyLaneAdapter

    ir = CircuitIR(1, name="unit_norm")
    ir.add("RX", 0, 1.1)
    ir.measure(Measurement("statevector", wires=(0,)))
    state = np.asarray(PennyLaneAdapter().run(ir).values)
    assert np.linalg.norm(state) == pytest.approx(1.0, abs=1e-12)


# --------------------------------------------------------------------------
# 7. configuration errors must not be masked by a missing optional package
# --------------------------------------------------------------------------


class _ImportBlocker:
    """Hide a top-level package from ``import`` for the duration of a test."""

    def __init__(self, blocked: str) -> None:
        self.blocked = blocked

    def find_spec(self, name, path=None, target=None):  # noqa: ANN001, ANN201
        if name.split(".")[0] == self.blocked:
            raise ImportError(f"No module named {name!r} (blocked for this test)")
        return None


@pytest.fixture()
def without_qiskit_machine_learning(monkeypatch):
    import sys

    blocker = _ImportBlocker("qiskit_machine_learning")
    monkeypatch.setattr(sys, "meta_path", [blocker, *sys.meta_path])
    for name in list(sys.modules):
        if name.split(".")[0] == "qiskit_machine_learning":
            monkeypatch.delitem(sys.modules, name)
    yield


def test_sampler_qnn_shot_contract_is_reported_without_qiskit_ml(
    without_qiskit_machine_learning,
) -> None:
    """This is the assertion that broke the core CI job.

    ``build_sampler_qnn`` imported ``qiskit_machine_learning`` *before*
    checking the shot contract, so on a ``pip install -e '.[dev]'` machine the
    test asking for a ConfigurationError received a MissingDependencyError and
    ``pytest tests`` -- the exact command in the ``core`` job -- failed.
    """

    from qforge_ai.models.qiskit_qnn import build_sampler_qnn

    with pytest.raises(ConfigurationError, match="shot"):
        build_sampler_qnn(
            _model_ir("probability_measurement"),
            backend=BackendSpec(engine=Engine.QISKIT),
        )


def test_every_qnn_configuration_check_precedes_the_optional_import(
    without_qiskit_machine_learning,
) -> None:
    from qforge_ai.models.qiskit_qnn import build_estimator_qnn, build_sampler_qnn

    probability = _model_ir("probability_measurement")
    shots = BackendSpec(engine=Engine.QISKIT, shots=64)

    with pytest.raises(ConfigurationError):
        build_sampler_qnn(probability, backend=shots, interpret="not callable")
    with pytest.raises(ConfigurationError):
        build_sampler_qnn(probability, backend=shots, interpret=lambda value: value % 2)
    with pytest.raises(ConfigurationError):
        build_estimator_qnn(probability, backend=BackendSpec(engine=Engine.QISKIT))


# --------------------------------------------------------------------------
# 8. the catalog must describe composition, not only components
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key,expected_absent",
    [
        ("basis_encoding", "differentiable"),
        ("binary_encoding", "differentiable"),
        ("superposition_encoding", "differentiable"),
        ("superposition_encoding", "data_encoding"),
        ("amplitude_encoding", "symbolic_parameters"),
        ("state_preparation_encoding", "symbolic_parameters"),
    ],
)
def test_encoding_capabilities_are_declared_per_entry(key, expected_absent) -> None:
    """The whole family claimed ``differentiable``; several entries are not."""

    from qforge_ai.catalog import catalog

    assert expected_absent not in catalog.get(key).capabilities


@pytest.mark.parametrize("key", ["angle_encoding", "zz_feature_map", "iqp_feature_map"])
def test_rotation_encodings_keep_every_capability(key) -> None:
    from qforge_ai.catalog import catalog

    capabilities = catalog.get(key).capabilities
    assert {"data_encoding", "differentiable", "continuous_input", "symbolic_parameters"} <= (
        capabilities
    )


@pytest.mark.parametrize(
    "encoding,expected",
    [
        ("angle_encoding", {"symbolic_parameters", "continuous_input", "differentiable"}),
        ("basis_encoding", set()),
        ("amplitude_encoding", {"continuous_input", "differentiable"}),
    ],
)
def test_declared_capabilities_match_the_runtime_profile(encoding, expected) -> None:
    """The catalog's claim and the composition preflight are one thing."""

    from qforge_ai.catalog import catalog
    from qforge_ai.composition import encoding_profile

    qubits = 2
    encoding_spec = (
        EncodingSpec(name=encoding, feature_dimension=2**qubits, normalize=True)
        if encoding in {"amplitude_encoding", "state_preparation_encoding"}
        else EncodingSpec(name=encoding, feature_dimension=qubits)
    )
    ir = CircuitFactory().build(
        CircuitSpec(num_qubits=qubits, encoding=encoding_spec, ansatz=AnsatzSpec("real_amplitudes"))
    )
    profile = encoding_profile(ir)
    observed = {
        name
        for name, value in (
            ("symbolic_parameters", profile.symbolic_parameters),
            ("continuous_input", profile.continuous_input),
            ("differentiable", profile.differentiable),
        )
        if value
    }
    assert observed == expected
    declared = catalog.get(encoding).capabilities
    assert observed <= declared | {"symbolic_parameters"}
    for name in ("differentiable", "continuous_input", "symbolic_parameters"):
        assert (name in declared) == (name in observed), name


@requires_qml
def test_basis_encoding_in_a_qiskit_qnn_is_refused_by_name() -> None:
    """It used to fail deep inside the QNN with "requires a concrete input"."""

    from qforge_ai.models import ModelFactory

    ir = CircuitFactory().build(
        CircuitSpec(
            num_qubits=2,
            encoding=EncodingSpec(name="basis_encoding", feature_dimension=2),
            ansatz=AnsatzSpec(name="real_amplitudes", layers=1),
        )
    )
    with pytest.raises(ConfigurationError, match="BASIS"):
        ModelFactory().build(
            ir,
            ModelSpec(mode=ModelMode.QUANTUM_ONLY, template="quantum_classifier"),
            BackendSpec(engine=Engine.QISKIT, device="statevector"),
        )


@requires_pennylane
@requires_torch
def test_basis_encoding_behind_a_classical_encoder_is_refused() -> None:
    """The encoder emits ``tanh(...)*pi``; basis encoding takes bits."""

    from qforge_ai.models import ModelFactory

    ir = CircuitFactory().build(
        CircuitSpec(
            num_qubits=2,
            encoding=EncodingSpec(name="basis_encoding", feature_dimension=2),
            ansatz=AnsatzSpec(name="real_amplitudes", layers=1),
        )
    )
    with pytest.raises(ConfigurationError, match="0/1"):
        ModelFactory().build(
            ir,
            ModelSpec(mode=ModelMode.HYBRID, input_dimension=2, output_dimension=2),
            BackendSpec(engine=Engine.PENNYLANE, device="default.qubit"),
        )


@requires_pennylane
def test_a_model_whose_circuit_ignores_its_input_is_refused() -> None:
    """Superposition encoding loads no data: the model was constant."""

    from qforge_ai.models import ModelFactory

    ir = CircuitFactory().build(
        CircuitSpec(
            num_qubits=2,
            encoding=EncodingSpec(name="superposition_encoding"),
            ansatz=AnsatzSpec(name="real_amplitudes", layers=1),
        )
    )
    with pytest.raises(ConfigurationError, match="input parameter"):
        ModelFactory().build(
            ir,
            ModelSpec(mode=ModelMode.HYBRID, input_dimension=2, output_dimension=2),
            BackendSpec(engine=Engine.PENNYLANE, device="default.qubit"),
        )


@requires_sklearn
@requires_pennylane
def test_trainable_feature_map_is_refused_as_a_fixed_kernel() -> None:
    from qforge_ai.models import ModelFactory

    ir = CircuitFactory().build(
        CircuitSpec(
            num_qubits=2,
            encoding=EncodingSpec(name="trainable_feature_map", feature_dimension=2),
            ansatz=AnsatzSpec(name="real_amplitudes", layers=1),
        )
    )
    with pytest.raises(ConfigurationError, match="fixed"):
        ModelFactory().build(
            ir,
            ModelSpec(mode=ModelMode.QUANTUM_KERNEL, template="quantum_kernel_model"),
            BackendSpec(engine=Engine.PENNYLANE, device="default.qubit"),
        )


@requires_both
@requires_torch
@requires_sklearn
@requires_qml
def test_every_encoding_model_engine_composition_behaves_as_declared() -> None:
    """The sweep the catalog was missing.

    Each component was previously verified against a baseline chosen to suit
    it, so two individually-true claims could still combine into a failure.
    This runs the cross product and requires that a composition the catalog
    supports actually runs, and one it does not is refused by name.
    """

    from qforge_ai.verification import verify_compositions

    results = verify_compositions()
    assert results
    failures = [item for item in results if item.failed]
    assert not failures, "composition claims are not backed by execution:\n" + "\n".join(
        f"  {item.engine} {item.key}: {item.detail}" for item in failures
    )
    assert any(item.ok and "refused" in item.detail for item in results)
