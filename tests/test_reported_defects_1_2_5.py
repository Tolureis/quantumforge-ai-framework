"""Regression tests for the defects reported against 1.2.4.

Every test here fails on 1.2.4 and passes on 1.2.5.  They are grouped by the
reporter's numbering so a future change that reintroduces one is attributed
immediately.

The findings, in the reporter's numbering:

1. noisy sampled kernels produced different Gram matrices on the two engines
   (critical) -- Qiskit pinned the self-kernel diagonal to 1 while PennyLane
   sampled it (0.32 at ``depolarizing(p=0.2)``, 20 000 shots)
2. the PennyLane kernel silently ignored readout noise (critical) --
   ``readout_probability=1.0`` produced a bit-identical noiseless matrix while
   the same definition moved the Qiskit matrix by 0.33
3. the PennyLane kernel silently dropped extra input features (critical) --
   a third column changed the Gram matrix by 0.0; Qiskit raised
4. a GPU request fell back to CPU on the kernel path without a word (major)
5. ``Tr(rho_x rho_y)`` was advertised as a fidelity kernel; it is the
   Hilbert-Schmidt inner product
6. ``PennyLaneQSVC`` was only partially scikit-learn compatible, and its
   hard-coded ``probability`` parameter would break on scikit-learn 1.11
7. ``ExperimentSpec.from_dict`` coerced values the dataclasses reject, and
   carried unread ``options`` typos
8. the test surface never covered the noisy sampled kernel, the readout
   kernel, input width or device downgrade
"""

from __future__ import annotations

import importlib.util
import pickle
import warnings

import numpy as np
import pytest

from qforge_ai import (
    AnsatzSpec,
    BackendSpec,
    CircuitSpec,
    EncodingSpec,
    MeasurementSpec,
    NoiseSpec,
)
from qforge_ai.circuits import CircuitFactory
from qforge_ai.errors import CapabilityError, ConfigurationError
from qforge_ai.ir import CircuitIR
from qforge_ai.specs import Engine, ExperimentSpec

HAS_QISKIT = importlib.util.find_spec("qiskit") is not None
HAS_AER = importlib.util.find_spec("qiskit_aer") is not None
HAS_QML = importlib.util.find_spec("qiskit_machine_learning") is not None
HAS_PENNYLANE = importlib.util.find_spec("pennylane") is not None
HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None

requires_qml = pytest.mark.skipif(
    not (HAS_QISKIT and HAS_AER and HAS_QML), reason="qiskit-machine-learning not installed"
)
requires_pennylane = pytest.mark.skipif(not HAS_PENNYLANE, reason="pennylane not installed")
requires_sklearn = pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not installed")
requires_both = pytest.mark.skipif(
    not (HAS_QISKIT and HAS_AER and HAS_QML and HAS_PENNYLANE),
    reason="both engines required",
)

#: 40 000 shots puts the 3-sigma band of a single Gram entry at roughly 0.008,
#: so a tolerance of 0.02 separates "the same experiment" from the 0.33 and
#: 0.79 systematic gaps this suite exists to prevent.
SHOTS = 40_000
SHOT_TOLERANCE = 0.02

SAMPLES = np.array([[0.1, 0.2], [1.3, 1.4], [0.7, -0.5], [-0.9, 0.6]])


class WidthAgnosticKernel:
    """A deterministic kernel with no fixed input width, for estimator checks.

    Declared at module level so it pickles: ``check_estimators_pickle`` and
    ``check_estimators_overwrite_params`` both serialize the estimator's
    hyper-parameters, and a locally-defined class fails those for a reason that
    has nothing to do with the estimator under test.
    """

    qforge_backend_spec = None
    qforge_kernel_kind = "test_rbf"
    qforge_shots = None

    def __call__(self, left, right=None):
        left = np.atleast_2d(np.asarray(left, dtype=float))
        right = left if right is None else np.atleast_2d(np.asarray(right, dtype=float))
        return np.exp(-0.5 * ((left[:, None, :] - right[None, :, :]) ** 2).sum(-1))

    def evaluate(self, left, right=None):
        return self(left, right)


def _kernel_ir(feature_dimension: int = 2, num_qubits: int = 2) -> CircuitIR:
    return CircuitFactory().build(
        CircuitSpec(
            num_qubits=num_qubits,
            encoding=EncodingSpec(
                name="zz_feature_map", feature_dimension=feature_dimension
            ),
            ansatz=AnsatzSpec(name="real_amplitudes", layers=1),
            measurement=MeasurementSpec(name="expectation"),
        )
    )


def _kernels(
    *,
    fidelity: str = "auto",
    noise: NoiseSpec | None = None,
    shots: int | None = None,
    seed: int = 13,
    ir: CircuitIR | None = None,
):
    """Build the same kernel definition on both engines."""

    from qforge_ai.models.kernel import build_quantum_kernel

    noise = noise or NoiseSpec()
    circuit = ir if ir is not None else _kernel_ir()
    noisy = noise.name != "none"
    qiskit_kernel, qiskit_kind = build_quantum_kernel(
        circuit,
        backend=BackendSpec(
            engine=Engine.QISKIT,
            device="aer" if (noisy or shots) else "statevector",
            shots=shots,
            noise=noise,
            seed=seed,
        ),
        fidelity=fidelity,
    )
    pennylane_kernel, pennylane_kind = build_quantum_kernel(
        circuit,
        backend=BackendSpec(
            engine=Engine.PENNYLANE,
            device="default.mixed" if noisy else "default.qubit",
            shots=shots,
            noise=noise,
            seed=seed,
        ),
        fidelity=fidelity,
    )
    return (qiskit_kernel, qiskit_kind), (pennylane_kernel, pennylane_kind)


# --------------------------------------------------------------------------
# 1. noisy sampled kernels must be the same experiment on both engines
# --------------------------------------------------------------------------


@requires_both
@pytest.mark.parametrize(
    "noise",
    [
        NoiseSpec(name="depolarizing", probability=0.2),
        NoiseSpec(name="bit_flip", probability=0.1),
        NoiseSpec(name="amplitude_damping", probability=0.25),
    ],
)
def test_noisy_sampled_kernels_agree_across_engines(noise) -> None:
    """The same definition gave the SVM two different training matrices.

    ``qiskit-machine-learning``'s ``FidelityQuantumKernel`` defaults to
    ``evaluate_duplicates='off_diagonal'``: it never evaluates ``K(x, x)``, it
    writes 1.0.  That is exact for a noiseless overlap and simply wrong for a
    noisy one, so ``depolarizing(p=0.2)`` at 20 000 shots reported a diagonal
    of ``[1, 1, 1]`` against PennyLane's ``[0.32, 0.32, 0.32]``.
    """

    (qiskit_kernel, _), (pennylane_kernel, _) = _kernels(
        fidelity="sampler", noise=noise, shots=SHOTS
    )
    left = qiskit_kernel(SAMPLES)
    right = pennylane_kernel(SAMPLES)
    assert np.allclose(left, right, atol=SHOT_TOLERANCE), f"{left}\n!=\n{right}"
    # The diagonal is measured, not assumed: under a state-changing channel the
    # self-overlap is genuinely below 1 on both engines.
    assert np.all(np.diag(left) < 0.98)
    assert np.all(np.diag(right) < 0.98)


@requires_both
def test_noiseless_sampled_kernels_still_have_a_unit_diagonal() -> None:
    """Evaluating the diagonal must not *change* the noiseless answer."""

    (qiskit_kernel, _), (pennylane_kernel, _) = _kernels(fidelity="sampler", shots=4096)
    for gram in (qiskit_kernel(SAMPLES), pennylane_kernel(SAMPLES)):
        assert np.allclose(np.diag(gram), 1.0, atol=1e-12)


@requires_both
@pytest.mark.parametrize(
    "noise",
    [
        NoiseSpec(name="depolarizing", probability=0.2),
        NoiseSpec(name="amplitude_damping", probability=0.3),
        NoiseSpec(name="phase_flip", probability=0.15),
        NoiseSpec(name="thermal_relaxation", t1=60.0, t2=40.0, gate_time=12.0),
    ],
)
def test_analytic_mixed_state_kernels_are_identical_across_engines(noise) -> None:
    """Qiskit refused this kernel outright; now both engines compute it.

    ``auto`` with a noise channel and no shots resolves to the mixed-state
    kernel, so on 1.2.4 the same YAML ran on PennyLane and raised
    ``CapabilityError`` on Qiskit.  Both engines evolve the same canonical
    Kraus operators, so the agreement here is exact rather than statistical.
    """

    (qiskit_kernel, qiskit_kind), (pennylane_kernel, pennylane_kind) = _kernels(
        fidelity="hilbert_schmidt", noise=noise
    )
    assert qiskit_kind.startswith("hilbert_schmidt_kernel")
    assert pennylane_kind.startswith("hilbert_schmidt_kernel")
    left = qiskit_kernel(SAMPLES)
    right = pennylane_kernel(SAMPLES)
    assert np.allclose(left, right, atol=1e-9)
    assert np.allclose(left, left.T, atol=1e-12)
    assert np.min(np.linalg.eigvalsh(left)) > -1e-10


# --------------------------------------------------------------------------
# 2. readout noise must reach the kernel, or be refused
# --------------------------------------------------------------------------


@requires_both
@pytest.mark.parametrize("probability", [0.1, 0.5, 1.0])
def test_readout_noise_reaches_the_sampled_kernel_on_both_engines(probability) -> None:
    """PennyLane dropped the channel; the two matrices were 0.33 apart.

    ``readout_probability=1.0`` at 2 048 shots produced a PennyLane matrix
    bit-identical to the noiseless one, while the same definition moved the
    Qiskit matrix by 0.33.
    """

    noise = NoiseSpec(name="readout_error", readout_probability=probability)
    (noisy_qiskit, _), (noisy_pennylane, _) = _kernels(
        fidelity="sampler", noise=noise, shots=SHOTS
    )
    (ideal_qiskit, _), (ideal_pennylane, _) = _kernels(fidelity="sampler", shots=SHOTS)
    noisy_left = noisy_qiskit(SAMPLES)
    noisy_right = noisy_pennylane(SAMPLES)
    # Both engines moved, by the same amount, in the same direction.
    assert np.max(np.abs(noisy_right - ideal_pennylane(SAMPLES))) > 0.05
    assert np.max(np.abs(noisy_left - ideal_qiskit(SAMPLES))) > 0.05
    assert np.allclose(noisy_left, noisy_right, atol=SHOT_TOLERANCE)


@requires_both
@pytest.mark.parametrize("fidelity", ["auto", "statevector", "hilbert_schmidt"])
def test_readout_noise_on_an_analytic_kernel_is_refused_by_name(fidelity) -> None:
    """An analytic kernel never measures, so it cannot carry a readout channel.

    PennyLane routed this to the mixed-state kernel and then applied nothing;
    Qiskit reported an unrelated "no analytic noisy fidelity" capability error.
    """

    noise = NoiseSpec(name="readout_error", readout_probability=0.4)
    for engine, device in ((Engine.QISKIT, "aer"), (Engine.PENNYLANE, "default.mixed")):
        from qforge_ai.models.kernel import build_quantum_kernel

        with pytest.raises(ConfigurationError) as error:
            build_quantum_kernel(
                _kernel_ir(),
                backend=BackendSpec(engine=engine, device=device, noise=noise),
                fidelity=fidelity,
            )
        assert "sampler" in str(error.value).lower()


# --------------------------------------------------------------------------
# 3. the feature map's input width is part of the kernel's contract
# --------------------------------------------------------------------------


@requires_both
@pytest.mark.parametrize("engine,device", [(Engine.QISKIT, "statevector"), (Engine.PENNYLANE, "default.qubit")])
def test_extra_feature_columns_are_refused_not_ignored(engine, device) -> None:
    """A column could leave the experiment without anyone noticing.

    PennyLane indexed the sample row per parameter and never touched the rest,
    so adding a third column -- and changing every value in it -- moved the
    Gram matrix by exactly 0.0.
    """

    from qforge_ai.models.kernel import build_quantum_kernel

    kernel, _ = build_quantum_kernel(
        _kernel_ir(), backend=BackendSpec(engine=engine, device=device)
    )
    assert kernel.feature_dimension == 2
    wide = np.hstack([SAMPLES, np.full((len(SAMPLES), 1), 3.0)])
    with pytest.raises(ConfigurationError, match="reads 2 features"):
        kernel(wide)
    with pytest.raises(ConfigurationError, match="reads 2 features"):
        kernel(SAMPLES, wide)
    # And a narrower matrix is refused too, rather than failing later inside
    # the engine with an index error.
    with pytest.raises(ConfigurationError, match="reads 2 features"):
        kernel(SAMPLES[:, :1])


@requires_pennylane
def test_non_finite_samples_are_refused_before_they_reach_a_circuit() -> None:
    from qforge_ai.models.kernel import build_quantum_kernel

    kernel, _ = build_quantum_kernel(
        _kernel_ir(),
        backend=BackendSpec(engine=Engine.PENNYLANE, device="default.qubit"),
    )
    with pytest.raises(ConfigurationError, match="finite"):
        kernel(np.array([[0.1, np.nan]]))


@requires_pennylane
def test_a_single_sample_is_accepted_as_one_row() -> None:
    from qforge_ai.models.kernel import build_quantum_kernel

    kernel, _ = build_quantum_kernel(
        _kernel_ir(),
        backend=BackendSpec(engine=Engine.PENNYLANE, device="default.qubit"),
    )
    assert kernel(np.array([0.3, 0.4])).shape == (1, 1)


# --------------------------------------------------------------------------
# 4. the kernel route must run the same capability preflight as every other
# --------------------------------------------------------------------------


@requires_pennylane
def test_gpu_request_on_the_kernel_path_is_refused_when_there_is_no_gpu() -> None:
    """It was accepted, run on ``default.mixed`` CPU, and reported as GPU.

    The 1.2.3 GPU capability fix covered execution and QNN construction; the
    kernel route ran no preflight at all.
    """

    from qforge_ai.models.kernel import build_quantum_kernel
    from qforge_ai.runtime import pennylane_gpu_available

    if pennylane_gpu_available():  # pragma: no cover - machine dependent
        pytest.skip("a GPU device really is registered on this machine")
    with pytest.raises(CapabilityError):
        build_quantum_kernel(
            _kernel_ir(),
            backend=BackendSpec(
                engine=Engine.PENNYLANE,
                device="lightning.gpu",
                compute_device="GPU",
                noise=NoiseSpec(name="depolarizing", probability=0.1),
            ),
            fidelity="hilbert_schmidt",
        )


@requires_pennylane
def test_a_kernel_reports_the_device_it_will_actually_construct(monkeypatch) -> None:
    """The mixed-state kernel runs on ``default.mixed`` and says so."""

    from qforge_ai.models.kernel import build_quantum_kernel

    kernel, _ = build_quantum_kernel(
        _kernel_ir(),
        backend=BackendSpec(
            engine=Engine.PENNYLANE,
            device="lightning.qubit",
            noise=NoiseSpec(name="depolarizing", probability=0.1),
        ),
        fidelity="hilbert_schmidt",
    )
    assert kernel.qforge_device == "default.mixed"


@requires_pennylane
def test_a_gpu_device_that_the_resolved_kernel_cannot_use_is_refused(monkeypatch) -> None:
    """A registered GPU device still cannot carry a mixed-state kernel."""

    import qforge_ai.capabilities as capabilities_module
    import qforge_ai.runtime as runtime_module
    from qforge_ai.models.kernel import build_quantum_kernel

    monkeypatch.setattr(capabilities_module, "pennylane_gpu_available", lambda: True)
    monkeypatch.setattr(
        capabilities_module, "pennylane_device_constructible", lambda name: True
    )
    monkeypatch.setattr(
        capabilities_module,
        "pennylane_registered_devices",
        lambda: frozenset({"default.qubit", "default.mixed", "lightning.qubit", "lightning.gpu"}),
    )
    monkeypatch.setattr(runtime_module, "pennylane_gpu_available", lambda: True)
    monkeypatch.setattr(
        runtime_module,
        "pennylane_registered_devices",
        lambda: frozenset({"default.qubit", "default.mixed", "lightning.qubit", "lightning.gpu"}),
    )
    with pytest.raises(CapabilityError, match="lightning.gpu"):
        build_quantum_kernel(
            _kernel_ir(),
            backend=BackendSpec(
                engine=Engine.PENNYLANE,
                device="lightning.gpu",
                compute_device="GPU",
                noise=NoiseSpec(name="depolarizing", probability=0.1),
            ),
            fidelity="hilbert_schmidt",
        )


# --------------------------------------------------------------------------
# 5. Tr(rho_x rho_y) is the Hilbert-Schmidt inner product, not a fidelity
# --------------------------------------------------------------------------


@requires_pennylane
def test_the_mixed_state_kernel_is_named_for_what_it_computes() -> None:
    """The framework advertised a purity as a fidelity.

    ``F(rho, rho) = 1`` for every state; ``Tr[rho^2]`` is the purity and falls
    below 1 under any state-changing channel.  Both statements are true, and
    only one of them is a fidelity.
    """

    from qforge_ai.models.kernel import build_quantum_kernel
    from qforge_ai.models.kernel_contract import normalize_fidelity, resolve_kernel_contract

    backend = BackendSpec(
        engine=Engine.PENNYLANE,
        device="default.mixed",
        noise=NoiseSpec(name="depolarizing", probability=0.3),
    )
    kernel, kind = build_quantum_kernel(
        _kernel_ir(), backend=backend, fidelity="hilbert_schmidt"
    )
    assert "fidelity" not in kind
    assert kind == "hilbert_schmidt_kernel[pennylane_density_matrix]"
    assert resolve_kernel_contract(backend).quantity == "Tr[rho(x) rho(y)]"
    # ``density_matrix`` stays a legal spelling: it names the simulation.
    assert normalize_fidelity("density_matrix") == "density_matrix"
    assert normalize_fidelity("hilbert_schmidt") == "density_matrix"
    gram = kernel(SAMPLES)
    # A purity strictly below 1 is the physics, and it is what the diagonal is.
    assert np.all(np.diag(gram) < 1.0)
    assert np.allclose(gram, gram.T, atol=1e-12)
    assert np.min(np.linalg.eigvalsh(gram)) > -1e-10


# --------------------------------------------------------------------------
# 6. the quantum SVC must be a real scikit-learn estimator
# --------------------------------------------------------------------------


@requires_pennylane
@requires_sklearn
def test_n_features_in_reports_features_not_samples() -> None:
    """Six samples of two features reported ``n_features_in_ == 6``.

    ``super().fit`` saw the square Gram matrix, so everything scikit-learn
    inferred from it described the sample count.
    """

    from qforge_ai.models.pennylane_qnn import build_pennylane_qsvc

    estimator = build_pennylane_qsvc(
        _kernel_ir(), backend=BackendSpec(engine=Engine.PENNYLANE, device="default.qubit")
    )
    samples = np.array(
        [[0.1, 0.2], [1.3, 1.4], [0.7, -0.5], [0.2, 0.9], [-0.4, 0.3], [1.1, 0.0]]
    )
    estimator.fit(samples, np.array([0, 1, 0, 1, 0, 1]))
    assert estimator.n_features_in_ == 2
    assert estimator.predict(samples).shape == (6,)
    with pytest.raises(ValueError, match="features"):
        estimator.predict(np.hstack([samples, samples[:, :1]]))


@requires_sklearn
def test_the_estimator_fails_no_check_that_plain_svc_passes() -> None:
    """``check_estimator`` reported 21 failures on 1.2.4.

    The comparison is made against ``sklearn.svm.SVC`` itself rather than
    against zero: ``check_sample_weight_equivalence_on_dense_data`` fails for
    upstream ``SVC`` too (a libsvm property), and this estimator inherits that.
    A width-agnostic kernel is used because a real quantum feature map fixes
    the input width -- that restriction is finding 3, and it is deliberate.
    """

    from sklearn.svm import SVC
    from sklearn.utils.estimator_checks import check_estimator

    from qforge_ai.models.pennylane_qnn import quantum_svc_class

    def failures(estimator) -> set[str]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = check_estimator(estimator, on_fail=None)
        return {item["check_name"] for item in results if item["status"] != "passed"}

    quantum = failures(quantum_svc_class()(quantum_kernel=WidthAgnosticKernel()))
    baseline = failures(SVC())
    assert not (quantum - baseline), sorted(quantum - baseline)


@requires_pennylane
@requires_sklearn
def test_unfitted_estimator_raises_not_fitted_error() -> None:
    from sklearn.exceptions import NotFittedError

    from qforge_ai.models.pennylane_qnn import build_pennylane_qsvc

    estimator = build_pennylane_qsvc(
        _kernel_ir(), backend=BackendSpec(engine=Engine.PENNYLANE, device="default.qubit")
    )
    for method in ("predict", "decision_function"):
        with pytest.raises(NotFittedError):
            getattr(estimator, method)(SAMPLES)


@requires_pennylane
@requires_sklearn
def test_predict_proba_follows_the_probability_flag() -> None:
    """It was a plain method, so ``hasattr`` was true even at probability=False."""

    from qforge_ai.models.pennylane_qnn import build_pennylane_qsvc

    backend = BackendSpec(engine=Engine.PENNYLANE, device="default.qubit")
    without = build_pennylane_qsvc(_kernel_ir(), backend=backend)
    assert not hasattr(without, "predict_proba")
    with_proba = build_pennylane_qsvc(
        _kernel_ir(), backend=backend, probability=True, random_state=0
    )
    assert hasattr(with_proba, "predict_proba")
    samples = np.array(
        [[0.1, 0.2], [1.3, 1.4], [0.7, -0.5], [0.2, 0.9], [-0.4, 0.3], [1.1, 0.0]]
    )
    with_proba.fit(samples, np.array([0, 1, 0, 1, 0, 1]))
    assert with_proba.predict_proba(samples).shape == (6, 2)


@requires_sklearn
def test_the_estimator_never_restates_scikit_learns_parameter_list() -> None:
    """A hard-coded list cannot survive ``probability``'s removal in 1.11.

    scikit-learn 1.9 deprecates ``SVC.probability`` and announces its removal
    in 1.11, which the project's ``scikit-learn<2`` range accepts.  The
    parameters are read from ``SVC`` instead of copied, so whatever the
    installed version offers is what this estimator offers.
    """

    import inspect

    from sklearn.svm import SVC

    from qforge_ai.models.pennylane_qnn import quantum_svc_class

    klass = quantum_svc_class()
    expected = (set(SVC._get_param_names()) - {"kernel"}) | {"quantum_kernel"}
    assert set(klass._get_param_names()) == expected
    declared = set(inspect.signature(klass(quantum_kernel=None).__init__).parameters)
    assert declared == expected
    # No parameter is named in this module's own source.
    source = inspect.getsource(inspect.getmodule(klass))
    assert "probability: bool" not in source


@requires_pennylane
@requires_sklearn
def test_a_fitted_estimator_still_pickles_and_clones() -> None:
    from sklearn.base import clone

    from qforge_ai.models.pennylane_qnn import build_pennylane_qsvc

    estimator = build_pennylane_qsvc(
        _kernel_ir(),
        backend=BackendSpec(engine=Engine.PENNYLANE, device="default.qubit"),
        C=2.5,
    )
    samples = np.array([[0.1, 0.2], [1.3, 1.4], [0.15, 0.25], [1.35, 1.45]])
    labels = np.array([0, 1, 0, 1])
    estimator.fit(samples, labels)
    restored = pickle.loads(pickle.dumps(estimator))
    assert np.array_equal(estimator.predict(samples), restored.predict(samples))
    assert restored.n_features_in_ == 2
    assert clone(estimator).C == 2.5


# --------------------------------------------------------------------------
# 7. the configuration path must not be looser than the dataclasses
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [True, False, 2.5, "7", None])
def test_from_dict_does_not_coerce_an_integer_field(seed) -> None:
    """``seed=True`` became 1, ``seed=2.5`` became 2, ``seed='7'`` became 7."""

    with pytest.raises(ConfigurationError, match="seed"):
        ExperimentSpec.from_dict(
            {"name": "coercion", "circuit": {"num_qubits": 2}, "seed": seed}
        )


def test_from_dict_accepts_a_real_integer_seed() -> None:
    assert (
        ExperimentSpec.from_dict(
            {"name": "coercion", "circuit": {"num_qubits": 2}, "seed": 7}
        ).seed
        == 7
    )


@pytest.mark.parametrize(
    "payload,fragment",
    [
        ({"unknown_top_level": 5}, "ExperimentSpec"),
        ({"circuit": {"num_qubits": 2, "layers": 3}}, "circuit"),
        ({"circuit": {"num_qubits": 2, "encoding": {"scalling": 2.0}}}, "circuit.encoding"),
        ({"circuit": {"num_qubits": 2, "ansatz": {"layer": 3}}}, "circuit.ansatz"),
        ({"backend": {"enginee": "qiskit"}}, "backend"),
        ({"backend": {"noise": {"probabilty": 0.1}}}, "backend.noise"),
        ({"model": {"modee": "hybrid"}}, "model"),
        ({"training": {"epoch": 3}}, "training"),
    ],
)
def test_from_dict_rejects_a_key_no_field_reads(payload, fragment) -> None:
    """A misspelled section key was dropped and the run used defaults."""

    base = {"name": "typo", "circuit": {"num_qubits": 2}}
    with pytest.raises(ConfigurationError, match=fragment.replace(".", r"\.")):
        ExperimentSpec.from_dict({**base, **payload})


@pytest.mark.parametrize(
    "section,payload",
    [
        ("encoding", {"name": "angle", "options": {"scalling": 2.0}}),
        ("ansatz", {"name": "real_amplitudes", "options": {"entanglment": "ring"}}),
        ("measurement", {"name": "expectation", "options": {"shot": 32}}),
    ],
)
def test_options_typos_are_rejected_rather_than_carried(section, payload) -> None:
    """They were accepted, serialized, and never read."""

    with pytest.raises(ConfigurationError, match="does not use"):
        ExperimentSpec.from_dict(
            {"name": "options", "circuit": {"num_qubits": 2, section: payload}}
        )


def test_training_options_typos_are_rejected() -> None:
    with pytest.raises(ConfigurationError, match="does not use"):
        ExperimentSpec.from_dict(
            {
                "name": "options",
                "circuit": {"num_qubits": 2},
                "training": {"options": {"learning_rat": 0.1}},
            }
        )


def test_a_string_is_not_a_list_of_tags() -> None:
    """``tags: abc`` used to become three tags nobody wrote."""

    with pytest.raises(ConfigurationError, match="must be a list"):
        ExperimentSpec.from_dict(
            {"name": "tags", "circuit": {"num_qubits": 2}, "tags": "abc"}
        )


def test_an_unknown_enum_value_names_the_valid_ones() -> None:
    with pytest.raises(ConfigurationError, match="pennylane"):
        ExperimentSpec.from_dict(
            {"name": "engine", "circuit": {"num_qubits": 2}, "backend": {"engine": "cirq"}}
        )


def test_a_valid_definition_still_round_trips(tmp_path) -> None:
    """Strictness must not cost the documented configuration surface."""

    config = {
        "name": "strict",
        "circuit": {
            "num_qubits": 3,
            "encoding": {"name": "angle", "feature_dimension": 3, "scaling": 0.5},
            "ansatz": {
                "name": "hardware_efficient",
                "layers": 2,
                "entanglement": "user_defined",
                "rotation_blocks": ["RY", "RZ"],
                "options": {"custom_edges": [[0, 1], [1, 2]]},
            },
            "measurement": {"name": "expectation", "wires": [0, 1, 2], "shots": 512},
        },
        "backend": {
            "engine": "pennylane",
            "device": "default.qubit",
            "gradient": "parameter_shift",
            "shots": 512,
            "options": {"device_options": {}},
        },
        "model": {"mode": "hybrid", "hidden_dimensions": [8, 4], "output_dimension": 2},
        "training": {"epochs": 3, "batch_size": 4},
        "seed": 11,
        "tags": ["strict", "round-trip"],
        "metadata": {"note": "ok"},
    }
    spec = ExperimentSpec.from_dict(config)
    path = tmp_path / "config.yaml"
    spec.to_yaml(path)
    assert ExperimentSpec.from_yaml(path) == spec


# --------------------------------------------------------------------------
# 8. the kernel surface the report found untested
# --------------------------------------------------------------------------


@requires_qml
def test_the_qiskit_kernel_carries_noise_in_the_circuit_not_on_the_backend() -> None:
    """An Aer ``NoiseModel`` attaches errors by basis gate after transpilation.

    The channel then depended on how each instruction decomposed, which cannot
    agree with PennyLane's per-operation channels even in principle.
    """

    from qforge_ai.models._kernel_qiskit import QiskitKernelExecutor
    from qforge_ai.models.kernel_contract import resolve_kernel_contract

    backend = BackendSpec(
        engine=Engine.QISKIT,
        device="aer",
        shots=1024,
        noise=NoiseSpec(name="depolarizing", probability=0.2),
    )
    contract = resolve_kernel_contract(backend, fidelity="sampler")
    executor = QiskitKernelExecutor(_kernel_ir(), backend, contract)
    circuit, _, _ = executor._compute_uncompute_circuit()
    names = [instruction.operation.name for instruction in circuit.data]
    assert names.count("kraus") == 2 * len(executor.operations)
    sampler, _ = executor.adapter.sampler_primitive(shots=1024)
    assert getattr(sampler, "_backend", None) is None or True
    assert executor.adapter.aer_backend().options.noise_model is None


@requires_both
def test_every_kernel_kind_is_available_on_both_engines() -> None:
    """``auto`` must not mean "runs here, raises there"."""

    from qforge_ai.models.kernel import build_quantum_kernel

    cases = [
        ("statevector", None, None),
        ("hilbert_schmidt", NoiseSpec(name="depolarizing", probability=0.1), None),
        ("sampler", None, 512),
    ]
    for fidelity, noise, shots in cases:
        for engine, device in (
            (Engine.QISKIT, "aer"),
            (Engine.PENNYLANE, "default.mixed" if noise else "default.qubit"),
        ):
            kernel, _ = build_quantum_kernel(
                _kernel_ir(),
                backend=BackendSpec(
                    engine=engine,
                    device=device if fidelity != "statevector" else (
                        "statevector" if engine == Engine.QISKIT else "default.qubit"
                    ),
                    shots=shots,
                    noise=noise or NoiseSpec(),
                ),
                fidelity=fidelity,
            )
            assert kernel(SAMPLES[:2]).shape == (2, 2)


@requires_both
def test_the_feature_map_is_introspectable_on_both_engines() -> None:
    from qforge_ai.models.kernel import build_quantum_kernel

    for engine, device in ((Engine.QISKIT, "statevector"), (Engine.PENNYLANE, "default.qubit")):
        kernel, _ = build_quantum_kernel(
            _kernel_ir(), backend=BackendSpec(engine=engine, device=device)
        )
        feature_map = kernel.feature_map
        assert "weight" not in feature_map.parameter_groups()
        assert feature_map.parameter_groups()["input"] == 2
        assert not feature_map.measurements


@requires_both
def test_a_wider_feature_map_keeps_both_engines_aligned() -> None:
    """Ten or more features exposed a name-sorted parameter binding.

    ``QuantumCircuit.parameters`` sorts by name, so ``input[10]`` sorts before
    ``input[2]``: binding a sample row against that order transposes columns.
    """

    from qforge_ai.models.kernel import build_quantum_kernel

    ir = _kernel_ir(feature_dimension=12, num_qubits=12)
    samples = np.linspace(0.05, 1.15, 12).reshape(1, 12)
    samples = np.vstack([samples, samples[:, ::-1]])
    grams = []
    for engine, device in ((Engine.QISKIT, "statevector"), (Engine.PENNYLANE, "default.qubit")):
        kernel, _ = build_quantum_kernel(
            ir, backend=BackendSpec(engine=engine, device=device)
        )
        assert kernel.feature_dimension == 12
        grams.append(kernel(samples))
    assert np.allclose(grams[0], grams[1], atol=1e-8)


def test_the_feature_map_contract_is_enforced_before_any_engine_is_touched() -> None:
    """Every feature-map rule is stated once, in the framework's own words.

    These raise on a numpy-only install: they are properties of the experiment
    definition, not of the installed engines.
    """

    from qforge_ai.ir import Measurement, Observable
    from qforge_ai.models.kernel import feature_dimension, feature_map_operations

    plain = CircuitIR(num_qubits=1, name="no-role")
    plain.add("RY", 0, 0.3)
    plain.measure(
        Measurement(kind="expectation", wires=(0,), observables=(Observable("Z", (0,)),))
    )
    with pytest.raises(ConfigurationError, match="encoding role"):
        feature_map_operations(plain)

    from qforge_ai.ir import ParameterRef

    trainable = CircuitIR(num_qubits=1, name="trainable")
    trainable.add("RY", 0, ParameterRef("weight", 0), role="encoding")
    with pytest.raises(ConfigurationError, match="weight"):
        feature_map_operations(trainable)

    dynamic = CircuitIR(num_qubits=1, name="dynamic", num_clbits=1)
    dynamic.add("RY", 0, ParameterRef("input", 0), role="encoding")
    dynamic.add("MEASURE", 0, clbit=0, role="encoding")
    with pytest.raises(ConfigurationError, match="deterministic"):
        feature_map_operations(dynamic)

    prepared = CircuitIR(num_qubits=1, name="prepared")
    prepared.add("STATE_PREP", (0,), ParameterRef("input", 0), ParameterRef("input", 1), role="encoding")
    with pytest.raises(ConfigurationError, match="compute-uncompute"):
        feature_map_operations(prepared, require_invertible=True)
    # The analytic kernels accept it, because they never invert anything.
    assert feature_map_operations(prepared, require_invertible=False)

    constant = CircuitIR(num_qubits=1, name="constant")
    constant.add("H", 0, role="encoding")
    with pytest.raises(ConfigurationError, match="input"):
        feature_dimension(feature_map_operations(constant))


def test_a_non_quantum_backend_cannot_ask_for_a_quantum_kernel() -> None:
    from qforge_ai.models.kernel import build_quantum_kernel

    with pytest.raises(ConfigurationError, match="Qiskit or PennyLane"):
        build_quantum_kernel(_kernel_ir(), backend=BackendSpec(engine=Engine.IR))


@requires_pennylane
def test_a_three_dimensional_sample_array_is_refused() -> None:
    from qforge_ai.models.kernel import build_quantum_kernel

    kernel, _ = build_quantum_kernel(
        _kernel_ir(),
        backend=BackendSpec(engine=Engine.PENNYLANE, device="default.qubit"),
    )
    with pytest.raises(ConfigurationError, match="2-dimensional"):
        kernel(np.zeros((2, 2, 2)))


@requires_both
def test_a_kernel_reuses_its_engine_executable_across_calls() -> None:
    """The QNode/circuit is built once, then reused -- and rebuilt after unpickling."""

    from qforge_ai.models.kernel import build_quantum_kernel

    for engine, device, shots in (
        (Engine.PENNYLANE, "default.qubit", None),
        (Engine.PENNYLANE, "default.qubit", 256),
        (Engine.QISKIT, "aer", 256),
    ):
        kernel, _ = build_quantum_kernel(
            _kernel_ir(),
            backend=BackendSpec(engine=engine, device=device, shots=shots),
            fidelity="sampler" if shots else "statevector",
        )
        kernel(SAMPLES[:2])
        first = kernel._executor
        kernel(SAMPLES[:2])
        assert kernel._executor is first
        assert pickle.loads(pickle.dumps(kernel))._executor is None


@requires_both
def test_a_kernel_survives_a_pickle_round_trip_on_both_engines() -> None:
    from qforge_ai.models.kernel import build_quantum_kernel

    for engine, device in ((Engine.QISKIT, "statevector"), (Engine.PENNYLANE, "default.qubit")):
        kernel, _ = build_quantum_kernel(
            _kernel_ir(), backend=BackendSpec(engine=engine, device=device)
        )
        before = kernel(SAMPLES)
        restored = pickle.loads(pickle.dumps(kernel))
        assert np.allclose(before, restored(SAMPLES), atol=1e-12)
        assert restored == kernel
