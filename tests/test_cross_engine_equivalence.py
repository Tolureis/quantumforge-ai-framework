"""The two engines must compute the same numbers from the same IR.

This is the guarantee the framework exists to provide, and the 1.2.2 report
showed it did not hold: the same amplitude vector prepared orthogonal states,
the same Hermitian observable read its qubits in opposite orders, and the same
noise parameter meant two different physical channels.

Each test here compares Qiskit and PennyLane on one shared ``CircuitIR`` and
asserts agreement to floating-point tolerance -- not "similar", identical.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from qforge_ai.ir import CircuitIR, Measurement, Observable
from qforge_ai.specs import NoiseSpec

requires_both = pytest.mark.skipif(
    importlib.util.find_spec("qiskit") is None
    or importlib.util.find_spec("qiskit_aer") is None
    or importlib.util.find_spec("pennylane") is None,
    reason="qiskit, qiskit-aer and pennylane required",
)


def _engines(noise: NoiseSpec | None = None):
    from qforge_ai.backends import PennyLaneAdapter, QiskitAdapter

    return QiskitAdapter(noise=noise), PennyLaneAdapter(noise=noise)


def _both(ir: CircuitIR, noise: NoiseSpec | None = None):
    qiskit_adapter, pennylane_adapter = _engines(noise)
    left = np.asarray(qiskit_adapter.run(ir).values)
    right = np.asarray(pennylane_adapter.run(ir).values)
    return left, right


# ---------------------------------------------------------------------- 1
@requires_both
@pytest.mark.parametrize(
    "amplitudes",
    [
        (0.182574, 0.365148, 0.547723, 0.730297),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.5, 0.5, 0.5, 0.5),
    ],
)
def test_state_preparation_prepares_the_same_state(amplitudes) -> None:
    """``[0,1,0,0]`` used to give |10> on Qiskit and |01> on PennyLane.

    The IR amplitude vector is wire-major, but ``QuantumCircuit.initialize``
    reads its qubit list little-endian, and the wire list was passed straight
    through.  The two engines prepared orthogonal states -- fidelity 0.0 --
    from one experiment definition.
    """

    ir = CircuitIR(num_qubits=2, name="amplitude")
    ir.add("STATE_PREP", (0, 1), *amplitudes, normalize=True)
    ir.measure(Measurement(kind="statevector", wires=(0, 1)))
    left, right = _both(ir)
    assert np.allclose(left, right, atol=1e-9)
    fidelity = abs(np.vdot(left, right)) ** 2
    assert fidelity == pytest.approx(1.0, abs=1e-9)
    # And the prepared state is the wire-major vector that was asked for.
    expected = np.asarray(amplitudes, dtype=float)
    expected = expected / np.linalg.norm(expected)
    assert np.allclose(np.abs(left), expected, atol=1e-9)


# ---------------------------------------------------------------------- 2
@requires_both
@pytest.mark.parametrize(
    "flipped,expected",
    [((), 0.0), ((0,), 2.0), ((1,), 1.0), ((0, 1), 3.0)],
)
def test_multi_qubit_hermitian_observable_reads_wires_in_ir_order(
    flipped, expected
) -> None:
    """``diag(0,1,2,3)`` used to read 1.0 on Qiskit against 2.0 on PennyLane.

    An IR observable matrix is wire-major, so index bit ``j`` belongs to
    ``wires[j]``.  Qiskit's ``qargs`` are little-endian and the wire tuple was
    passed unreversed, so the two qubits were swapped.
    """

    matrix = tuple(
        tuple(complex(value) for value in row) for row in np.diag([0.0, 1.0, 2.0, 3.0])
    )
    ir = CircuitIR(num_qubits=2, name="hermitian")
    for wire in flipped:
        ir.add("X", wire)
    ir.measure(
        Measurement(
            kind="expectation",
            wires=(0, 1),
            observables=(Observable(name="HERMITIAN", wires=(0, 1), matrix=matrix),),
        )
    )
    left, right = _both(ir)
    assert np.allclose(left, right, atol=1e-9)
    assert float(left[0]) == pytest.approx(expected, abs=1e-9)


@requires_both
def test_estimator_qnn_observable_uses_the_same_wire_order() -> None:
    """The full-system embedding must agree with direct execution."""

    pytest.importorskip("qiskit_machine_learning")
    from qiskit.quantum_info import Operator, Statevector

    from qforge_ai.backends.qiskit_adapter import _full_qiskit_observable

    matrix = tuple(
        tuple(complex(value) for value in row) for row in np.diag([0.0, 1.0, 2.0, 3.0])
    )
    observable = Observable(name="HERMITIAN", wires=(0, 1), matrix=matrix)
    embedded = _full_qiskit_observable(
        __import__("qiskit.quantum_info", fromlist=["x"]), observable, 2
    )
    # |w0=1, w1=0> is wire-major index 2, so the embedded operator must read 2.
    state = Statevector.from_label("01")  # Qiskit label is little-endian: q0=1
    value = float(np.real(state.expectation_value(Operator(embedded))))
    assert value == pytest.approx(2.0, abs=1e-9)


# ---------------------------------------------------------------------- 3
@requires_both
@pytest.mark.parametrize(
    "noise",
    [
        NoiseSpec(name="bit_flip_noise", probability=0.23),
        NoiseSpec(name="phase_flip_noise", probability=0.23),
        NoiseSpec(name="bit_phase_flip_noise", probability=0.23),
        NoiseSpec(name="depolarizing_noise", probability=0.31),
        NoiseSpec(name="amplitude_damping", probability=0.37),
        NoiseSpec(name="phase_damping", probability=0.37),
        NoiseSpec(name="thermal_relaxation", t1=60.0, t2=40.0, gate_time=12.0),
        NoiseSpec(name="thermal_relaxation", t1=60.0, t2=110.0, gate_time=12.0),
    ],
)
def test_every_noise_channel_is_numerically_identical_across_engines(noise) -> None:
    """One ``NoiseSpec`` must mean one physical channel.

    ``depolarizing(p=0.3)`` used to give -0.7 on Qiskit and -0.6 on PennyLane
    because Qiskit's parameter is the probability of full depolarization while
    PennyLane's is the total Pauli-error probability.  Both engines now
    execute the same canonical Kraus operators from ``qforge_ai.noise``.
    """

    ir = CircuitIR(num_qubits=2, name="channels")
    ir.add("H", 0)
    ir.add("RY", 1, 0.7)
    ir.add("CX", (0, 1))
    ir.add("RZ", 0, 0.3)
    ir.add("CRY", (1, 0), 0.9)
    ir.measure(
        Measurement(
            kind="expectation",
            wires=(0, 1),
            observables=(
                Observable("Z", (0,)),
                Observable("X", (1,)),
                Observable("Y", (0,)),
            ),
        )
    )
    left, right = _both(ir, noise)
    assert np.allclose(left, right, atol=1e-9), f"{noise.name}: {left} != {right}"


@requires_both
def test_depolarizing_uses_the_declared_full_depolarization_convention() -> None:
    ir = CircuitIR(num_qubits=1, name="depolarizing")
    ir.add("X", 0)
    ir.measure(
        Measurement(kind="expectation", wires=(0,), observables=(Observable("Z", (0,)),))
    )
    left, right = _both(ir, NoiseSpec(name="depolarizing_noise", probability=0.3))
    # rho -> (1-p) rho + p I/2, so <Z> = -(1-p) = -0.7 on both engines.
    assert float(left[0]) == pytest.approx(-0.7, abs=1e-9)
    assert float(right[0]) == pytest.approx(-0.7, abs=1e-9)


@requires_both
@pytest.mark.parametrize(
    "name,params,expected",
    [
        ("BASIS", (0.0,), 1.0),
        ("BASIS", (1.0,), -1.0),
        ("STATE_PREP", (1.0, 0.0), 1.0),
        ("STATE_PREP", (0.0, 1.0), -1.0),
    ],
)
def test_state_preparation_is_ideal_on_both_engines(name, params, expected) -> None:
    """Preparation defines the initial state; it is not a noisy gate.

    ``bit_flip(p=1)`` over ``BASIS(0)`` used to read +1 on Qiskit (which
    emitted no gate at all) and -1 on PennyLane (which applied a channel after
    ``BasisState``).  Preparation now carries no gate noise on either engine,
    which is the only choice the two can agree on: Qiskit's ``initialize``
    decomposes into a state-dependent number of basis gates.
    """

    ir = CircuitIR(num_qubits=1, name="preparation")
    wires = 0 if name == "BASIS" else (0,)
    ir.add(name, wires, *params)
    ir.measure(
        Measurement(kind="expectation", wires=(0,), observables=(Observable("Z", (0,)),))
    )
    left, right = _both(ir, NoiseSpec(name="bit_flip_noise", probability=1.0))
    assert float(left[0]) == pytest.approx(expected, abs=1e-9)
    assert float(right[0]) == pytest.approx(expected, abs=1e-9)


@requires_both
def test_density_matrices_agree_across_engines() -> None:
    ir = CircuitIR(num_qubits=2, name="density")
    ir.add("H", 0)
    ir.add("CX", (0, 1))
    ir.measure(Measurement(kind="density_matrix", wires=(0, 1)))
    left, right = _both(ir, NoiseSpec(name="depolarizing_noise", probability=0.2))
    assert np.allclose(left, right, atol=1e-9)


@requires_both
def test_analytic_probability_vectors_agree_across_engines() -> None:
    ir = CircuitIR(num_qubits=2, name="probability")
    ir.add("RY", 0, 0.9)
    ir.add("CX", (0, 1))
    ir.add("RX", 1, 0.4)
    ir.measure(Measurement(kind="probability", wires=(0, 1)))
    left, right = _both(ir)
    assert np.allclose(left, right, atol=1e-9)


def _kernel_ir():
    from qforge_ai.circuits import CircuitFactory
    from qforge_ai.specs import AnsatzSpec, CircuitSpec, EncodingSpec, MeasurementSpec

    return CircuitFactory().build(
        CircuitSpec(
            num_qubits=2,
            encoding=EncodingSpec(name="zz_feature_map", feature_dimension=2),
            ansatz=AnsatzSpec(name="real_amplitudes", layers=1),
            measurement=MeasurementSpec(name="expectation"),
        )
    )


def _kernel_pair(*, fidelity, noise=None, shots=None, seed=13):
    """Build the same kernel definition on both engines."""

    from qforge_ai.models.kernel import build_quantum_kernel
    from qforge_ai.specs import BackendSpec, Engine

    noise = noise or NoiseSpec()
    ir = _kernel_ir()
    noisy = noise.name != "none"
    left, _ = build_quantum_kernel(
        ir,
        backend=BackendSpec(
            engine=Engine.QISKIT,
            device="aer" if (noisy or shots) else "statevector",
            shots=shots,
            noise=noise,
            seed=seed,
        ),
        fidelity=fidelity,
    )
    right, _ = build_quantum_kernel(
        ir,
        backend=BackendSpec(
            engine=Engine.PENNYLANE,
            device="default.mixed" if noisy else "default.qubit",
            shots=shots,
            noise=noise,
            seed=seed,
        ),
        fidelity=fidelity,
    )
    return left, right


_KERNEL_SAMPLES = np.array([[0.1, 0.2], [1.3, 1.4], [0.7, -0.5], [-0.9, 0.6]])


@requires_both
def test_analytic_kernels_agree_across_engines() -> None:
    """The two fidelity kernels must produce one Gram matrix."""

    pytest.importorskip("qiskit_machine_learning")
    pytest.importorskip("sklearn")
    left, right = _kernel_pair(fidelity="statevector")
    assert np.allclose(left(_KERNEL_SAMPLES), right(_KERNEL_SAMPLES), atol=1e-8)
    assert np.allclose(np.diag(right(_KERNEL_SAMPLES)), 1.0, atol=1e-8)


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
def test_mixed_state_kernels_are_identical_across_engines(noise) -> None:
    """``Tr[rho(x) rho(y)]`` is exact on both engines, not merely close.

    Qiskit used to refuse this kernel outright, so the same ``auto``
    definition -- a noise channel and no shots -- ran on PennyLane and raised
    ``CapabilityError`` on Qiskit.
    """

    pytest.importorskip("qiskit_machine_learning")
    left, right = _kernel_pair(fidelity="hilbert_schmidt", noise=noise)
    assert np.allclose(left(_KERNEL_SAMPLES), right(_KERNEL_SAMPLES), atol=1e-9)


@requires_both
@pytest.mark.parametrize(
    "noise",
    [
        NoiseSpec(name="none"),
        NoiseSpec(name="depolarizing", probability=0.2),
        NoiseSpec(name="bit_flip", probability=0.1),
        NoiseSpec(name="readout_error", readout_probability=0.4),
    ],
)
def test_sampled_kernels_agree_across_engines(noise) -> None:
    """One compute-uncompute experiment, sampled by two engines.

    The two engines used to differ *systematically*, not statistically: Qiskit
    wrote 1.0 into the self-kernel diagonal instead of measuring it, carried
    noise through a transpilation-dependent Aer ``NoiseModel``, and applied a
    readout channel PennyLane dropped entirely.  40 000 shots put the 3-sigma
    band of one entry near 0.008.
    """

    pytest.importorskip("qiskit_machine_learning")
    left, right = _kernel_pair(fidelity="sampler", noise=noise, shots=40_000)
    assert np.allclose(left(_KERNEL_SAMPLES), right(_KERNEL_SAMPLES), atol=0.02)


@requires_both
def test_the_kernel_feature_width_contract_is_the_same_on_both_engines() -> None:
    """Both engines refuse data the feature map does not read."""

    pytest.importorskip("qiskit_machine_learning")
    from qforge_ai.errors import ConfigurationError

    left, right = _kernel_pair(fidelity="statevector")
    wide = np.hstack([_KERNEL_SAMPLES, _KERNEL_SAMPLES[:, :1]])
    for kernel in (left, right):
        assert kernel.feature_dimension == 2
        with pytest.raises(ConfigurationError):
            kernel(wide)
