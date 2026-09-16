"""Cross-engine noise with a single canonical physical definition.

Both engines used to build their own channel from the same ``NoiseSpec``
number, and the two builders did not agree on what that number *meant*.
``depolarizing`` with ``probability=0.3`` produced ``<Z> = -0.7`` on Qiskit and
``-0.6`` on PennyLane, because Qiskit's ``depolarizing_error`` takes the
probability of full depolarization while PennyLane's ``DepolarizingChannel``
takes the total probability of a Pauli error.  Neither was wrong on its own;
the framework simply never declared which one it meant.

This module now declares it.  Every channel is defined once as an explicit set
of Kraus operators in wire-major order, and both adapters insert *those same
operators* into the circuit at the same positions.  Cross-engine noisy results
are therefore identical up to floating point, not merely "similar".

Canonical parameter contract
----------------------------
``bit_flip`` / ``phase_flip`` / ``bit_phase_flip``
    ``p`` is the probability that the Pauli is applied.  On two wires the
    error is *correlated* (``P⊗P`` with probability ``p``), matching the
    two-qubit Pauli error a gate-level model attaches to an entangling gate.

``depolarizing``
    ``p`` is the probability that the state is replaced by the maximally
    mixed state -- the Nielsen & Chuang / Qiskit "depolarizing parameter".
    Equivalently the channel applies each of the ``4**n - 1`` non-identity
    Paulis with probability ``p / 4**n``.  A caller who thinks in terms of
    total Pauli-error probability wants ``p * 4**n / (4**n - 1)``.

``amplitude_damping`` / ``phase_damping``
    ``p`` is the damping parameter ``gamma`` / ``lambda``.  On two wires the
    channel acts independently on each wire.

``thermal_relaxation``
    Derived from ``t1``, ``t2`` and ``gate_time``: amplitude damping with
    ``gamma = 1 - exp(-tg/t1)`` composed with the extra dephasing that brings
    the coherence decay to ``exp(-tg/t2)``.  Defined for the validated domain
    ``t2 <= 2*t1``, on both engines -- it is no longer Qiskit-only.

``readout``
    ``p`` is the symmetric confusion probability applied to each measured
    wire at the measurement boundary.

State preparation
-----------------
``BASIS`` and ``STATE_PREP`` define the *initial state*; they are not gates
and carry no gate noise on either engine.  This is the only cross-engine
consistent choice: PennyLane emits one ``AmplitudeEmbedding`` operation while
Qiskit's ``initialize`` decomposes into a state-dependent number of basis
gates, so a per-gate channel attached to preparation cannot agree between the
engines -- and did not: ``bit_flip(p=1)`` over ``BASIS(0)`` gave ``+1`` on
Qiskit and ``-1`` on PennyLane.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .backends._optional import require
from .backends.gates import NOISY_ONE_QUBIT_BASIS, NOISY_TWO_QUBIT_BASIS
from .errors import CapabilityError, ConfigurationError
from .specs import NoiseSpec

_I = np.eye(2, dtype=complex)
_X = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_Y = np.asarray([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
_Z = np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=complex)

_PAULIS: dict[str, np.ndarray] = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}

#: Every spelling the framework accepts, mapped onto its canonical channel.
_CANONICAL_NAMES: dict[str, str] = {
    "none": "none",
    "no_noise": "none",
    "ideal": "none",
    "shot_noise": "shot_noise",
    "bit_flip": "bit_flip",
    "bit_flip_noise": "bit_flip",
    "phase_flip": "phase_flip",
    "phase_flip_noise": "phase_flip",
    "bit_phase_flip": "bit_phase_flip",
    "bit_phase_flip_noise": "bit_phase_flip",
    "depolarizing": "depolarizing",
    "depolarizing_noise": "depolarizing",
    "amplitude_damping": "amplitude_damping",
    "amplitude_damping_noise": "amplitude_damping",
    "phase_damping": "phase_damping",
    "phase_damping_noise": "phase_damping",
    "thermal_relaxation": "thermal_relaxation",
    "thermal_relaxation_noise": "thermal_relaxation",
    "readout": "readout",
    "readout_error": "readout",
    "readout_noise": "readout",
}

#: Channels that leave the state untouched (sampling-only or ideal).
IDEAL_CHANNELS = frozenset({"none", "shot_noise"})

#: Channels applied at the measurement boundary rather than after each gate.
TERMINAL_CHANNELS = frozenset({"readout"})

#: Every canonical channel both engines implement.
CANONICAL_CHANNELS = frozenset(_CANONICAL_NAMES.values())

#: IR operations that define the initial state instead of acting on it.  They
#: are ideal on both engines; see the module docstring.
STATE_PREPARATION_OPERATIONS = frozenset({"BASIS", "STATE_PREP"})

#: IR operations that never carry a gate channel.
NOISELESS_OPERATIONS = STATE_PREPARATION_OPERATIONS | {"BARRIER", "RESET", "MEASURE"}


def canonical_noise_name(name: str) -> str:
    """Return the canonical channel for *name*, or raise naming the options."""

    key = "_".join(str(name).strip().lower().replace("-", "_").split())
    try:
        return _CANONICAL_NAMES[key]
    except KeyError as exc:
        raise CapabilityError(
            f"The core does not support this noise channel: {name!r}. "
            f"Supported: {sorted(CANONICAL_CHANNELS)}"
        ) from exc


def is_ideal(spec: NoiseSpec) -> bool:
    """Return whether *spec* leaves the quantum state unchanged."""

    return canonical_noise_name(spec.name) in IDEAL_CHANNELS


def is_terminal_only(spec: NoiseSpec) -> bool:
    """Return whether *spec* acts only at the measurement boundary."""

    return canonical_noise_name(spec.name) in TERMINAL_CHANNELS


def gate_probability(spec: NoiseSpec, num_wires: int) -> float:
    """Return the channel parameter that applies to a *num_wires* operation."""

    if num_wires == 1:
        value = spec.one_qubit_probability
    else:
        value = spec.two_qubit_probability
    return float(spec.probability if value is None else value)


def readout_probability(spec: NoiseSpec) -> float:
    value = spec.readout_probability
    return float(spec.probability if value is None else value)


def _tensor(left: list[np.ndarray], right: list[np.ndarray]) -> list[np.ndarray]:
    """Independent channels on two wires, in wire-major (big-endian) order."""

    return [np.kron(a, b) for a in left for b in right]


def _pauli_pair_kraus(probability: float, label: str, num_wires: int) -> list[np.ndarray]:
    pauli = _PAULIS[label]
    identity = np.eye(2**num_wires, dtype=complex)
    error = pauli if num_wires == 1 else np.kron(pauli, pauli)
    return [
        np.sqrt(max(0.0, 1.0 - probability)) * identity,
        np.sqrt(max(0.0, probability)) * error,
    ]


def _depolarizing_kraus(probability: float, num_wires: int) -> list[np.ndarray]:
    """Kraus for ``rho -> (1-p) rho + p * I/d``, the canonical convention."""

    dimension = 4**num_wires
    identity_weight = 1.0 - probability * (dimension - 1) / dimension
    each = probability / dimension
    labels = ["I", "X", "Y", "Z"]
    operators: list[np.ndarray] = []
    if num_wires == 1:
        basis = [(label, _PAULIS[label]) for label in labels]
    else:
        basis = [
            (left + right, np.kron(_PAULIS[left], _PAULIS[right]))
            for left in labels
            for right in labels
        ]
    for label, matrix in basis:
        weight = identity_weight if set(label) == {"I"} else each
        if weight <= 0.0:
            continue
        operators.append(np.sqrt(weight) * matrix)
    return operators


def _amplitude_damping_kraus(gamma: float) -> list[np.ndarray]:
    gamma = float(np.clip(gamma, 0.0, 1.0))
    return [
        np.asarray([[1.0, 0.0], [0.0, np.sqrt(1.0 - gamma)]], dtype=complex),
        np.asarray([[0.0, np.sqrt(gamma)], [0.0, 0.0]], dtype=complex),
    ]


def _phase_damping_kraus(lambda_: float) -> list[np.ndarray]:
    lambda_ = float(np.clip(lambda_, 0.0, 1.0))
    return [
        np.asarray([[1.0, 0.0], [0.0, np.sqrt(1.0 - lambda_)]], dtype=complex),
        np.asarray([[0.0, 0.0], [0.0, np.sqrt(lambda_)]], dtype=complex),
    ]


def _compose(outer: list[np.ndarray], inner: list[np.ndarray]) -> list[np.ndarray]:
    """Kraus operators of ``outer`` applied after ``inner``."""

    return [a @ b for a in outer for b in inner]


def _thermal_relaxation_kraus(spec: NoiseSpec) -> list[np.ndarray]:
    if spec.t1 is None or spec.t2 is None or spec.gate_time is None:
        raise CapabilityError("Thermal relaxation requires t1, t2 and gate_time.")
    t1 = float(spec.t1)
    t2 = float(spec.t2)
    gate_time = float(spec.gate_time)
    if t2 > 2.0 * t1:
        raise ConfigurationError("Physical thermal relaxation requires t2 <= 2*t1.")
    # Population decay to |0>, then the extra dephasing that brings coherence
    # from exp(-tg/(2*t1)) down to exp(-tg/t2).  t2 <= 2*t1 keeps the exponent
    # non-positive, so this covers the whole physically valid domain.
    gamma = 1.0 - np.exp(-gate_time / t1)
    extra = 1.0 - np.exp(-gate_time * (2.0 / t2 - 1.0 / t1))
    return _compose(_amplitude_damping_kraus(gamma), _phase_damping_kraus(extra))


def _single_wire_kraus(channel: str, spec: NoiseSpec, probability: float) -> list[np.ndarray]:
    if channel == "bit_flip":
        return _pauli_pair_kraus(probability, "X", 1)
    if channel == "phase_flip":
        return _pauli_pair_kraus(probability, "Z", 1)
    if channel == "bit_phase_flip":
        return _pauli_pair_kraus(probability, "Y", 1)
    if channel == "depolarizing":
        return _depolarizing_kraus(probability, 1)
    if channel == "amplitude_damping":
        return _amplitude_damping_kraus(probability)
    if channel == "phase_damping":
        return _phase_damping_kraus(probability)
    if channel == "thermal_relaxation":
        return _thermal_relaxation_kraus(spec)
    raise CapabilityError(f"The core does not support this noise channel: {channel}")


def channel_kraus(spec: NoiseSpec, num_wires: int) -> tuple[np.ndarray, ...]:
    """Return the canonical Kraus operators for *spec* on *num_wires* wires.

    Operators are ordered wire-major: for two wires the left tensor factor
    belongs to ``wires[0]``.  Qiskit's little-endian ``qargs`` therefore take
    the wire tuple reversed, exactly like observables and state preparation.
    """

    channel = canonical_noise_name(spec.name)
    if channel in IDEAL_CHANNELS:
        return ()
    if channel in TERMINAL_CHANNELS:
        return ()
    if num_wires not in {1, 2}:
        raise CapabilityError(
            f"The core noise layer supports only 1 or 2 wires; {num_wires} were given."
        )
    probability = gate_probability(spec, num_wires)
    if num_wires == 1:
        operators = _single_wire_kraus(channel, spec, probability)
    elif channel in {"bit_flip", "phase_flip", "bit_phase_flip"}:
        label = {"bit_flip": "X", "phase_flip": "Z", "bit_phase_flip": "Y"}[channel]
        operators = _pauli_pair_kraus(probability, label, 2)
    elif channel == "depolarizing":
        operators = _depolarizing_kraus(probability, 2)
    else:
        # Damping and relaxation act independently on each wire.
        single = _single_wire_kraus(channel, spec, probability)
        operators = _tensor(single, single)
    return tuple(np.asarray(item, dtype=complex) for item in operators)


def readout_kraus(spec: NoiseSpec) -> tuple[np.ndarray, ...]:
    """Return the per-wire confusion channel applied before measurement."""

    if canonical_noise_name(spec.name) not in TERMINAL_CHANNELS:
        return ()
    return tuple(_pauli_pair_kraus(readout_probability(spec), "X", 1))


def verify_cptp(operators: tuple[np.ndarray, ...], *, label: str = "channel") -> None:
    """Raise unless ``sum(K^dag K) == I`` -- a channel must be trace preserving."""

    if not operators:
        return
    dimension = operators[0].shape[0]
    total = sum(item.conj().T @ item for item in operators)
    if not np.allclose(total, np.eye(dimension), atol=1e-9):
        raise ConfigurationError(f"{label} is not trace-preserving (sum K†K != I).")


# --------------------------------------------------------------------------
# Engine-facing helpers
# --------------------------------------------------------------------------


def qiskit_channel_instruction(spec: NoiseSpec, num_wires: int) -> Any | None:
    """Return the Aer ``kraus`` instruction for a *num_wires* gate channel.

    The adapter appends this into the circuit rather than configuring an Aer
    ``NoiseModel``.  A ``NoiseModel`` attaches errors by *basis-gate name after
    transpilation*, which made noise depend on how an instruction happened to
    decompose -- ``initialize`` of ``|0>`` picked up no error while ``|1>``
    picked up one -- and could never be aligned with PennyLane's per-operation
    channels.  An in-circuit Kraus instruction is applied exactly where the IR
    says, on both engines.
    """

    operators = channel_kraus(spec, num_wires)
    if not operators:
        return None
    verify_cptp(operators, label=f"{spec.name} ({num_wires} wire)")
    quantum_info = require(
        "qiskit.quantum_info", package="qiskit", extra="qiskit", feature="Qiskit noise"
    )
    return quantum_info.Kraus([np.asarray(item) for item in operators]).to_instruction()


def qiskit_readout_instruction(spec: NoiseSpec) -> Any | None:
    """Return the single-wire readout confusion channel as a Kraus instruction."""

    operators = readout_kraus(spec)
    if not operators:
        return None
    verify_cptp(operators, label=f"{spec.name} (readout)")
    quantum_info = require(
        "qiskit.quantum_info", package="qiskit", extra="qiskit", feature="Qiskit noise"
    )
    return quantum_info.Kraus([np.asarray(item) for item in operators]).to_instruction()


def apply_pennylane_channel(qml: Any, spec: NoiseSpec, wires: tuple[int, ...]) -> None:
    """Apply the canonical gate channel to *wires* on a PennyLane tape."""

    operators = channel_kraus(spec, len(wires))
    if not operators:
        return
    verify_cptp(operators, label=f"{spec.name} ({len(wires)} wire)")
    qml.QubitChannel([np.asarray(item) for item in operators], wires=list(wires))


def apply_pennylane_readout(qml: Any, spec: NoiseSpec, wires: tuple[int, ...]) -> None:
    """Apply the canonical readout confusion channel to each measured wire."""

    operators = readout_kraus(spec)
    if not operators:
        return
    matrices = [np.asarray(item) for item in operators]
    for wire in wires:
        qml.QubitChannel(matrices, wires=[wire])


def build_qiskit_noise_model(spec: NoiseSpec) -> Any | None:
    """Return an Aer ``NoiseModel`` for the canonical channel.

    The adapters no longer use this -- they insert the channel into the
    circuit so both engines apply identical operators at identical positions
    -- but it remains public because an Aer ``NoiseModel`` is what a caller
    needs to hand a channel to Qiskit code outside this framework.  It is
    built from the same canonical Kraus operators, so it agrees numerically
    with the in-circuit path for every gate it covers.
    """

    channel = canonical_noise_name(spec.name)
    if channel in IDEAL_CHANNELS:
        return None
    require("qiskit_aer", package="qiskit-aer", extra="qiskit", feature="Qiskit noise")
    noise = require(
        "qiskit_aer.noise", package="qiskit-aer", extra="qiskit", feature="Qiskit noise"
    )
    model = noise.NoiseModel()
    if channel in TERMINAL_CHANNELS:
        probability = readout_probability(spec)
        model.add_all_qubit_readout_error(
            noise.ReadoutError([[1 - probability, probability], [probability, 1 - probability]])
        )
        return model
    quantum_info = require(
        "qiskit.quantum_info", package="qiskit", extra="qiskit", feature="Qiskit noise"
    )
    one = channel_kraus(spec, 1)
    two = channel_kraus(spec, 2)
    verify_cptp(one, label=f"{spec.name} (1 wire)")
    verify_cptp(two, label=f"{spec.name} (2 wire)")
    model.add_all_qubit_quantum_error(
        noise.QuantumError(quantum_info.Kraus([np.asarray(item) for item in one])),
        list(NOISY_ONE_QUBIT_BASIS),
    )
    model.add_all_qubit_quantum_error(
        noise.QuantumError(quantum_info.Kraus([np.asarray(item) for item in two])),
        list(NOISY_TWO_QUBIT_BASIS),
    )
    return model


__all__ = [
    "CANONICAL_CHANNELS",
    "IDEAL_CHANNELS",
    "TERMINAL_CHANNELS",
    "NOISELESS_OPERATIONS",
    "STATE_PREPARATION_OPERATIONS",
    "canonical_noise_name",
    "is_ideal",
    "is_terminal_only",
    "gate_probability",
    "readout_probability",
    "channel_kraus",
    "readout_kraus",
    "verify_cptp",
    "qiskit_channel_instruction",
    "qiskit_readout_instruction",
    "apply_pennylane_channel",
    "apply_pennylane_readout",
    "build_qiskit_noise_model",
]
