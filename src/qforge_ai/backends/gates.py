"""Canonical IR-to-Qiskit gate names shared by the adapter and the noise builder.

The noise model and the circuit translator used to keep two independent gate
lists.  They drifted: ``S``, ``T``, ``CY``, ``CRX``, ``CRY`` and ``CRZ`` were
translatable but carried no noise, so a QCNN -- which leans on ``CRY`` for
pooling -- had materially different noise semantics on Qiskit than on
PennyLane.  Both sides now read the same tables, so a gate cannot be added to
one without the other.
"""

from __future__ import annotations

#: IR operation name -> ``QuantumCircuit`` method for one-qubit gates.
QISKIT_SINGLE_QUBIT_GATES: dict[str, str] = {
    "X": "x",
    "Y": "y",
    "Z": "z",
    "H": "h",
    "S": "s",
    "T": "t",
    "SX": "sx",
    "RX": "rx",
    "RY": "ry",
    "RZ": "rz",
    "U": "u",
}

#: IR operation name -> ``QuantumCircuit`` method for two-qubit gates.
QISKIT_TWO_QUBIT_GATES: dict[str, str] = {
    "CX": "cx",
    "CNOT": "cx",
    "CZ": "cz",
    "CY": "cy",
    "SWAP": "swap",
    "CRX": "crx",
    "CRY": "cry",
    "CRZ": "crz",
    "RXX": "rxx",
    "RYY": "ryy",
    "RZZ": "rzz",
}

#: Basis names an Aer ``NoiseModel`` must cover for one-qubit operations.
#: ``RZ_PRODUCT`` and ``BASIS`` compile down to ``rz``/``x``, so they are
#: covered transitively.
NOISY_ONE_QUBIT_BASIS: tuple[str, ...] = tuple(
    dict.fromkeys(QISKIT_SINGLE_QUBIT_GATES.values())
)

#: Basis names an Aer ``NoiseModel`` must cover for two-qubit operations.
#: ``RZZ_PRODUCT`` compiles down to ``rzz``.
NOISY_TWO_QUBIT_BASIS: tuple[str, ...] = tuple(
    dict.fromkeys(QISKIT_TWO_QUBIT_GATES.values())
)

__all__ = [
    "QISKIT_SINGLE_QUBIT_GATES",
    "QISKIT_TWO_QUBIT_GATES",
    "NOISY_ONE_QUBIT_BASIS",
    "NOISY_TWO_QUBIT_BASIS",
]
