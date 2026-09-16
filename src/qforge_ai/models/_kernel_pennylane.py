"""PennyLane primitives behind :class:`~qforge_ai.models.kernel.QuantumKernel`.

This module owns no kernel mathematics.  It answers exactly two questions --
"what is the encoded state of one sample?" and "what is the measured overlap of
one pair?" -- and :mod:`qforge_ai.models.kernel` assembles the Gram matrix from
the answers, identically for both engines.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..backends._optional import require
from ..backends.pennylane_adapter import PennyLaneAdapter
from ..ir import CircuitIR, ParameterRef
from ..noise import apply_pennylane_readout
from ..specs import BackendSpec
from .kernel import (
    feature_map_operations,
    kernel_measured_wires,
    readout_active,
)
from .kernel_contract import KernelContract


def kernel_adapter(backend: BackendSpec) -> PennyLaneAdapter:
    return PennyLaneAdapter(
        device=backend.device if backend.device != "statevector" else "default.qubit",
        shots=backend.shots,
        seed=backend.seed,
        noise=backend.noise,
        device_options=dict(backend.options.get("device_options", {})),
    )


def kernel_device_name(backend: BackendSpec, contract: KernelContract) -> str:
    """Return the PennyLane device this kernel will actually construct.

    A mixed-state kernel needs a mixed-state device, and a noisy channel moves
    ``default.qubit``/``lightning.qubit`` onto ``default.mixed``.  Reporting the
    *requested* device instead made a result produced on ``default.mixed``
    claim to come from ``lightning.gpu``.
    """

    if contract.fidelity == "density_matrix":
        return "default.mixed"
    return kernel_adapter(backend).resolved_device_name()


class PennyLaneKernelExecutor:
    """Evaluates encoded states and compute-uncompute overlaps on PennyLane."""

    def __init__(
        self, ir: CircuitIR, backend: BackendSpec, contract: KernelContract
    ) -> None:
        self.ir = ir
        self.backend = backend
        self.contract = contract
        self.operations = feature_map_operations(
            ir, require_invertible=contract.sampled
        )
        self._qnode: Any | None = None

    # -- construction -----------------------------------------------------
    def _build(self) -> Any:
        if self._qnode is not None:
            return self._qnode
        qml = require(
            "pennylane", package="pennylane", extra="pennylane", feature="quantum kernel"
        )
        adapter = kernel_adapter(self.backend)
        operations = self.operations
        noisy = self.contract.noisy
        device_options = dict(adapter.device_options)
        device_options["seed"] = adapter.seed
        device = qml.device(
            kernel_device_name(self.backend, self.contract),
            wires=self.ir.num_qubits,
            **device_options,
        )
        wires = kernel_measured_wires(self.ir)

        def binder(sample: Any) -> Any:
            def resolve(param: Any) -> Any:
                if not isinstance(param, ParameterRef):
                    return param
                return sample[..., param.index] * param.scale + param.offset

            return resolve

        if self.contract.analytic:

            @qml.qnode(device)
            def encoded(sample: Any) -> Any:
                for operation in operations:
                    adapter._apply(qml, operation, binder(sample))
                    adapter._apply_channel(qml, operation, noisy=noisy)
                if self.contract.fidelity == "density_matrix":
                    return qml.density_matrix(wires=wires)
                return qml.state()

            self._qnode = encoded
            return encoded

        readout = readout_active(self.backend, self.contract)
        noise = self.backend.noise

        @qml.qnode(device)
        def overlap(left: Any, right: Any) -> Any:
            for operation in operations:
                adapter._apply(qml, operation, binder(left))
                adapter._apply_channel(qml, operation, noisy=noisy)
            for operation in reversed(operations):
                # ``adjoint`` of the gate, then the channel: noise is physical
                # and is not undone by running the circuit backwards.
                qml.adjoint(adapter._apply)(qml, operation, binder(right))
                adapter._apply_channel(qml, operation, noisy=noisy)
            if readout:
                # The confusion channel lives at the measurement boundary of
                # the fidelity circuit.  It used to be dropped here entirely,
                # so ``readout_probability=1.0`` produced a Gram matrix
                # bit-identical to the noiseless one while the same definition
                # moved the Qiskit matrix by 0.79.
                apply_pennylane_readout(qml, noise, wires)
            return qml.probs(wires=wires)

        self._qnode = qml.set_shots(overlap, shots=self.contract.shots)
        return self._qnode

    # -- primitives -------------------------------------------------------
    def states(self, samples: np.ndarray) -> np.ndarray:
        executable = self._build()
        return np.asarray(
            [np.asarray(executable(row)).reshape(-1) for row in samples]
        )

    def overlaps(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        executable = self._build()
        matrix = np.empty((left.shape[0], right.shape[0]), dtype=float)
        for row, x in enumerate(left):
            for column, y in enumerate(right):
                probabilities = np.asarray(executable(x, y), dtype=float)
                matrix[row, column] = float(probabilities[0])
        return matrix


__all__ = ["PennyLaneKernelExecutor", "kernel_adapter", "kernel_device_name"]
