"""Qiskit primitives behind :class:`~qforge_ai.models.kernel.QuantumKernel`.

The kernel used to be delegated wholesale to
``qiskit_machine_learning.kernels``, and two of that stack's defaults quietly
changed the experiment:

* ``FidelityQuantumKernel(evaluate_duplicates='off_diagonal')`` never evaluates
  ``K(x, x)`` -- it writes 1.0.  That is exact for a noiseless overlap and
  wrong for every noisy one, so a ``depolarizing(p=0.2)`` kernel reported a
  diagonal of 1 against PennyLane's 0.32.
* noise reached the kernel through an Aer ``NoiseModel``, which attaches errors
  by *basis-gate name after transpilation*.  The channel then depended on how
  each instruction happened to decompose, so it could not agree with
  PennyLane's per-operation channels even in principle.

Both are fixed by building the circuits here, from the IR, with the same
in-circuit Kraus operators the direct-execution path already uses -- the
placement that :mod:`qforge_ai.noise` was written to make identical across
engines.  This module answers only "what is the encoded state of one sample?"
and "what is the measured overlap of one pair?"; the Gram matrix is assembled
by :mod:`qforge_ai.models.kernel`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..backends._optional import require
from ..backends.qiskit_adapter import QiskitAdapter
from ..errors import ConfigurationError
from ..ir import CircuitIR
from ..noise import qiskit_readout_instruction
from ..specs import BackendSpec
from .kernel import (
    feature_dimension,
    feature_map_operations,
    kernel_measured_wires,
    readout_active,
)
from .kernel_contract import KernelContract

#: Parameter sets pushed through one sampler PUB.  Bounded so an ``n x n``
#: Gram matrix on a wide feature map cannot allocate an unbounded batch.
_PUB_CHUNK = 512


def feature_map_ir(ir: CircuitIR) -> CircuitIR:
    """Return the encoding-only sub-circuit used as a kernel feature map."""

    feature_ir = ir.copy(name=f"{ir.name}_kernel_feature_map")
    feature_ir.operations = list(feature_map_operations(ir, require_invertible=False))
    feature_ir.measurements.clear()
    return feature_ir


def feature_map_circuit(ir: CircuitIR, adapter: QiskitAdapter, *, include_noise: bool = False) -> Any:
    """Compile the encoding-only sub-circuit as a parameterized circuit.

    ``include_noise=False`` keeps the circuit invertible, which the
    compute-uncompute step needs.  ``include_noise=True`` is the analytic
    mixed-state path, where the Kraus channels sit in the circuit at exactly
    the positions the IR names -- the same positions the PennyLane adapter
    uses.
    """

    return adapter.compile(feature_map_ir(ir), include_noise=include_noise)


def _sorted_input_parameters(circuit: Any) -> list[Any]:
    """Return ``input[0..d-1]`` in index order.

    ``QuantumCircuit.parameters`` is sorted by *name*, so ``input[10]`` sorts
    before ``input[2]``.  Binding a sample row positionally against that order
    would transpose feature columns for any feature map wider than ten.
    """

    def index(parameter: Any) -> int:
        text = str(parameter)
        return int(text.rsplit("[", 1)[-1].rstrip("]"))

    return sorted(
        (item for item in circuit.parameters if str(item).startswith("input")),
        key=index,
    )


class QiskitKernelExecutor:
    """Evaluates encoded states and compute-uncompute overlaps on Qiskit."""

    def __init__(
        self, ir: CircuitIR, backend: BackendSpec, contract: KernelContract
    ) -> None:
        self.ir = ir
        self.backend = backend
        self.contract = contract
        self.adapter = QiskitAdapter.from_spec(backend)
        self.operations = feature_map_operations(
            ir, require_invertible=contract.sampled
        )
        self.dimension = feature_dimension(self.operations)
        self._states_circuit: Any | None = None
        self._overlap_circuit: Any | None = None
        self._overlap_parameters: tuple[list[Any], list[Any]] | None = None
        self._sampler: Any | None = None

    # -- analytic ---------------------------------------------------------
    def _encoded_circuit(self) -> tuple[Any, list[Any]]:
        if self._states_circuit is None:
            circuit = feature_map_circuit(
                self.ir,
                self.adapter,
                include_noise=self.contract.fidelity == "density_matrix",
            )
            self._states_circuit = circuit
        return self._states_circuit, _sorted_input_parameters(self._states_circuit)

    def states(self, samples: np.ndarray) -> np.ndarray:
        circuit, parameters = self._encoded_circuit()
        bound = [
            circuit.assign_parameters(dict(zip(parameters, row, strict=True)))
            for row in samples
        ]
        if self.contract.fidelity == "density_matrix":
            return self._density_matrices(bound)
        return self._statevectors(bound)

    def _statevectors(self, circuits: list[Any]) -> np.ndarray:
        quantum_info = require(
            "qiskit.quantum_info",
            package="qiskit",
            extra="qiskit",
            feature="statevector quantum kernel",
        )
        return np.asarray(
            [
                np.asarray(quantum_info.Statevector.from_instruction(item).data).reshape(-1)
                for item in circuits
            ]
        )

    def _density_matrices(self, circuits: list[Any]) -> np.ndarray:
        """Return each encoded ``rho(x)``, flattened.

        Qiskit's kernel stack has no analytic noisy fidelity, so this route
        used to raise ``CapabilityError`` and the same experiment definition
        ran on PennyLane and refused on Qiskit.  Aer evolves the Kraus
        channels exactly, so the mixed-state kernel is available on both
        engines now.
        """

        qiskit = require(
            "qiskit", package="qiskit", extra="qiskit", feature="mixed-state quantum kernel"
        )
        quantum_info = require(
            "qiskit.quantum_info",
            package="qiskit",
            extra="qiskit",
            feature="mixed-state quantum kernel",
        )
        backend = self.adapter.aer_backend(method="density_matrix")
        prepared = []
        for item in circuits:
            copy = item.copy()
            copy.save_density_matrix()
            prepared.append(copy)
        compiled = qiskit.transpile(
            prepared, backend, optimization_level=0, seed_transpiler=self.adapter.seed
        )
        result = backend.run(compiled, seed_simulator=self.adapter.seed).result()
        return np.asarray(
            [
                np.asarray(
                    quantum_info.DensityMatrix(result.data(index)["density_matrix"]).data
                ).reshape(-1)
                for index in range(len(prepared))
            ]
        )

    # -- sampled ----------------------------------------------------------
    def _compute_uncompute_circuit(self) -> tuple[Any, list[Any], list[Any]]:
        """Build ``U(x) [E] U(y)^dagger [E]`` with the channels in the circuit.

        Every gate is emitted through the same ``QiskitAdapter._apply`` the
        execution path uses, and every channel through the same
        ``_apply_channel``; the uncompute leg inverts one operation at a time so
        the channel positions mirror the compute leg exactly, as they do on
        PennyLane.
        """

        if self._overlap_circuit is not None and self._overlap_parameters is not None:
            left, right = self._overlap_parameters
            return self._overlap_circuit, left, right
        qiskit = require(
            "qiskit", package="qiskit", extra="qiskit", feature="sampled quantum kernel"
        )
        circuit_mod = require(
            "qiskit.circuit", package="qiskit", extra="qiskit", feature="sampled quantum kernel"
        )
        width = self.ir.num_qubits
        left_vector = circuit_mod.ParameterVector("left", self.dimension)
        right_vector = circuit_mod.ParameterVector("right", self.dimension)

        def binder(vector: Any) -> Any:
            def value(param: Any) -> Any:
                from ..ir import ParameterRef

                if not isinstance(param, ParameterRef):
                    return param
                return vector[param.index] * param.scale + param.offset

            return value

        circuit = qiskit.QuantumCircuit(width, width)
        circuit.name = f"{self.ir.name}_compute_uncompute"
        noisy = self.contract.noisy
        for operation in self.operations:
            self.adapter._apply(circuit, operation, binder(left_vector))
            if noisy:
                self.adapter._apply_channel(circuit, operation)
        for operation in reversed(self.operations):
            block = qiskit.QuantumCircuit(width)
            self.adapter._apply(block, operation, binder(right_vector))
            circuit.compose(block.inverse(), inplace=True)
            if noisy:
                self.adapter._apply_channel(circuit, operation)
        if readout_active(self.backend, self.contract):
            confusion = qiskit_readout_instruction(self.backend.noise)
            if confusion is not None:
                for wire in kernel_measured_wires(self.ir):
                    circuit.append(confusion, [wire])
        measured = list(kernel_measured_wires(self.ir))
        circuit.measure(measured, measured)
        self._overlap_circuit = circuit
        self._overlap_parameters = (list(left_vector), list(right_vector))
        return circuit, list(left_vector), list(right_vector)

    def _sampler_primitive(self) -> Any:
        if self._sampler is None:
            # Noise travels in the circuit, so the backend must *not* carry a
            # noise model as well: that applied every channel twice.
            self._sampler, _ = self.adapter.sampler_primitive(
                shots=self.contract.shots
            )
        return self._sampler

    def overlaps(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        circuit, left_params, right_params = self._compute_uncompute_circuit()
        shots = self.contract.shots
        if shots is None:  # pragma: no cover - the contract guarantees shots
            raise ConfigurationError("A sampled quantum kernel requires an explicit shots value.")
        sampler = self._sampler_primitive()
        register = circuit.cregs[0].name
        zero = "0" * circuit.num_qubits
        values = np.asarray(
            [np.concatenate((x, y)) for x in left for y in right], dtype=float
        )
        flat = np.empty(values.shape[0], dtype=float)
        for start in range(0, values.shape[0], _PUB_CHUNK):
            batch = values[start : start + _PUB_CHUNK]
            result = sampler.run([(circuit, batch)]).result()[0]
            data = getattr(result.data, register)
            for offset in range(batch.shape[0]):
                counts = data.get_counts(offset)
                total = sum(counts.values()) or shots
                flat[start + offset] = counts.get(zero, 0) / float(total)
        return flat.reshape(left.shape[0], right.shape[0])


__all__ = [
    "QiskitKernelExecutor",
    "feature_map_circuit",
    "feature_map_ir",
]
