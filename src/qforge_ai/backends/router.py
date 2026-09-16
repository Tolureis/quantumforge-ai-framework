"""Capability-aware backend selection."""

from __future__ import annotations

from ..capabilities import CapabilityRegistry
from ..errors import CapabilityError
from ..ir import CircuitIR
from ..runtime import pennylane_device_constructible, qiskit_aer_methods
from ..specs import BackendSpec, Engine
from .base import BackendAdapter
from .pennylane_adapter import PennyLaneAdapter
from .qiskit_adapter import QiskitAdapter


class BackendRouter:
    def create(self, spec: BackendSpec) -> BackendAdapter:
        if spec.engine == Engine.QISKIT:
            return QiskitAdapter.from_spec(spec)
        if spec.engine == Engine.PENNYLANE:
            return PennyLaneAdapter.from_spec(spec)
        raise CapabilityError("The IR engine represents the circuit but does not perform direct simulation.")

    def recommend(
        self, circuit: CircuitIR, *, noisy: bool = False, gpu: bool = False
    ) -> BackendSpec:
        dynamic = any(op.name in {"RESET", "MEASURE", "CONDITIONAL"} for op in circuit.operations)
        if dynamic:
            return BackendSpec(engine=Engine.QISKIT, device="aer", method="automatic")
        if noisy:
            return BackendSpec(engine=Engine.QISKIT, device="aer", method="density_matrix")
        if gpu:
            if not CapabilityRegistry().get(Engine.QISKIT).gpu:
                raise CapabilityError(
                    "A GPU was requested but the installed Qiskit Aer build offers no GPU device "
                    "on this machine. Verify qiskit-aer-gpu, a compatible CUDA and the NVIDIA driver."
                )
            # Only recommend a method the installed build actually offers:
            # ``tensor_network`` ships with some Aer builds and not others, and
            # a recommendation that fails preflight is worse than a safe one.
            preferred = "tensor_network" if circuit.num_qubits >= 16 else "statevector"
            installed = qiskit_aer_methods()
            method = preferred if (not installed or preferred in installed) else "automatic"
            return BackendSpec(
                engine=Engine.QISKIT,
                device="aer",
                method=method,
                compute_device="GPU",
            )
        if circuit.num_qubits >= 20 and circuit.two_qubit_gate_count() < 4 * circuit.num_qubits:
            return BackendSpec(engine=Engine.QISKIT, device="aer", method="matrix_product_state")
        if pennylane_device_constructible("lightning.qubit"):
            return BackendSpec(engine=Engine.PENNYLANE, device="lightning.qubit")
        return BackendSpec(engine=Engine.PENNYLANE, device="default.qubit")


__all__ = ["BackendRouter"]
