"""Quantum and hybrid model construction."""

from .factory import ModelFactory
from .kernel import QuantumKernel, build_quantum_kernel
from .kernel_contract import KERNEL_FIDELITIES, KernelContract, resolve_kernel_contract
from .qiskit_qnn import build_estimator_qnn, build_qsvc, build_sampler_qnn
from .templates import QuantumParallel, QuantumResidual, QuantumSequential

__all__ = [
    "ModelFactory",
    "QuantumSequential",
    "QuantumParallel",
    "QuantumResidual",
    "QuantumKernel",
    "KernelContract",
    "KERNEL_FIDELITIES",
    "build_estimator_qnn",
    "build_sampler_qnn",
    "build_qsvc",
    "build_quantum_kernel",
    "resolve_kernel_contract",
]
