"""Backend capability contracts and strict preflight validation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from .errors import CapabilityError, ConfigurationError
from .ir import CircuitIR
from .noise import canonical_noise_name, is_ideal
from .runtime import (
    pennylane_device_constructible,
    pennylane_gpu_available,
    pennylane_registered_devices,
    qiskit_aer_methods,
    qiskit_gpu_available,
)
from .specs import PENNYLANE_GPU_DEVICES, BackendSpec, Engine, GradientMethod

if TYPE_CHECKING:  # pragma: no cover - import cycle broken for runtime
    from .models.kernel_contract import KernelContract

#: Channels the shared Kraus core implements, in the spelling the catalog and
#: NoiseSpec use.  Both engines execute the *same* operators, so this table is
#: identical for Qiskit and PennyLane.
CORE_NOISE_CHANNELS = frozenset(
    {
        "none",
        "no_noise",
        "shot_noise",
        "bit_flip_noise",
        "phase_flip_noise",
        "bit_phase_flip_noise",
        "depolarizing_noise",
        "amplitude_damping",
        "phase_damping",
        "readout_error",
        "thermal_relaxation",
    }
)

SUPPORTED_MEASUREMENTS = frozenset(
    {"expectation", "variance", "probability", "sample", "counts", "statevector", "density_matrix"}
)
COMMON_OPERATIONS = frozenset(
    {
        "X",
        "Y",
        "Z",
        "H",
        "S",
        "T",
        "SX",
        "RX",
        "RY",
        "RZ",
        "U",
        "CX",
        "CNOT",
        "CZ",
        "CY",
        "SWAP",
        "CRX",
        "CRY",
        "CRZ",
        "RXX",
        "RYY",
        "RZZ",
        "BASIS",
        "STATE_PREP",
        "RZ_PRODUCT",
        "RZZ_PRODUCT",
        "RESET",
        "MEASURE",
        "BARRIER",
    }
)


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    engine: Engine
    operations: frozenset[str]
    measurements: frozenset[str]
    noise_channels: frozenset[str]
    gradient_methods: frozenset[GradientMethod]
    devices: frozenset[str]
    finite_shots: bool
    mid_circuit_measurement: bool
    continuous_variable: bool
    gpu: bool

    @property
    def statevector(self) -> bool:
        return "statevector" in self.measurements

    @property
    def noise(self) -> bool:
        return bool(self.noise_channels - {"none", "no_noise", "shot_noise"})

    @property
    def analytic_gradient(self) -> bool:
        return bool(
            self.gradient_methods
            & {GradientMethod.ADJOINT, GradientMethod.BACKPROP, GradientMethod.PARAMETER_SHIFT}
        )


CAPABILITIES = {
    Engine.QISKIT: BackendCapabilities(
        engine=Engine.QISKIT,
        operations=COMMON_OPERATIONS,
        measurements=SUPPORTED_MEASUREMENTS,
        noise_channels=CORE_NOISE_CHANNELS,
        gradient_methods=frozenset(
            {
                GradientMethod.AUTO,
                GradientMethod.PARAMETER_SHIFT,
                GradientMethod.FINITE_DIFF,
                GradientMethod.SPSA,
            }
        ),
        devices=frozenset({"aer", "aer_simulator", "statevector"}),
        finite_shots=True,
        mid_circuit_measurement=True,
        continuous_variable=False,
        # Potential support is not a runtime guarantee. CapabilityRegistry.get
        # replaces this value after probing the installed Aer build.
        gpu=False,
    ),
    Engine.PENNYLANE: BackendCapabilities(
        engine=Engine.PENNYLANE,
        operations=COMMON_OPERATIONS,
        measurements=SUPPORTED_MEASUREMENTS,
        # Identical to Qiskit: both engines execute the same canonical Kraus
        # operators from qforge_ai.noise, so thermal relaxation is no longer
        # Qiskit-only and bit-phase-flip is available on both.
        noise_channels=CORE_NOISE_CHANNELS,
        gradient_methods=frozenset(
            {
                GradientMethod.AUTO,
                GradientMethod.PARAMETER_SHIFT,
                GradientMethod.ADJOINT,
                GradientMethod.BACKPROP,
                GradientMethod.FINITE_DIFF,
                GradientMethod.SPSA,
            }
        ),
        devices=frozenset(
            {"statevector", "default.qubit", "default.mixed", "lightning.qubit", "lightning.gpu"}
        ),
        finite_shots=True,
        mid_circuit_measurement=True,
        continuous_variable=False,
        # Declared potential only. ``CapabilityRegistry.get`` narrows this to
        # devices that are actually registered on this machine, because a
        # plain pennylane-lightning install has no lightning.gpu at all.
        gpu=True,
    ),
    Engine.IR: BackendCapabilities(
        engine=Engine.IR,
        operations=frozenset(),
        measurements=frozenset(),
        noise_channels=frozenset(),
        gradient_methods=frozenset(),
        devices=frozenset({"statevector"}),
        finite_shots=False,
        mid_circuit_measurement=False,
        continuous_variable=False,
        gpu=False,
    ),
}


class CapabilityRegistry:
    def get(self, engine: Engine | str) -> BackendCapabilities:
        """Return capabilities as probed on *this* machine.

        The declared tables describe what the framework supports in principle.
        What matters to a caller is what the installed runtime can actually
        do, so device and GPU claims are replaced with probe results before
        the record is handed out.  Reporting ``lightning.gpu`` on an install
        that raises ``DeviceError`` for it is worse than reporting nothing.
        """

        resolved = Engine(engine)
        capability = CAPABILITIES[resolved]
        if resolved == Engine.QISKIT:
            return replace(capability, gpu=qiskit_gpu_available())
        if resolved == Engine.PENNYLANE:
            registered = pennylane_registered_devices()
            if not registered:
                # PennyLane itself is not importable; keep the declared record
                # so an offline catalog listing still describes the contract.
                return replace(capability, gpu=False)
            available = frozenset(
                name
                for name in capability.devices
                # ``statevector`` is a QuantumForge alias resolved to
                # default.qubit by the adapter, not a PennyLane device name.
                if name == "statevector" or name in registered
            )
            return replace(capability, devices=available, gpu=pennylane_gpu_available())
        return capability

    def validate(self, circuit: CircuitIR, backend: BackendSpec) -> list[str]:
        circuit.validate()
        capability = self.get(backend.engine)
        warnings = list(circuit.metadata.get("warnings", ()))
        if backend.engine == Engine.IR:
            warnings.append(
                "The IR backend is valid for analysis and serialization; Qiskit or PennyLane "
                "must be selected for a run/build_model call."
            )
            return warnings

        if backend.device not in capability.devices:
            raise CapabilityError(
                f"{backend.engine} does not support this device: {backend.device}. "
                f"Supported: {sorted(capability.devices)}"
            )

        unsupported_operations = sorted(
            {operation.name for operation in circuit.operations} - capability.operations
        )
        if unsupported_operations:
            raise CapabilityError(
                f"{backend.engine} does not support these IR operations: {unsupported_operations}"
            )

        measurement_kinds = {measurement.kind for measurement in circuit.measurements}
        unsupported_measurements = sorted(measurement_kinds - capability.measurements)
        if unsupported_measurements:
            raise CapabilityError(
                f"{backend.engine} does not support these measurements: {unsupported_measurements}"
            )

        dynamic = any(
            operation.name in {"RESET", "MEASURE"} or operation.condition is not None
            for operation in circuit.operations
        )
        if dynamic and not capability.mid_circuit_measurement:
            raise CapabilityError(f"{backend.engine} does not support dynamic circuits.")
        if dynamic and backend.engine == Engine.QISKIT and backend.device == "statevector":
            raise CapabilityError("Qiskit requires device='aer' for dynamic circuits.")

        noise_name = backend.noise.name
        try:
            canonical = canonical_noise_name(noise_name)
        except CapabilityError:
            raise CapabilityError(
                f"{backend.engine} does not support this noise channel: {noise_name}. "
                f"Supported: {sorted(capability.noise_channels)}"
            ) from None
        if canonical not in {canonical_noise_name(item) for item in capability.noise_channels}:
            raise CapabilityError(
                f"{backend.engine} does not support this noise channel: {noise_name}. "
                f"Supported: {sorted(capability.noise_channels)}"
            )
        if canonical == "readout" and not measurement_kinds <= {
            "counts",
            "sample",
        }:
            raise CapabilityError(
                "Readout error is supported only with a computational-basis counts or sample "
                "measurement."
            )
        noisy = canonical not in {"none", "shot_noise"}
        if noisy and "statevector" in measurement_kinds:
            raise CapabilityError(
                "A noisy channel does not produce a pure statevector; use a density_matrix measurement."
            )
        if noisy and backend.engine == Engine.PENNYLANE:
            if backend.device in {"default.qubit", "lightning.qubit"}:
                warnings.append(
                    f"{backend.device} will be switched automatically to the default.mixed device for noise."
                )
            elif backend.device != "default.mixed":
                raise CapabilityError(
                    f"The verified device for PennyLane noise is default.mixed; got {backend.device}."
                )

        effective_shots = backend.shots
        if effective_shots is None and circuit.measurements:
            effective_shots = circuit.measurements[0].shots
        if effective_shots is not None and not capability.finite_shots:
            raise CapabilityError(f"{backend.engine} does not support finite shots.")
        if measurement_kinds & {"sample", "counts"} and effective_shots is None:
            warnings.append("No shots were specified for sample/counts; the executor will use 1024 shots.")

        if backend.gradient not in capability.gradient_methods:
            raise CapabilityError(
                f"{backend.engine} does not support this gradient method: {backend.gradient}. "
                f"Supported: {sorted(map(str, capability.gradient_methods))}"
            )
        if effective_shots is not None and backend.gradient in {
            GradientMethod.BACKPROP,
            GradientMethod.ADJOINT,
        }:
            raise CapabilityError(
                f"{backend.gradient} cannot be used on a finite-shot circuit; choose parameter_shift."
            )
        if noisy and backend.gradient == GradientMethod.ADJOINT:
            raise CapabilityError(
                "adjoint cannot be used on a noisy circuit; choose parameter_shift or backprop."
            )

        if backend.engine == Engine.QISKIT and backend.compute_device == "GPU" and not capability.gpu:
            raise CapabilityError(
                "Qiskit GPU execution is not available on this machine: the installed Aer build "
                "offers no GPU device. Verify qiskit-aer-gpu, a compatible CUDA and the NVIDIA driver."
            )
        if backend.engine == Engine.QISKIT:
            installed_methods = qiskit_aer_methods()
            if installed_methods and backend.method not in installed_methods:
                raise CapabilityError(
                    f"The installed Aer build does not offer the '{backend.method}' method. "
                    f"Methods available on this machine: {sorted(installed_methods)}"
                )
        if backend.engine == Engine.PENNYLANE and backend.device != "statevector":
            if not pennylane_device_constructible(backend.device):
                raise CapabilityError(
                    f"The PennyLane device cannot be created on this machine: {backend.device}. "
                    f"Registered devices: {sorted(pennylane_registered_devices())}. "
                    "Install the required plugin (e.g. pennylane-lightning-gpu for lightning.gpu)."
                )
        if backend.engine == Engine.PENNYLANE and backend.compute_device == "GPU" and not capability.gpu:
            raise CapabilityError(
                "PennyLane GPU execution is not available on this machine: no registered GPU device "
                "(lightning.gpu/lightning.kokkos) was found."
            )
        if circuit.num_qubits > 24 and backend.method in {"statevector", "automatic"}:
            warnings.append(
                "A 24+ qubit statevector can consume a lot of memory; consider MPS/tensor-network."
            )
        return warnings


def validate_kernel_backend(
    backend: BackendSpec, contract: KernelContract | Any
) -> str:
    """Preflight a quantum-kernel backend and return the device that will run.

    Every other route runs :meth:`CapabilityRegistry.validate` before it
    executes; the kernel route ran nothing.  ``compute_device='GPU'`` with
    ``device='lightning.gpu'`` was therefore accepted on a machine with no GPU
    at all, the kernel silently fell back to ``default.mixed`` on the CPU, and
    the reported provenance still said GPU.  The earlier GPU capability fix
    covered execution and QNN construction only -- not this path.

    The returned name is the device the kernel will *actually* construct, so a
    caller that reports it cannot describe a run that never happened.
    """

    from .models.kernel_contract import ANALYTIC_FIDELITIES

    registry = CapabilityRegistry()
    capability = registry.get(backend.engine)
    if backend.device not in capability.devices:
        raise CapabilityError(
            f"{backend.engine} does not support this device: {backend.device}. "
            f"Supported: {sorted(capability.devices)}"
        )
    if backend.compute_device == "GPU" and not capability.gpu:
        raise CapabilityError(
            f"{backend.engine} GPU execution is not available on this machine, so the "
            "quantum-kernel path rejects it rather than falling back to the CPU silently. "
            + (
                "Verify qiskit-aer-gpu, a compatible CUDA and the NVIDIA driver."
                if backend.engine == Engine.QISKIT
                else "No registered GPU device (lightning.gpu/lightning.kokkos) was found."
            )
        )
    if backend.engine == Engine.QISKIT:
        methods = qiskit_aer_methods()
        if methods and backend.method not in methods:
            raise CapabilityError(
                f"The installed Aer build does not offer the '{backend.method}' method. "
                f"Methods available on this machine: {sorted(methods)}"
            )
        return backend.device
    if backend.device != "statevector" and not pennylane_device_constructible(backend.device):
        raise CapabilityError(
            f"The PennyLane device cannot be created on this machine: {backend.device}. "
            f"Registered devices: {sorted(pennylane_registered_devices())}. "
            "Install the required plugin (e.g. pennylane-lightning-gpu for lightning.gpu)."
        )
    from .models._kernel_pennylane import kernel_device_name

    device = kernel_device_name(backend, contract)
    requested = backend.device if backend.device != "statevector" else "default.qubit"
    if device != requested and requested in PENNYLANE_GPU_DEVICES:
        # A GPU device that the resolved kernel cannot run on is a contradiction
        # in the definition, not a detail to resolve silently: the mixed-state
        # kernel needs default.mixed and lightning.gpu carries no channels.
        raise CapabilityError(
            f"The {contract.fidelity} kernel requires the {device!r} device, but "
            f"device={requested!r} was requested. A GPU request is rejected rather than "
            "being dropped silently onto the CPU: choose fidelity='statevector' for a "
            "noiseless analytic kernel, fidelity='sampler' for sampling on the GPU, "
            "or make device consistent with the device the kernel runs on."
        )
    if (
        contract.fidelity in ANALYTIC_FIDELITIES
        and backend.compute_device == "GPU"
        and device not in PENNYLANE_GPU_DEVICES
    ):
        raise CapabilityError(
            f"compute_device='GPU' was requested, but the {contract.fidelity} kernel "
            f"runs on {device!r}. In PennyLane the GPU is selected by device name; "
            "this kernel is rejected rather than being run on the CPU silently."
        )
    if not is_ideal(backend.noise) and device == backend.device == "default.qubit":
        raise ConfigurationError(  # pragma: no cover - defensive
            "A noisy kernel cannot be run on default.qubit."
        )
    return device


__all__ = [
    "BackendCapabilities",
    "CapabilityRegistry",
    "CAPABILITIES",
    "validate_kernel_backend",
]
