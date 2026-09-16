"""One quantum-kernel implementation, executed by two engines.

Before this module the two engines each owned a *separate* kernel
implementation, and the same ``ExperimentSpec`` produced two different Gram
matrices:

======================================================  ======================  ======================
definition                                              Qiskit                  PennyLane
======================================================  ======================  ======================
``depolarizing(p=0.2)``, ``fidelity='sampler'``         diagonal pinned to 1    diagonal ~0.32
``readout_probability=1.0``, ``fidelity='sampler'``     applied                 silently dropped
a third feature column on a 2-feature map               explicit error          silently ignored
``compute_device='GPU'`` on a noisy kernel              accepted, ran on CPU    accepted, ran on CPU
``fidelity='density_matrix'``                           CapabilityError         supported
======================================================  ======================  ======================

None of those differences were declared anywhere: the first three were
side effects of two third-party kernel classes disagreeing about their
defaults, the fourth of the kernel route skipping the capability preflight
that every other route runs.

:class:`QuantumKernel` is now the framework's *only* kernel.  It owns the
mathematics -- which quantity is computed, how the Gram matrix is assembled,
which inputs are legal -- and each engine supplies nothing but the primitive
that evaluates one encoded state or one overlap.  The engines therefore cannot
drift apart again, because there is only one definition left to drift from.

The three kernel kinds
----------------------
``statevector``
    ``K(x,y) = |<phi(x)|phi(y)>|^2``.  A genuine fidelity kernel between two
    *pure* states, positive semi-definite by the Schur product theorem.

``density_matrix``
    ``K(x,y) = Tr[rho(x) rho(y)]``, the **Hilbert-Schmidt inner product** of
    the two noisy encoded states.  This is a valid Mercer kernel -- it is the
    Gram matrix of the vectorized density matrices, so it is symmetric and PSD
    by construction -- but it is *not* the mixed-state (Uhlmann) fidelity, and
    the framework no longer calls it one.  ``F(rho, rho) = 1`` for every state,
    while ``Tr[rho^2]`` is the purity and drops below 1 under any
    state-changing channel.  ``hilbert_schmidt`` is the preferred spelling;
    ``density_matrix`` remains accepted as the name of the simulation method it
    needs.

``sampler``
    Finite-shot compute-uncompute: prepare ``U(x)``, apply the noise channels,
    invert with ``U(y)^dagger`` -- applying the channels again, because noise is
    physical and is not undone by running a circuit backwards -- and read the
    probability of the all-zero string.  Every entry is measured, including the
    diagonal: under a state-changing channel ``K(x,x)`` is genuinely below 1,
    and pinning it to 1 (``qiskit-machine-learning``'s default
    ``evaluate_duplicates='off_diagonal'``) hands the SVM a Gram matrix whose
    diagonal comes from a different experiment than its off-diagonal.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..capabilities import validate_kernel_backend
from ..composition import validate_kernel_composition
from ..errors import ConfigurationError
from ..ir import CircuitIR, Operation, ParameterRef
from ..noise import is_terminal_only
from ..specs import BackendSpec, Engine
from .kernel_contract import KernelContract, resolve_kernel_contract

#: Operations no kernel can use: the encoded state they leave behind is not a
#: deterministic function of the input, so ``K(x, x)`` is not even well defined.
NON_DETERMINISTIC_OPERATIONS = frozenset({"MEASURE", "RESET"})

#: Operations a *compute-uncompute* kernel cannot use.  State preparation
#: overwrites the register instead of acting on it, so ``U(y)†`` does not exist
#: for it on either engine.  The analytic kernels never invert anything -- they
#: build the encoded state directly -- so this restriction applies only to the
#: sampled path.
NON_INVERTIBLE_OPERATIONS = frozenset({"STATE_PREP", "BASIS"})


def feature_map_operations(
    ir: CircuitIR,
    *,
    require_invertible: bool = True,
) -> list[Operation]:
    """Return the encoding-only operations that form a kernel feature map.

    Shared by both engines so a kernel built from the same IR uses the same
    sub-circuit, and so the "this feature map cannot be inverted" error is
    raised once, in the same words, instead of surfacing as an engine-specific
    failure deep inside a fidelity primitive.

    ``require_invertible=False`` is for the analytic kernels, which evaluate
    the encoded states directly and therefore accept a state-preparation
    feature map that compute-uncompute cannot.
    """

    operations = [
        operation
        for operation in ir.operations
        if operation.metadata.get("role") == "encoding"
    ]
    if not operations:
        raise ConfigurationError(
            "No encoding role was found for the kernel feature map. Build the circuit with "
            "CircuitFactory, or use CircuitFactory.build_feature_map()."
        )
    if any(
        isinstance(param, ParameterRef) and param.group == "weight"
        for operation in operations
        for param in operation.params
    ):
        raise ConfigurationError(
            "A kernel feature map cannot contain a trainable weight parameter; "
            "use a plugin for a trainable kernel."
        )
    names = {op.name for op in operations}
    non_deterministic = sorted(names & NON_DETERMINISTIC_OPERATIONS)
    if non_deterministic:
        raise ConfigurationError(
            f"A kernel feature map must be deterministic; these operations make the encoded "
            f"state depend on a measurement result: {non_deterministic}."
        )
    blocked = sorted(names & NON_INVERTIBLE_OPERATIONS) if require_invertible else []
    if blocked:
        raise ConfigurationError(
            f"The fidelity kernel compute-uncompute step inverts the feature map and "
            f"the inverse of these operations is not defined: {blocked}. For a kernel use an "
            "angle/phase based encoding (angle, phase, zz_feature_map, etc.) or an analytic "
            "kernel (fidelity='statevector'/'hilbert_schmidt')."
        )
    return operations


def feature_dimension(operations: list[Operation]) -> int:
    """Return how many input features *operations* actually consume.

    This is the kernel's contract with its data: a Gram matrix built from a
    feature map that reads two columns is a function of two columns, whatever
    the caller passes.  PennyLane used to index the sample row per parameter
    and simply never touch the rest, so a third column could be added to the
    data -- or two columns swapped in -- with the Gram matrix unchanged and no
    diagnostic anywhere.  Qiskit rejected the same call outright.
    """

    indices = [
        param.index
        for operation in operations
        for param in operation.params
        if isinstance(param, ParameterRef) and param.group == "input"
    ]
    if not indices:
        raise ConfigurationError(
            "A kernel feature map must depend on the data; the encoding layer has no "
            "input parameter at all, so the Gram matrix would be constant for every "
            "pair of samples."
        )
    return max(indices) + 1


def enforce_psd(matrix: np.ndarray) -> np.ndarray:
    """Return the nearest symmetric positive-semi-definite matrix to *matrix*.

    A Mercer kernel's Gram matrix is symmetric and has no negative eigenvalues.
    A finite-shot compute-uncompute estimate satisfies neither: ``K(x,y)`` and
    ``K(y,x)`` are two independent sampling experiments, so the raw matrix is
    asymmetric and can carry small negative eigenvalues -- which is enough to
    make an SVM's dual problem non-convex.  Both engines apply this same
    repair, so a finite-shot Gram matrix is a usable Mercer kernel whichever
    engine produced it.
    """

    symmetric = np.asarray(0.5 * (matrix + matrix.T), dtype=float)
    values, vectors = np.linalg.eigh(symmetric)
    if values.min() >= 0.0:
        return symmetric
    clipped = np.clip(values, 0.0, None)
    return np.asarray((vectors * clipped) @ vectors.T)


# ---------------------------------------------------------------------------
# The kernel
# ---------------------------------------------------------------------------


class QuantumKernel:
    """Callable Gram-matrix builder with an explicit, engine-independent contract.

    The kernel is picklable: the engine executable is rebuilt from the IR and
    the backend spec on first use, so an estimator holding one survives
    ``joblib``/``pickle`` round trips inside a scikit-learn pipeline.
    """

    def __init__(
        self,
        *,
        ir: CircuitIR,
        backend: BackendSpec,
        contract: KernelContract,
        device: str,
        kind: str,
    ) -> None:
        self.qforge_ir = ir
        self.qforge_backend_spec = backend
        self.qforge_contract = contract
        self.qforge_device = device
        self.qforge_shots = contract.shots
        self.qforge_kernel_kind = kind
        self.qforge_feature_dimension = feature_dimension(
            feature_map_operations(ir, require_invertible=contract.sampled)
        )
        self._executor: Any | None = None

    # -- introspection ----------------------------------------------------
    @property
    def qforge_engine(self) -> Engine:
        return self.qforge_backend_spec.engine

    @property
    def feature_dimension(self) -> int:
        """Number of input columns this kernel's feature map consumes."""

        return self.qforge_feature_dimension

    @property
    def feature_map(self) -> CircuitIR:
        """The encoding-only sub-circuit this kernel evaluates.

        Returned as a :class:`~qforge_ai.ir.CircuitIR` on *both* engines.  It
        used to be a ``qiskit`` ``QuantumCircuit`` on one route and nothing at
        all on the other, so "which circuit is my kernel actually using?" could
        only be answered on Qiskit.  Use :meth:`engine_feature_map` for the
        compiled engine object.
        """

        feature_ir = self.qforge_ir.copy(
            name=f"{self.qforge_ir.name}_kernel_feature_map"
        )
        feature_ir.operations = list(
            feature_map_operations(
                self.qforge_ir, require_invertible=self.qforge_contract.sampled
            )
        )
        feature_ir.measurements.clear()
        return feature_ir

    def engine_feature_map(self) -> Any:
        """Return the compiled feature map, in the engine's own representation.

        A ``QuantumCircuit`` on Qiskit and the executable QNode on PennyLane.
        """

        if self.qforge_engine == Engine.QISKIT:
            from ..backends.qiskit_adapter import QiskitAdapter
            from ._kernel_qiskit import feature_map_circuit

            return feature_map_circuit(
                self.qforge_ir,
                QiskitAdapter.from_spec(self.qforge_backend_spec),
                include_noise=self.qforge_contract.fidelity == "density_matrix",
            )
        return self._build()._build()

    # -- pickling ---------------------------------------------------------
    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        # An engine executable closes over a live device and a traced or
        # compiled circuit; it is rebuilt from the IR and the spec instead of
        # being serialized.
        state["_executor"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, QuantumKernel):
            return NotImplemented
        return (
            self.qforge_ir.to_dict() == other.qforge_ir.to_dict()
            and self.qforge_backend_spec == other.qforge_backend_spec
            and self.qforge_contract == other.qforge_contract
        )

    def __repr__(self) -> str:
        return (
            f"QuantumKernel(kind={self.qforge_kernel_kind!r}, "
            f"engine={str(self.qforge_engine)!r}, device={self.qforge_device!r}, "
            f"shots={self.qforge_shots!r}, features={self.feature_dimension})"
        )

    # -- evaluation -------------------------------------------------------
    def _build(self) -> Any:
        if self._executor is None:
            self._executor = _build_executor(
                self.qforge_ir, self.qforge_backend_spec, self.qforge_contract
            )
        return self._executor

    def _samples(self, values: Any, *, label: str) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2:
            raise ConfigurationError(
                f"Quantum kernel {label} data must be 2-dimensional (samples, features); "
                f"a {array.ndim}-dimensional array was given."
            )
        expected = self.feature_dimension
        if array.shape[1] != expected:
            raise ConfigurationError(
                f"The quantum kernel feature map reads {expected} features; the {label} data "
                f"carries {array.shape[1]} columns. Extra columns are rejected rather than being "
                "dropped silently: the data and the feature map must describe the same feature space. "
                f"Grow the circuit with EncodingSpec(feature_dimension={array.shape[1]}) "
                "or reduce the data to the width of the feature map."
            )
        if not np.all(np.isfinite(array)):
            raise ConfigurationError(
                f"Quantum kernel {label} data may contain only finite numbers; "
                "once NaN/Inf is converted into an angle the circuit becomes undefined."
            )
        return array

    def __call__(self, left: Any, right: Any | None = None) -> np.ndarray:
        """Return the Gram matrix of *left* against *right* (or itself)."""

        left_array = self._samples(left, label="left")
        # scikit-learn's ``SVC`` calls a callable kernel as ``kernel(X, X)``
        # rather than ``kernel(X)``, so "is this the training Gram matrix?"
        # cannot be answered by ``right is None`` alone -- and the answer
        # decides whether the PSD repair runs at all.
        if right is None:
            right_array = left_array
            self_gram = True
        else:
            right_array = self._samples(right, label="right")
            self_gram = right_array.shape == left_array.shape and bool(
                np.array_equal(left_array, right_array)
            )
        executor = self._build()
        if self.qforge_contract.analytic:
            matrix = _analytic_gram(
                executor,
                left_array,
                right_array,
                mixed=self.qforge_contract.fidelity == "density_matrix",
                self_gram=self_gram,
            )
        else:
            matrix = np.asarray(executor.overlaps(left_array, right_array), dtype=float)
            if self_gram and self.qforge_contract.enforce_psd:
                matrix = enforce_psd(matrix)
        return np.asarray(matrix, dtype=float)

    def evaluate(self, left: Any, right: Any | None = None) -> np.ndarray:
        """Alias matching the ``qiskit-machine-learning`` kernel API surface."""

        return self(left, right)


#: 1.2.4 name kept so pickles and third-party imports keep resolving.
FidelityKernel = QuantumKernel


def _analytic_gram(
    executor: Any,
    left: np.ndarray,
    right: np.ndarray,
    *,
    mixed: bool,
    self_gram: bool,
) -> np.ndarray:
    """Build the Gram matrix from encoded states rather than pairwise circuits.

    Both analytic kinds cost ``n`` circuit evaluations instead of ``n^2``,
    because the encoded states are computed once per sample and combined
    classically.
    """

    left_states = np.asarray(executor.states(left))
    right_states = left_states if self_gram else np.asarray(executor.states(right))
    overlap = left_states @ np.conjugate(right_states).T
    # Pure states: |<phi_x|phi_y>|^2.  Density matrices: sum_ab rho_x[a,b] *
    # conj(rho_y[a,b]) == Tr[rho_x rho_y], because a density matrix is
    # Hermitian.  Both are real; the imaginary part is float noise.  Both are
    # also invariant under a consistent relabelling of the computational
    # basis, which is why the two engines agree without a bit-order
    # conversion here.
    matrix = np.real(overlap) if mixed else np.abs(overlap) ** 2
    if self_gram:
        # PSD holds by construction here, so only floating-point asymmetry is
        # removed -- no spectrum is touched.
        matrix = 0.5 * (matrix + matrix.T)
    return np.asarray(matrix, dtype=float)


def _build_executor(
    ir: CircuitIR, backend: BackendSpec, contract: KernelContract
) -> Any:
    if backend.engine == Engine.PENNYLANE:
        from ._kernel_pennylane import PennyLaneKernelExecutor

        return PennyLaneKernelExecutor(ir, backend, contract)
    if backend.engine == Engine.QISKIT:
        from ._kernel_qiskit import QiskitKernelExecutor

        return QiskitKernelExecutor(ir, backend, contract)
    raise ConfigurationError(
        f"The quantum-kernel path requires a Qiskit or PennyLane backend; got "
        f"{backend.engine}."
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

#: How each resolved kernel kind is reported, per engine.  ``density_matrix``
#: is deliberately *not* advertised as a fidelity: see the module docstring.
_KIND_TEMPLATES: dict[tuple[Engine, str], str] = {
    (Engine.QISKIT, "statevector"): "fidelity_statevector_kernel",
    (Engine.QISKIT, "density_matrix"): "hilbert_schmidt_kernel[qiskit_density_matrix]",
    (Engine.PENNYLANE, "statevector"): "pennylane_fidelity_kernel[statevector]",
    (Engine.PENNYLANE, "density_matrix"): "hilbert_schmidt_kernel[pennylane_density_matrix]",
}


def build_quantum_kernel(
    ir: CircuitIR,
    *,
    backend: BackendSpec,
    fidelity: str = "auto",
) -> tuple[QuantumKernel, str]:
    """Build the kernel *backend* and *fidelity* describe, on either engine.

    Every check that can be answered from the definition alone runs before an
    engine device is constructed, and it runs identically for both engines:
    the fidelity vocabulary, the settings the chosen kernel cannot honour, the
    explicit-shots rule, the feature-map restrictions and the machine's actual
    device/GPU capabilities.
    """

    if backend.engine not in {Engine.QISKIT, Engine.PENNYLANE}:
        raise ConfigurationError(
            f"The quantum-kernel path requires a Qiskit or PennyLane backend; got "
            f"{backend.engine}."
        )
    contract = resolve_kernel_contract(backend, fidelity=fidelity)
    validate_kernel_composition(ir, engine=backend.engine)
    operations = feature_map_operations(ir, require_invertible=contract.sampled)
    feature_dimension(operations)
    # The kernel route used to skip the capability preflight entirely, so a
    # GPU request was accepted, dropped, and the run reported as a GPU run.
    device = validate_kernel_backend(backend, contract)
    kind = _KIND_TEMPLATES.get((backend.engine, contract.fidelity), "")
    if not kind:
        # ``sampler`` names the primitive that will actually run, which is a
        # property of the installed stack rather than of the definition.
        if backend.engine == Engine.QISKIT:
            from ..backends.qiskit_adapter import QiskitAdapter

            _, primitive_kind = QiskitAdapter.from_spec(backend).sampler_primitive(
                shots=contract.shots
            )
            kind = f"fidelity_quantum_kernel[{primitive_kind}]"
        else:
            kind = "pennylane_fidelity_kernel[sampler]"
    kernel = QuantumKernel(
        ir=ir,
        backend=backend,
        contract=contract,
        device=device,
        kind=kind,
    )
    return kernel, kind


def kernel_measured_wires(ir: CircuitIR) -> tuple[int, ...]:
    """Wires the compute-uncompute projector reads.

    The projector is ``|0...0><0...0|`` over the whole register, so a readout
    channel -- which is defined at the measurement boundary -- attaches to
    every wire the fidelity circuit measures, on both engines.
    """

    return tuple(range(ir.num_qubits))


def readout_active(backend: BackendSpec, contract: KernelContract) -> bool:
    """Whether a measurement-boundary confusion channel applies to this kernel."""

    return contract.sampled and is_terminal_only(backend.noise)


__all__ = [
    "NON_DETERMINISTIC_OPERATIONS",
    "NON_INVERTIBLE_OPERATIONS",
    "QuantumKernel",
    "FidelityKernel",
    "build_quantum_kernel",
    "enforce_psd",
    "feature_dimension",
    "feature_map_operations",
    "kernel_measured_wires",
    "readout_active",
]
