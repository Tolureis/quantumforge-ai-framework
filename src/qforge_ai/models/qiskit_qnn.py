"""Qiskit Machine Learning factories kept behind optional imports."""

from __future__ import annotations

import inspect
from dataclasses import replace
from functools import partial
from typing import Any

import numpy as np

from ..backends._optional import require
from ..backends.qiskit_adapter import (
    QiskitAdapter,
    _full_qiskit_observable,
)
from ..errors import ConfigurationError
from ..ir import CircuitIR
from ..specs import BackendSpec, Engine, GradientMethod, validate_output_shape
from .kernel import build_quantum_kernel as build_shared_quantum_kernel
from .kernel_contract import normalize_fidelity


def _partition_parameters(circuit: Any) -> tuple[list[Any], list[Any]]:
    def parameter_index(parameter: Any) -> int:
        index = getattr(parameter, "index", None)
        if index is not None:
            return int(index)
        text = str(parameter)
        return int(text.rsplit("[", 1)[-1].rstrip("]"))

    inputs = sorted(
        (param for param in circuit.parameters if str(param).startswith("input")),
        key=parameter_index,
    )
    weights = sorted(
        (param for param in circuit.parameters if str(param).startswith("weight")),
        key=parameter_index,
    )
    return inputs, weights


def build_estimator_qnn(
    ir: CircuitIR,
    *,
    backend: BackendSpec | None = None,
    observable: Any | None = None,
    input_gradients: bool = True,
) -> Any:
    # Configuration is validated *before* the optional import, so a mistake in
    # the experiment definition reports itself as a ConfigurationError on a
    # dependency-light install instead of being masked by a
    # MissingDependencyError for qiskit-machine-learning.
    _require_qnn_measurement(ir, expected="expectation", qnn="EstimatorQNN")
    resolved_backend = _effective_backend(ir, backend)
    _reject_readout_noise(resolved_backend, qnn="EstimatorQNN")
    qml = require(
        "qiskit_machine_learning.neural_networks",
        package="qiskit-machine-learning",
        extra="qiskit",
        feature="EstimatorQNN",
    )
    adapter = QiskitAdapter.from_spec(resolved_backend)
    circuit = adapter.compile(ir)
    input_params, weight_params = _partition_parameters(circuit)
    estimator, precision, primitive_kind = adapter.estimator_primitive()
    gradient = _build_estimator_gradient(estimator, resolved_backend)
    if observable is None and ir.measurements and ir.measurements[0].observables:
        quantum_info = require(
            "qiskit.quantum_info",
            package="qiskit",
            extra="qiskit",
            feature="EstimatorQNN observables",
        )
        observable = [
            _full_qiskit_observable(quantum_info, item, ir.num_qubits)
            for item in ir.measurements[0].observables
        ]
    elif observable is not None:
        _validate_observable_override(observable, num_qubits=ir.num_qubits)
    kwargs: dict[str, Any] = {
        "circuit": circuit,
        "estimator": estimator,
        "input_params": input_params,
        "weight_params": weight_params,
        "gradient": gradient,
        "input_gradients": input_gradients,
        "default_precision": precision,
    }
    if observable is not None:
        kwargs["observables"] = observable
    qnn = qml.EstimatorQNN(**kwargs)
    qnn.qforge_backend_spec = resolved_backend
    qnn.qforge_primitive_kind = primitive_kind
    qnn.qforge_shots = resolved_backend.shots
    qnn.qforge_gradient_method = str(resolved_backend.gradient)
    return qnn


def build_sampler_qnn(
    ir: CircuitIR,
    *,
    backend: BackendSpec | None = None,
    interpret: Any | None = None,
    output_shape: int | tuple[int, ...] | None = None,
    input_gradients: bool = True,
) -> Any:
    # Every configuration check runs before the optional import.  The shot
    # contract below is a property of the experiment definition, not of the
    # installed packages, so on a ``pip install -e '.[dev]'` machine it used to
    # be reported as "qiskit-machine-learning is missing" -- which is why the
    # core CI job could not stay green.
    _require_qnn_measurement(ir, expected="probability", qnn="SamplerQNN")
    resolved_backend = _effective_backend(ir, backend)
    if resolved_backend.shots is None:
        # ``sampler_primitive`` used to substitute 1024 shots here, so an
        # experiment that specified no shots ran finite-shot on Qiskit while
        # running analytically on PennyLane -- a cross-engine difference the
        # spec never asked for and the result metadata never mentioned.
        raise ConfigurationError(
            "SamplerQNN relies on sampling and cannot run analytically; rather than falling "
            "back to 1024 shots silently it requires an explicit shot count. Set BackendSpec.shots "
            "or Measurement.shots (for analytic probability in PennyLane use "
            "backend.engine='pennylane')."
        )
    if interpret is not None and not callable(interpret):
        raise ConfigurationError("SamplerQNN interpret must be a callable.")
    if interpret is not None and output_shape is None:
        raise ConfigurationError(
            "sampler_output_shape is required when a custom SamplerQNN interpret is used."
        )
    qml = require(
        "qiskit_machine_learning.neural_networks",
        package="qiskit-machine-learning",
        extra="qiskit",
        feature="SamplerQNN",
    )
    adapter = QiskitAdapter.from_spec(resolved_backend)
    measurement_wires = (
        ir.measurements[0].wires if ir.measurements else ir.data_wires
    )
    circuit = adapter.compile(
        ir,
        add_measurements=True,
        measurement_wires=measurement_wires,
    )
    input_params, weight_params = _partition_parameters(circuit)
    sampler, primitive_kind = adapter.sampler_primitive()
    gradient = _build_sampler_gradient(sampler, resolved_backend)
    default_output_shape = 2 ** len(measurement_wires)
    if interpret is None:
        if output_shape is not None and validate_output_shape(output_shape) != (
            default_output_shape,
        ):
            raise ConfigurationError(
                "If sampler_output_shape differs from the measurement space, a custom interpret "
                "function must be given as well."
            )
        interpret = partial(_reverse_bits, width=len(measurement_wires))
        if output_shape is None:
            output_shape = default_output_shape
    if output_shape is not None:
        validate_output_shape(output_shape)
    kwargs: dict[str, Any] = {
        "circuit": circuit,
        "sampler": sampler,
        "input_params": input_params,
        "weight_params": weight_params,
        "gradient": gradient,
        "input_gradients": input_gradients,
    }
    if interpret is not None:
        kwargs["interpret"] = interpret
    if output_shape is not None:
        kwargs["output_shape"] = output_shape
    qnn = qml.SamplerQNN(**kwargs)
    qnn.qforge_backend_spec = resolved_backend
    qnn.qforge_primitive_kind = primitive_kind
    qnn.qforge_measurement_wires = tuple(measurement_wires)
    qnn.qforge_shots = resolved_backend.shots
    qnn.qforge_gradient_method = str(resolved_backend.gradient)
    return qnn


def _validate_observable_override(observable: Any, *, num_qubits: int) -> None:
    """Check a user-supplied observable against the circuit width.

    A mismatch used to surface much later as an opaque matmul error inside the
    first forward pass; here it names the actual problem.
    """

    items = observable if isinstance(observable, (list, tuple)) else [observable]
    if not items:
        raise ConfigurationError("An observable override cannot be an empty sequence.")
    for index, item in enumerate(items):
        width = getattr(item, "num_qubits", None)
        if width is not None and int(width) != num_qubits:
            raise ConfigurationError(
                f"observable[{index}] is defined on {width} qubits; the circuit has {num_qubits} "
                "qubits. EstimatorQNN observables must span the full system width."
            )


def _reject_readout_noise(backend: BackendSpec, *, qnn: str) -> None:
    """Refuse a readout channel on a primitive that performs no measurement.

    ``EstimatorQNN`` evaluates ``Tr[rho O]`` on an unmeasured circuit, so the
    measurement-boundary confusion channel has nothing to attach to and was
    dropped without a word: the same ``NoiseSpec`` changed the result on the
    direct execution path (which rejects the combination outright) and changed
    nothing here.
    """

    from ..noise import is_terminal_only

    if is_terminal_only(backend.noise):
        raise ConfigurationError(
            f"{qnn} computes an expectation value without measuring; readout error is defined "
            "at the measurement boundary and cannot be applied on this path. It is rejected "
            "rather than dropped silently: choose a sampling-based path (SamplerQNN or a "
            "counts/sample measurement), or use a gate channel."
        )


def _require_qnn_measurement(ir: CircuitIR, *, expected: str, qnn: str) -> None:
    ir.validate()
    if not ir.measurements:
        raise ConfigurationError(f"{qnn} requires an explicit {expected} measurement.")
    actual = ir.measurements[0].kind
    if actual != expected:
        raise ConfigurationError(
            f"The {qnn} measurement contract requires '{expected}'; got '{actual}'."
        )


def _build_estimator_gradient(estimator: Any, backend: BackendSpec) -> Any:
    gradients = require(
        "qiskit_machine_learning.gradients",
        package="qiskit-machine-learning",
        extra="qiskit",
        feature="EstimatorQNN gradient",
    )
    method = backend.gradient
    options = dict(backend.options.get("gradient_options", {}))
    if method in {GradientMethod.AUTO, GradientMethod.PARAMETER_SHIFT}:
        _reject_unused_gradient_options(options, method)
        return gradients.ParamShiftEstimatorGradient(estimator=estimator)
    if method == GradientMethod.SPSA:
        epsilon, batch_size = _spsa_options(options)
        return gradients.SPSAEstimatorGradient(
            estimator=estimator,
            epsilon=epsilon,
            batch_size=batch_size,
            seed=backend.seed,
        )
    if method == GradientMethod.FINITE_DIFF:
        return _finite_diff_estimator_gradient(gradients, estimator, options)
    raise ConfigurationError(
        f"Qiskit EstimatorQNN gradient method not supported: {method}. "
        "Use auto, parameter_shift, finite_diff or spsa."
    )


def _build_sampler_gradient(sampler: Any, backend: BackendSpec) -> Any:
    gradients = require(
        "qiskit_machine_learning.gradients",
        package="qiskit-machine-learning",
        extra="qiskit",
        feature="SamplerQNN gradient",
    )
    method = backend.gradient
    options = dict(backend.options.get("gradient_options", {}))
    if method in {GradientMethod.AUTO, GradientMethod.PARAMETER_SHIFT}:
        _reject_unused_gradient_options(options, method)
        return gradients.ParamShiftSamplerGradient(sampler=sampler)
    if method == GradientMethod.SPSA:
        epsilon, batch_size = _spsa_options(options)
        return gradients.SPSASamplerGradient(
            sampler=sampler,
            epsilon=epsilon,
            batch_size=batch_size,
            seed=backend.seed,
        )
    raise ConfigurationError(
        f"Qiskit SamplerQNN gradient method not supported: {method}. "
        "Use auto, parameter_shift or spsa."
    )


def _reject_unused_gradient_options(options: dict[str, Any], method: Any) -> None:
    """Fail when gradient_options were supplied to a method that ignores them.

    Parameter-shift takes no tuning knobs, so an ``epsilon`` meant for
    finite-diff used to be accepted and quietly discarded.
    """

    if options:
        raise ConfigurationError(
            f"gradient={method} uses no gradient_options value; the given keys are "
            f"rejected rather than dropped silently: {sorted(options)}. "
            "For epsilon/batch_size choose gradient='finite_diff' or 'spsa'."
        )


def _spsa_options(options: dict[str, Any]) -> tuple[float, int]:
    unknown = sorted(set(options) - {"epsilon", "batch_size", "method"})
    if unknown:
        raise ConfigurationError(f"Unknown Qiskit gradient_options: {unknown}")
    epsilon = float(options.get("epsilon", 1e-6))
    batch_size = int(options.get("batch_size", 1))
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ConfigurationError("SPSA gradient epsilon must be positive and finite.")
    if batch_size < 1:
        raise ConfigurationError("SPSA gradient batch_size must be at least 1.")
    return epsilon, batch_size


def _finite_diff_estimator_gradient(
    gradients: Any,
    estimator: Any,
    options: dict[str, Any],
) -> Any:
    providers = require(
        "qiskit.providers",
        package="qiskit",
        extra="qiskit",
        feature="finite-difference EstimatorQNN gradient",
    )
    unknown = sorted(set(options) - {"epsilon", "method"})
    if unknown:
        raise ConfigurationError(f"Unknown finite-diff gradient_options: {unknown}")
    epsilon = float(options.get("epsilon", 1e-6))
    method = str(options.get("method", "central"))
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ConfigurationError("Finite-diff epsilon must be positive and finite.")
    if method not in {"central", "forward", "backward"}:
        raise ConfigurationError("Finite-diff method must be central, forward or backward.")

    class FiniteDiffEstimatorGradient(gradients.BaseEstimatorGradient):
        """V2-primitive finite differences compatible with QML 0.9 results."""

        def __init__(self) -> None:
            super().__init__(estimator=estimator)

        def _run(
            self,
            circuits: Any,
            observables: Any,
            parameter_values: Any,
            parameters: Any,
            **run_options: Any,
        ) -> Any:
            unsupported = sorted(set(run_options) - {"precision"})
            if unsupported:
                raise ValueError(f"Finite-diff run options not supported: {unsupported}")
            primitive_kwargs = (
                {"precision": run_options["precision"]}
                if run_options.get("precision") is not None
                else {}
            )
            computed: list[np.ndarray] = []
            metadata: list[dict[str, Any]] = []
            for circuit, observable, values, requested in zip(
                circuits, observables, parameter_values, parameters, strict=True
            ):
                base = np.asarray(values, dtype=float)
                ordered = list(circuit.parameters)
                indices = [ordered.index(parameter) for parameter in requested]
                pubs: list[tuple[Any, Any, np.ndarray]] = []
                for index in indices:
                    if method in {"central", "forward"}:
                        plus = base.copy()
                        plus[index] += epsilon
                        pubs.append((circuit, observable, plus))
                    if method in {"central", "backward"}:
                        minus = base.copy()
                        minus[index] -= epsilon
                        pubs.append((circuit, observable, minus))
                    if method in {"forward", "backward"}:
                        pubs.append((circuit, observable, base))
                results = self._estimator.run(pubs, **primitive_kwargs).result()
                values_iter = iter(
                    float(np.asarray(item.data.evs).reshape(-1)[0]) for item in results
                )
                derivative: list[float] = []
                for _ in indices:
                    first = next(values_iter)
                    second = next(values_iter)
                    if method == "central":
                        derivative.append((first - second) / (2.0 * epsilon))
                    elif method == "forward":
                        derivative.append((first - second) / epsilon)
                    else:
                        derivative.append((second - first) / epsilon)
                computed.append(np.asarray(derivative, dtype=float))
                metadata.append(
                    {"parameters": tuple(requested), "method": method, "epsilon": epsilon}
                )
            return gradients.EstimatorGradientResult(
                gradients=computed,
                metadata=metadata,
                # StatevectorEstimator intentionally has no public ``options``
                # attribute, so BaseEstimatorGradient._get_local_options cannot
                # be used on every supported V2 primitive.
                options=providers.Options(**run_options),
            )

    return FiniteDiffEstimatorGradient()


def _reverse_bits(value: int, *, width: int) -> int:
    return int(f"{value:0{width}b}"[::-1], 2)


def _effective_backend(ir: CircuitIR, backend: BackendSpec | None) -> BackendSpec:
    resolved = backend or BackendSpec(engine=Engine.QISKIT, device="statevector")
    measurement_shots = ir.measurements[0].shots if ir.measurements else None
    if resolved.shots is None and measurement_shots is not None:
        return replace(resolved, shots=measurement_shots)
    return resolved


def feature_map_circuit(ir: CircuitIR, adapter: QiskitAdapter) -> Any:
    """Compile the encoding-only sub-circuit used as a kernel feature map.

    Compiled without in-circuit noise: the fidelity kernel's compute-uncompute
    step inverts this circuit, and a Kraus channel has no inverse.  The noisy
    variants build their own circuits -- see
    :mod:`qforge_ai.models._kernel_qiskit`.
    """

    from ._kernel_qiskit import feature_map_circuit as compile_feature_map

    return compile_feature_map(ir, adapter, include_noise=False)


def build_quantum_kernel(
    ir: CircuitIR,
    *,
    backend: BackendSpec | None = None,
    fidelity: str = "auto",
) -> tuple[Any, str]:
    """Build the quantum kernel dictated by *backend*.

    The implementation is :func:`qforge_ai.models.kernel.build_quantum_kernel`,
    shared with the PennyLane route.  Delegating rather than wrapping
    ``qiskit-machine-learning``'s kernel classes is what closed the last three
    cross-engine gaps on this path:

    * ``FidelityQuantumKernel`` never evaluated ``K(x, x)`` -- it wrote 1.0 --
      so a noisy sampled kernel reported a diagonal of 1 against PennyLane's
      0.32 from the same definition;
    * noise arrived through an Aer ``NoiseModel``, keyed by basis-gate name
      after transpilation, and therefore could not be placed where PennyLane
      places it;
    * ``density_matrix`` raised ``CapabilityError``, so an ``auto`` kernel with
      a noise channel and no shots ran on one engine and refused on the other.
    """

    resolved = _effective_backend(ir, backend)
    if resolved.engine != Engine.QISKIT:
        raise ConfigurationError(
            f"The core quantum-kernel path requires a Qiskit BackendSpec; got "
            f"{resolved.engine}."
        )
    require(
        "qiskit_machine_learning.kernels",
        package="qiskit-machine-learning",
        extra="qiskit",
        feature="quantum kernel",
    )
    return build_shared_quantum_kernel(ir, backend=resolved, fidelity=fidelity)


def _svc_parameter_names() -> frozenset[str]:
    algorithms = require(
        "qiskit_machine_learning.algorithms",
        package="qiskit-machine-learning",
        extra="qiskit",
        feature="QSVC",
    )
    names: set[str] = set()
    getter = getattr(algorithms.QSVC, "_get_param_names", None)
    if callable(getter):
        try:
            names.update(getter())
        except Exception:  # pragma: no cover - sklearn API drift
            names.clear()
    if not names:
        for klass in algorithms.QSVC.__mro__:
            init = klass.__dict__.get("__init__")
            if init is None:
                continue
            for name, parameter in inspect.signature(init).parameters.items():
                if name != "self" and parameter.kind is not parameter.VAR_KEYWORD:
                    names.add(name)
    # ``kernel`` is owned by the quantum kernel; ``quantum_kernel`` is supplied
    # by this factory.
    return frozenset(names - {"self", "kernel", "quantum_kernel"})


def build_qsvc(
    ir: CircuitIR,
    *,
    backend: BackendSpec | None = None,
    fidelity: str = "auto",
    **kwargs: Any,
) -> Any:
    algorithms = require(
        "qiskit_machine_learning.algorithms",
        package="qiskit-machine-learning",
        extra="qiskit",
        feature="QSVC",
    )
    # ``fidelity`` is a first-class kernel setting, so ``ModelSpec.options``
    # is allowed to carry it.  Forwarding options blindly made it collide with
    # the keyword the factory already passes, and the user saw a bare
    # ``TypeError: got multiple values for keyword argument 'fidelity'``.
    if "fidelity" in kwargs:
        requested = normalize_fidelity(kwargs.pop("fidelity"))
        if normalize_fidelity(fidelity) not in {"auto", requested}:
            raise ConfigurationError(
                f"Kernel fidelity is requested by both the template ({fidelity!r}) and options ({requested!r}) "
                "with different values; choose one."
            )
        fidelity = str(requested)
    allowed = _svc_parameter_names()
    unknown = sorted(set(kwargs) - allowed)
    if unknown:
        raise ConfigurationError(
            f"QSVC does not accept these options keys: {unknown}. "
            f"Valid ones: {sorted(allowed | {'fidelity'})}"
        )
    kernel, kernel_kind = build_quantum_kernel(ir, backend=backend, fidelity=fidelity)
    qsvc = algorithms.QSVC(quantum_kernel=kernel, **kwargs)
    resolved = _effective_backend(ir, backend)
    qsvc.qforge_backend_spec = resolved
    qsvc.qforge_kernel_kind = kernel_kind
    qsvc.qforge_shots = resolved.shots
    # Same accessor as the PennyLane route: the kernel object is what a caller
    # needs to inspect a Gram matrix, and it used to be reachable on one engine
    # only.
    qsvc.qforge_quantum_kernel = kernel
    return qsvc


def observable_output_count(observable: Any) -> int:
    """Return how many expectation values *observable* produces.

    ``EstimatorQNN`` treats a bare operator as one observable and a sequence
    as several.  A multi-term ``SparsePauliOp`` is a single observable even
    though it has a length, so length is only consulted for real sequences.
    """

    if observable is None:
        return 0
    if isinstance(observable, (list, tuple)):
        return len(observable)
    return 1


def as_torch_layer(qnn: Any, *, initial_weights: Any | None = None) -> Any:
    connectors = require(
        "qiskit_machine_learning.connectors",
        package="qiskit-machine-learning",
        extra="qiskit",
        feature="TorchConnector",
    )
    return connectors.TorchConnector(qnn, initial_weights=initial_weights)


__all__ = [
    "build_estimator_qnn",
    "build_sampler_qnn",
    "build_quantum_kernel",
    "build_qsvc",
    "feature_map_circuit",
    "observable_output_count",
    "as_torch_layer",
]
