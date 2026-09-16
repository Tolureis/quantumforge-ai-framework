"""End-to-end verification of the catalog's executability claims.

``catalog.is_executable`` answers a static question: does a route exist for
this name?  That is necessary but not sufficient, and the catalog was counting
it as proof.  A name can resolve to a factory and still fail the moment an
encoding, a measurement, a QNN primitive and a backend have to work together
-- which is the only thing a user actually cares about.

This module answers the real question by *building and running* a minimal but
complete experiment for every entry the catalog claims is executable: the
component under test, a compatible circuit around it, a real backend, and an
actual execution or forward pass.  ``tests/test_catalog_verification.py`` runs
the sweep, so a claim that stops being true fails the build.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .catalog import catalog, executable_components, is_executable
from .errors import CapabilityError, ConfigurationError, MissingDependencyError, RegistryError
from .ir import CircuitIR
from .registry import ComponentDescriptor
from .specs import (
    AnsatzSpec,
    BackendSpec,
    CircuitSpec,
    EncodingSpec,
    Engine,
    MeasurementSpec,
    ModelMode,
    ModelSpec,
    NoiseSpec,
)

#: Components that need more than the default two data qubits.
_WIDE_COMPONENTS = frozenset({"qcnn", "quantum_convolution", "quantum_pooling"})

#: Measurements that must be sampled, and therefore need an explicit shot count.
_SAMPLED_MEASUREMENTS = frozenset({"sample", "counts"})

#: Model templates the factory routes, mapped onto the mode they need.
_TEMPLATE_MODES: dict[str, ModelMode] = {
    "quantum_classifier": ModelMode.QUANTUM_ONLY,
    "quantum_regressor": ModelMode.QUANTUM_ONLY,
    "estimator_qnn": ModelMode.QUANTUM_ONLY,
    "sampler_qnn": ModelMode.QUANTUM_ONLY,
    "variational_quantum_classifier": ModelMode.QUANTUM_ONLY,
    "variational_quantum_regressor": ModelMode.QUANTUM_ONLY,
    "quantum_only_classifier": ModelMode.QUANTUM_ONLY,
    "quantum_only_regressor": ModelMode.QUANTUM_ONLY,
    "classical_quantum_classical": ModelMode.HYBRID,
    "classical_quantum_classical_network": ModelMode.HYBRID,
    "hybrid_quantum_model": ModelMode.HYBRID,
    "quantum_sequential": ModelMode.HYBRID,
    "quantum_residual": ModelMode.HYBRID,
    "residual_hybrid_qnn": ModelMode.HYBRID,
    "ids_quantum_bottleneck": ModelMode.HYBRID,
    "quantum_bottleneck_network": ModelMode.HYBRID,
    "quantum_parallel": ModelMode.DISTRIBUTED_QNN,
    "parallel_quantum_layer_network": ModelMode.DISTRIBUTED_QNN,
    "quantum_ensemble": ModelMode.ENSEMBLE,
    "quantum_kernel_model": ModelMode.QUANTUM_KERNEL,
    "fidelity_quantum_kernel": ModelMode.QUANTUM_KERNEL,
    "statevector_quantum_kernel": ModelMode.QUANTUM_KERNEL,
    "qsvc": ModelMode.QUANTUM_KERNEL,
}


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Outcome of actually building and running one catalog entry."""

    key: str
    category: str
    engine: str
    status: str
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    @property
    def failed(self) -> bool:
        return self.status == "failed"


def _missing(engine: Engine) -> str | None:
    """Return the missing optional package for *engine*, if any."""

    import importlib.util

    if engine == Engine.QISKIT:
        for module, name in (
            ("qiskit", "qiskit"),
            ("qiskit_aer", "qiskit-aer"),
        ):
            if importlib.util.find_spec(module) is None:
                return name
    if engine == Engine.PENNYLANE and importlib.util.find_spec("pennylane") is None:
        return "pennylane"
    return None


def _backend(engine: Engine, *, shots: int | None = None, noise: NoiseSpec | None = None) -> BackendSpec:
    if engine == Engine.QISKIT:
        return BackendSpec(
            engine=engine,
            device="aer" if (noise is not None or shots is not None) else "statevector",
            shots=shots,
            noise=noise or NoiseSpec(),
            seed=7,
        )
    return BackendSpec(
        engine=engine,
        device="default.qubit",
        shots=shots,
        noise=noise or NoiseSpec(),
        seed=7,
    )


def _run(ir: CircuitIR, backend: BackendSpec, *, binary_inputs: bool = False) -> Any:
    """Execute *ir* through the real adapter for *backend*."""

    from .backends.router import BackendRouter

    adapter = BackendRouter().create(backend)
    input_count = ir.parameter_groups().get("input", 0)
    # Basis encoding is defined only on computational-basis bits, so feeding it
    # a generic angle is a harness error rather than a framework one.
    inputs = (
        np.array([index % 2 for index in range(input_count)], dtype=float)
        if binary_inputs
        else np.full(input_count, 0.35)
    )
    weights = np.full(ir.parameter_groups().get("weight", 0), 0.21)
    return adapter.run(ir, inputs=inputs, weights=weights)


def _finite(values: Any) -> bool:
    if isinstance(values, dict):
        return bool(values)
    array = np.asarray(values)
    if array.dtype.kind in {"U", "S", "O"}:
        return array.size > 0
    return bool(np.all(np.isfinite(array)))


def _circuit_spec(
    descriptor: ComponentDescriptor,
    *,
    encoding: str = "angle_encoding",
    ansatz: str = "real_amplitudes",
    measurement: str = "expectation_measurement",
    observable: str = "pauli_z",
    entanglement: str = "linear",
    qubits: int = 2,
    shots: int | None = None,
) -> CircuitSpec:
    del descriptor
    repeats = 1
    layers = 1
    if encoding == "data_reuploading_feature_map":
        repeats = 2
        layers = 2
    if encoding in {"amplitude_encoding", "state_preparation_encoding"}:
        encoding_spec = EncodingSpec(
            name=encoding, feature_dimension=2**qubits, normalize=True
        )
    elif encoding == "dense_angle_encoding":
        encoding_spec = EncodingSpec(name=encoding, feature_dimension=2 * qubits)
    else:
        encoding_spec = EncodingSpec(
            name=encoding, feature_dimension=qubits, repeats=repeats
        )
    # Topologies that are defined by an explicit edge list need that list; the
    # component is executable, it is simply not parameter-free.
    ansatz_options: dict[str, Any] = {}
    if entanglement in {"hardware", "user_defined"} or entanglement.startswith(
        ("hardware_topology", "user_defined")
    ):
        ansatz_options["custom_edges"] = tuple(
            (wire, wire + 1) for wire in range(qubits - 1)
        )
    # Likewise a Hermitian/custom observable is defined by its matrix.
    observable_matrix = None
    if observable in {"hermitian_observable", "custom_observable", "hermitian"}:
        # A matrix observable spans every measured wire, so it is 2**n square.
        dimension = 2**qubits
        observable_matrix = tuple(
            tuple(complex(float(row) if row == column else 0.0) for column in range(dimension))
            for row in range(dimension)
        )
    return CircuitSpec(
        num_qubits=qubits,
        encoding=encoding_spec,
        ansatz=AnsatzSpec(
            name=ansatz,
            layers=layers,
            entanglement=entanglement,
            options=ansatz_options,
        ),
        measurement=MeasurementSpec(
            name=measurement,
            observable=observable,
            observable_matrix=observable_matrix,
            shots=shots,
        ),
        name="verification_circuit",
    )


def _verify_circuit_component(
    descriptor: ComponentDescriptor,
    engine: Engine,
) -> VerificationResult:
    """Build and execute a circuit that actually uses *descriptor*."""

    from .circuits.factory import CircuitFactory

    key = descriptor.key
    qubits = 4 if key in _WIDE_COMPONENTS else 2
    kwargs: dict[str, Any] = {"qubits": qubits}
    shots: int | None = None
    if descriptor.category == "encoding":
        kwargs["encoding"] = key
    elif descriptor.category in {"ansatz", "tensor_network"}:
        kwargs["ansatz"] = key
    elif descriptor.category == "entanglement":
        kwargs["entanglement"] = key
    elif descriptor.category == "measurement":
        from .circuits.factory import SUPPORTED_MEASUREMENTS

        kwargs["measurement"] = key
        if SUPPORTED_MEASUREMENTS[key] in _SAMPLED_MEASUREMENTS:
            shots = 64
            kwargs["shots"] = shots
    elif descriptor.category == "observable":
        kwargs["observable"] = key
    spec = _circuit_spec(descriptor, **kwargs)
    noise: NoiseSpec | None = None
    if descriptor.category == "noise":
        from .noise import canonical_noise_name

        channel = canonical_noise_name(key)
        if channel == "readout":
            kwargs["measurement"] = "count_measurement"
            shots = 64
            kwargs["shots"] = shots
            spec = _circuit_spec(descriptor, **kwargs)
            noise = NoiseSpec(name=key, readout_probability=0.1)
        elif channel == "thermal_relaxation":
            noise = NoiseSpec(name=key, t1=60.0, t2=40.0, gate_time=5.0)
        elif channel in {"none", "shot_noise"}:
            noise = NoiseSpec(name=key)
        else:
            noise = NoiseSpec(name=key, probability=0.05)
    ir = CircuitFactory().build(spec)
    backend = _backend(engine, shots=shots, noise=noise)
    result = _run(
        ir,
        backend,
        binary_inputs=kwargs.get("encoding") in {"basis_encoding", "binary_encoding"},
    )
    if not _finite(result.values):
        return VerificationResult(
            key, descriptor.category, str(engine), "failed", "result is not finite"
        )
    return VerificationResult(
        key,
        descriptor.category,
        str(engine),
        "ok",
        f"{len(ir.operations)} operations executed",
    )


def _verify_model_component(
    descriptor: ComponentDescriptor,
    engine: Engine,
) -> VerificationResult:
    """Build the routed model and run a real forward pass through it."""

    import importlib.util

    from .circuits.factory import CircuitFactory
    from .models.factory import ModelFactory

    template = ModelFactory.resolve_template(descriptor.key)
    if template is None:
        return VerificationResult(
            descriptor.key,
            descriptor.category,
            str(engine),
            "failed",
            "no model_factory route",
        )
    mode = _TEMPLATE_MODES.get(descriptor.key) or _TEMPLATE_MODES[template]
    if mode == ModelMode.QUANTUM_KERNEL:
        spec = _circuit_spec(descriptor)
        ir = CircuitFactory().build(spec)
        model = ModelFactory().build(
            ir,
            ModelSpec(mode=mode, template=descriptor.key),
            _backend(engine),
        )
        samples = np.array([[0.1, 0.2], [1.3, 1.4], [0.15, 0.25], [1.35, 1.45]])
        labels = np.array([0, 1, 0, 1])
        model.fit(samples, labels)
        predictions = np.asarray(model.predict(samples))
        if predictions.shape[0] != samples.shape[0]:
            return VerificationResult(
                descriptor.key, descriptor.category, str(engine), "failed", "prediction shape is wrong"
            )
        return VerificationResult(
            descriptor.key, descriptor.category, str(engine), "ok", "kernel fit/predict"
        )
    if importlib.util.find_spec("torch") is None:
        return VerificationResult(
            descriptor.key, descriptor.category, str(engine), "skipped", "torch is not installed"
        )
    import torch

    sampler_template = template == "sampler_qnn"
    measurement = "probability_measurement" if sampler_template else "expectation_measurement"
    shots = 128 if (sampler_template and engine == Engine.QISKIT) else None
    spec = _circuit_spec(descriptor, measurement=measurement)
    ir = CircuitFactory().build(spec)
    if mode == ModelMode.QUANTUM_ONLY:
        _, _, width = ModelFactory._dimensions(ir, ModelSpec(mode=mode))
        model_spec = ModelSpec(mode=mode, template=descriptor.key, output_dimension=width)
    else:
        model_spec = ModelSpec(
            mode=mode, template=descriptor.key, input_dimension=2, output_dimension=2
        )
    model = ModelFactory().build(ir, model_spec, _backend(engine, shots=shots))
    features = ir.parameter_groups().get("input", 2)
    inputs = torch.zeros((2, features if mode == ModelMode.QUANTUM_ONLY else 2))
    output = model(inputs)
    if not torch.isfinite(output).all():
        return VerificationResult(
            descriptor.key, descriptor.category, str(engine), "failed", "output is not finite"
        )
    return VerificationResult(
        descriptor.key,
        descriptor.category,
        str(engine),
        "ok",
        f"forward {tuple(output.shape)}",
    )


def verify_component(key: str, *, engine: Engine | str) -> VerificationResult:
    """Build and run *key* end to end on *engine*.

    Returns a result rather than raising: a sweep wants every outcome, not the
    first failure.
    """

    resolved = Engine(engine)
    descriptor = catalog.get(key)
    if resolved not in descriptor.engines:
        return VerificationResult(
            descriptor.key,
            descriptor.category,
            str(resolved),
            "skipped",
            "the catalog does not declare this engine",
        )
    if not is_executable(descriptor.key):
        return VerificationResult(
            descriptor.key,
            descriptor.category,
            str(resolved),
            "skipped",
            "the catalog does not mark it as executable",
        )
    missing = _missing(resolved)
    if missing is not None:
        return VerificationResult(
            descriptor.key, descriptor.category, str(resolved), "skipped", f"{missing} is not available"
        )
    try:
        if descriptor.implementation == "model_factory":
            return _verify_model_component(descriptor, resolved)
        return _verify_circuit_component(descriptor, resolved)
    except MissingDependencyError as exc:
        return VerificationResult(
            descriptor.key, descriptor.category, str(resolved), "skipped", str(exc)[:160]
        )
    except (CapabilityError, ConfigurationError, RegistryError) as exc:
        return VerificationResult(
            descriptor.key,
            descriptor.category,
            str(resolved),
            "failed",
            f"{type(exc).__name__}: {exc}"[:240],
        )
    except Exception as exc:  # pragma: no cover - unexpected engine failure
        return VerificationResult(
            descriptor.key,
            descriptor.category,
            str(resolved),
            "failed",
            f"{type(exc).__name__}: {exc}"[:240],
        )


def verify_catalog(
    *,
    engines: Sequence[Engine] = (Engine.QISKIT, Engine.PENNYLANE),
    categories: Iterable[str] | None = None,
) -> list[VerificationResult]:
    """Verify every entry the catalog claims is executable, on every engine."""

    wanted = set(categories) if categories is not None else None
    results: list[VerificationResult] = []
    for descriptor in executable_components():
        if wanted is not None and descriptor.category not in wanted:
            continue
        for engine in engines:
            results.append(verify_component(descriptor.key, engine=engine))
    return results


# ---------------------------------------------------------------------------
# Composition verification
# ---------------------------------------------------------------------------
#
# Verifying components one at a time proves nothing about combining them: each
# component was checked against a baseline chosen to suit it.  The sweep below
# takes the cross product of the executable encodings and the routed model
# templates on both engines, and checks the *declared capabilities* against
# what actually happens -- a build that the catalog says should work must work,
# and one the catalog says cannot must be refused with a ConfigurationError
# rather than crashing somewhere inside an engine.

#: Model templates that exercise each distinct composition path once.
_COMPOSITION_TEMPLATES: tuple[str, ...] = (
    "quantum_classifier",
    "classical_quantum_classical",
    "quantum_kernel_model",
)


def _composition_supported(
    profile: Any,
    *,
    engine: Engine,
    mode: ModelMode,
) -> tuple[bool, str]:
    """Return whether this composition is expected to build, and why not.

    Derived from the declared :class:`~qforge_ai.composition.EncodingProfile`
    only -- never from the outcome -- so the sweep is a genuine check of the
    catalog's claims rather than a recording of whatever happened.
    """

    from .composition import CLASSICAL_ENCODER_MODES

    if not profile.encodes_data:
        return False, "encodes_data"
    if mode == ModelMode.QUANTUM_KERNEL:
        if profile.trainable_encoding:
            return False, "trainable_encoding"
        if not profile.continuous_input:
            return False, "continuous_input"
        if engine == Engine.QISKIT and not profile.symbolic_parameters:
            return False, "symbolic_parameters"
        return True, ""
    if engine == Engine.QISKIT and not profile.symbolic_parameters:
        return False, "symbolic_parameters"
    if mode in CLASSICAL_ENCODER_MODES:
        if not profile.continuous_input:
            return False, "continuous_input"
        if profile.requires_normalized_input:
            return False, "requires_normalized_input"
    return True, ""


def verify_composition(
    encoding: str,
    template: str,
    *,
    engine: Engine | str,
) -> VerificationResult:
    """Build *encoding* inside *template* on *engine* and check the outcome.

    A composition the catalog's capabilities say is supported must build and
    run; one they say is unsupported must be refused by
    :func:`qforge_ai.composition.validate_composition` -- not by an engine
    error five layers down, and never by silently producing a model that fails
    on its first forward pass.
    """

    import importlib.util

    from .circuits.factory import CircuitFactory
    from .composition import encoding_profile
    from .models.factory import ModelFactory

    resolved = Engine(engine)
    descriptor = catalog.get(encoding)
    label = f"{encoding}+{template}"
    missing = _missing(resolved)
    if missing is not None:
        return VerificationResult(label, "composition", str(resolved), "skipped", f"{missing} is not available")
    mode = _TEMPLATE_MODES[template]
    if mode != ModelMode.QUANTUM_KERNEL and importlib.util.find_spec("torch") is None:
        return VerificationResult(
            label, "composition", str(resolved), "skipped", "torch is not installed"
        )
    if mode == ModelMode.QUANTUM_KERNEL and importlib.util.find_spec("sklearn") is None:
        return VerificationResult(
            label, "composition", str(resolved), "skipped", "scikit-learn is not installed"
        )

    measurement = "expectation_measurement"
    spec = _circuit_spec(descriptor, encoding=encoding, measurement=measurement)
    try:
        ir = CircuitFactory().build(spec)
    except (CapabilityError, ConfigurationError, RegistryError) as exc:
        return VerificationResult(
            label, "composition", str(resolved), "failed", f"the circuit could not be built: {exc}"[:200]
        )
    profile = encoding_profile(ir)
    supported, reason = _composition_supported(profile, engine=resolved, mode=mode)

    if mode == ModelMode.QUANTUM_ONLY:
        _, _, width = ModelFactory._dimensions(ir, ModelSpec(mode=mode))
        model_spec = ModelSpec(mode=mode, template=template, output_dimension=width)
    else:
        model_spec = ModelSpec(
            mode=mode, template=template, input_dimension=2, output_dimension=2
        )
    try:
        model = ModelFactory().build(ir, model_spec, _backend(resolved))
    except ConfigurationError as exc:
        if supported:
            return VerificationResult(
                label,
                "composition",
                str(resolved),
                "failed",
                f"the catalog supports it but the setup was refused: {exc}"[:220],
            )
        return VerificationResult(
            label, "composition", str(resolved), "ok", f"refused as expected ({reason})"
        )
    except MissingDependencyError as exc:
        return VerificationResult(
            label, "composition", str(resolved), "skipped", str(exc)[:160]
        )
    except Exception as exc:  # noqa: BLE001 - an engine error is the failure mode
        return VerificationResult(
            label,
            "composition",
            str(resolved),
            "failed",
            f"engine error ({type(exc).__name__}): {exc}"[:220],
        )
    if not supported:
        return VerificationResult(
            label,
            "composition",
            str(resolved),
            "failed",
            f"the catalog does not support it ({reason}) but the setup silently succeeded",
        )
    return _exercise_composition(model, profile, label, mode, resolved)


def _composition_samples(profile: Any, rows: int) -> np.ndarray:
    """Return sample data every encoding under test can legally consume.

    The harness has to satisfy each encoding's own input domain -- bits for
    basis encoding, a non-degenerate vector for amplitude encoding, the right
    feature width for a dense map.  Feeding one shape to all of them measures
    the harness, not the framework.
    """

    width = max(int(profile.input_features), 1)
    if not profile.continuous_input:
        return np.asarray(
            [[(row + column) % 2 for column in range(width)] for row in range(rows)],
            dtype=float,
        )
    base = np.linspace(0.15, 1.25, width, dtype=float)
    return np.asarray([base + 0.37 * row for row in range(rows)], dtype=float)


def _exercise_composition(
    model: Any,
    profile: Any,
    label: str,
    mode: ModelMode,
    engine: Engine,
) -> VerificationResult:
    """Run a real forward pass / fit so "it built" is not mistaken for "it works"."""

    try:
        if mode == ModelMode.QUANTUM_KERNEL:
            samples = _composition_samples(profile, 4)
            model.fit(samples, np.array([0, 1, 0, 1]))
            if np.asarray(model.predict(samples)).shape[0] != samples.shape[0]:
                return VerificationResult(
                    label, "composition", str(engine), "failed", "prediction shape is wrong"
                )
            # ``evaluate`` is the API surface both engines share.
            gram = np.asarray(model.qforge_quantum_kernel.evaluate(samples))
            if not np.allclose(gram, gram.T, atol=1e-8):
                return VerificationResult(
                    label, "composition", str(engine), "failed", "the Gram matrix is not symmetric"
                )
            return VerificationResult(label, "composition", str(engine), "ok", "kernel fit/predict")
        import torch

        rows = 2
        if mode == ModelMode.QUANTUM_ONLY:
            samples = _composition_samples(profile, rows)
        else:
            # A hybrid template's classical encoder owns the circuit's input
            # width, so the model itself takes a plain 2-feature vector.
            samples = np.asarray([[0.4, -0.6], [-0.2, 0.9]], dtype=float)
        output = model(torch.as_tensor(samples, dtype=torch.float32))
        if not torch.isfinite(output).all():
            return VerificationResult(
                label, "composition", str(engine), "failed", "output is not finite"
            )
        return VerificationResult(
            label, "composition", str(engine), "ok", f"forward {tuple(output.shape)}"
        )
    except Exception as exc:  # noqa: BLE001
        return VerificationResult(
            label,
            "composition",
            str(engine),
            "failed",
            f"execution error ({type(exc).__name__}): {exc}"[:220],
        )


def verify_compositions(
    *,
    engines: Sequence[Engine] = (Engine.QISKIT, Engine.PENNYLANE),
    templates: Sequence[str] = _COMPOSITION_TEMPLATES,
) -> list[VerificationResult]:
    """Sweep every executable encoding against every composition template."""

    encodings = [
        descriptor.key
        for descriptor in executable_components()
        if descriptor.category == "encoding"
    ]
    results: list[VerificationResult] = []
    for encoding in sorted(encodings):
        for template in templates:
            for engine in engines:
                results.append(verify_composition(encoding, template, engine=engine))
    return results


def verification_report() -> dict[str, Any]:
    """Return counts of what was actually built and run, per engine."""

    results = verify_catalog()
    summary: dict[str, Any] = {"total": len(results), "ok": 0, "skipped": 0, "failed": 0}
    failures: list[dict[str, str]] = []
    for item in results:
        summary[item.status] = summary.get(item.status, 0) + 1
        if item.failed:
            failures.append(
                {
                    "key": item.key,
                    "category": item.category,
                    "engine": item.engine,
                    "detail": item.detail,
                }
            )
    summary["failures"] = failures
    return summary


__all__ = [
    "VerificationResult",
    "verify_component",
    "verify_catalog",
    "verify_composition",
    "verify_compositions",
    "verification_report",
]
