"""High-level model-template routing for Qiskit, PennyLane and PyTorch."""

from __future__ import annotations

from math import prod
from typing import Any

from ..backends._optional import require
from ..backends.gradients import pennylane_diff_method
from ..composition import validate_composition
from ..errors import CapabilityError, ConfigurationError
from ..ir import CircuitIR
from ..registry import normalize_key
from ..runtime import seed_everything
from ..specs import BackendSpec, Engine, ModelMode, ModelSpec, validate_output_shape
from .pennylane_qnn import as_torch_layer as pennylane_torch_layer
from .pennylane_qnn import build_pennylane_qsvc
from .qiskit_qnn import as_torch_layer as qiskit_torch_layer
from .qiskit_qnn import (
    build_estimator_qnn,
    build_qsvc,
    build_sampler_qnn,
    observable_output_count,
)
from .templates import QuantumParallel

_AUTO_TEMPLATE = {
    ModelMode.QUANTUM_ONLY: "quantum_classifier",
    ModelMode.HYBRID: "classical_quantum_classical",
    ModelMode.QUANVOLUTION: "classical_quantum_classical",
    ModelMode.QUANTUM_KERNEL: "quantum_kernel_model",
    ModelMode.ENSEMBLE: "quantum_ensemble",
    ModelMode.DISTRIBUTED_QNN: "quantum_parallel",
}

_TEMPLATE_MODES: dict[str, frozenset[ModelMode]] = {
    "quantum_classifier": frozenset({ModelMode.QUANTUM_ONLY}),
    "quantum_regressor": frozenset({ModelMode.QUANTUM_ONLY}),
    "estimator_qnn": frozenset({ModelMode.QUANTUM_ONLY}),
    "sampler_qnn": frozenset({ModelMode.QUANTUM_ONLY}),
    "classical_quantum_classical": frozenset({ModelMode.HYBRID, ModelMode.QUANVOLUTION}),
    "hybrid_quantum_model": frozenset({ModelMode.HYBRID, ModelMode.QUANVOLUTION}),
    "quantum_sequential": frozenset({ModelMode.HYBRID, ModelMode.QUANVOLUTION}),
    "quantum_parallel": frozenset(
        {ModelMode.HYBRID, ModelMode.QUANVOLUTION, ModelMode.DISTRIBUTED_QNN}
    ),
    "quantum_residual": frozenset({ModelMode.HYBRID, ModelMode.QUANVOLUTION}),
    "ids_quantum_bottleneck": frozenset({ModelMode.HYBRID}),
    "quantum_ensemble": frozenset({ModelMode.ENSEMBLE}),
    "quantum_kernel_model": frozenset({ModelMode.QUANTUM_KERNEL}),
}

#: Catalog names that denote an architecture the factory already builds.
#: These used to be advertised as ``composite`` (i.e. runnable) while
#: ``ModelFactory`` had no route for them, so the catalog counted capacity the
#: framework did not have.  Routing them here makes the catalog claim true;
#: ``tests/test_catalog.py`` asserts that no composite entry is left unrouted.
_TEMPLATE_ALIASES: dict[str, str] = {
    "variational_quantum_classifier": "quantum_classifier",
    "quantum_only_classifier": "quantum_classifier",
    "variational_quantum_regressor": "quantum_regressor",
    "quantum_only_regressor": "quantum_regressor",
    "classical_quantum_classical_network": "classical_quantum_classical",
    "parallel_quantum_layer_network": "quantum_parallel",
    "residual_hybrid_qnn": "quantum_residual",
    "quantum_bottleneck_network": "ids_quantum_bottleneck",
    "fidelity_quantum_kernel": "quantum_kernel_model",
    "statevector_quantum_kernel": "quantum_kernel_model",
    "qsvc": "quantum_kernel_model",
}

#: Options each template consumes.  Anything else is a typo or a misplaced
#: setting and is rejected rather than silently dropped.
_COMMON_MODEL_OPTIONS: frozenset[str] = frozenset()
#: ``None`` means "forwarded to an external estimator that validates for us".
_TEMPLATE_OPTIONS: dict[str, frozenset[str] | None] = {
    "quantum_classifier": frozenset({"qnn", "observable", "interpret", "sampler_output_shape"}),
    "quantum_regressor": frozenset({"qnn", "observable", "interpret", "sampler_output_shape"}),
    "estimator_qnn": frozenset({"observable"}),
    "sampler_qnn": frozenset({"interpret", "sampler_output_shape"}),
    "classical_quantum_classical": frozenset(
        {"qnn", "observable", "interpret", "sampler_output_shape"}
    ),
    "hybrid_quantum_model": frozenset({"qnn", "observable", "interpret", "sampler_output_shape"}),
    "quantum_sequential": frozenset(
        {"qnn", "observable", "interpret", "sampler_output_shape", "classical_blocks"}
    ),
    "quantum_parallel": frozenset(
        {"qnn", "observable", "interpret", "sampler_output_shape", "branches", "reduction"}
    ),
    "quantum_residual": frozenset({"qnn", "observable", "interpret", "sampler_output_shape"}),
    "ids_quantum_bottleneck": frozenset({"qnn", "observable", "interpret", "sampler_output_shape"}),
    "quantum_ensemble": frozenset(
        {"qnn", "observable", "interpret", "sampler_output_shape", "branches"}
    ),
    # Kernel estimators forward their remaining options to QSVC/sklearn, which
    # validates them against its own signature.
    "quantum_kernel_model": None,
}


class ModelFactory:
    def build(
        self,
        ir: CircuitIR,
        model: ModelSpec,
        backend: BackendSpec,
        *,
        seed: int | None = None,
    ) -> Any:
        """Build the declared model template.

        ``seed`` is applied *before* any layer is constructed, which is what
        makes two builds of the same :class:`~qforge_ai.specs.ExperimentSpec`
        start from identical weights.  It defaults to ``backend.seed`` so a
        direct ``ModelFactory`` caller is reproducible too.
        """

        template = self._resolve_template(model)
        self._validate_options(template, model)
        # Encoding, template and engine have to be checked *together*: each is
        # individually supported, and several combinations of them are not.
        validate_composition(
            ir, engine=backend.engine, mode=model.mode, template=template
        )
        seed_everything(backend.seed if seed is None else seed)
        if template == "quantum_kernel_model":
            # ``fidelity`` must be resolved and removed here: forwarding the
            # option dict alongside the keyword raised a bare ``TypeError: got
            # multiple values for keyword argument 'fidelity'`` before the
            # builder was even entered, so no amount of validation inside it
            # could produce a useful error.  Both engines now receive it -- the
            # PennyLane route used to drop it silently, which is why a
            # ``statevector_quantum_kernel`` request there behaved like
            # ``auto``.
            requested = self._kernel_fidelity(model)
            options = dict(model.options)
            options.pop("fidelity", None)
            if backend.engine == Engine.QISKIT:
                return build_qsvc(ir, backend=backend, fidelity=requested, **options)
            if backend.engine == Engine.PENNYLANE:
                return build_pennylane_qsvc(
                    ir, backend=backend, fidelity=requested, **options
                )
            raise CapabilityError(
                "The quantum-kernel template requires a Qiskit or PennyLane backend; "
                f"got {backend.engine}."
            )
        if template in {"quantum_classifier", "quantum_regressor", "estimator_qnn", "sampler_qnn"}:
            self._validate_quantum_only_output(ir, model, backend, template=template)
            return self._quantum_layer(ir, model, backend, template=template)
        if template == "quantum_ensemble":
            branches = self._branch_count(model)
            models = [
                self._hybrid_torch(
                    ir,
                    self._quantum_layer(ir, model, backend, template=template),
                    model,
                    architecture="standard",
                )
                for _ in range(branches)
            ]
            return QuantumParallel.build(models, reduction="mean")
        if template == "quantum_parallel":
            layers = [
                self._quantum_layer(ir, model, backend, template=template)
                for _ in range(self._branch_count(model))
            ]
            return self._parallel_hybrid(ir, layers, model)
        if template == "quantum_sequential":
            blocks = int(model.options.get("classical_blocks", 1))
            if blocks < 1:
                raise ConfigurationError("quantum_sequential classical_blocks must be at least 1.")
            hybrid = self._hybrid_torch(
                ir,
                self._quantum_layer(ir, model, backend, template=template),
                model,
                architecture="standard",
            )
            if blocks == 1:
                return hybrid
            nn = require(
                "torch.nn", package="torch", extra="torch", feature="QuantumSequential"
            )
            tail: list[Any] = []
            for _ in range(blocks - 1):
                tail.extend(
                    [
                        nn.Linear(model.output_dimension, model.output_dimension),
                        nn.GELU(),
                    ]
                )
            return nn.Sequential(hybrid, *tail)
        architecture = {
            "quantum_residual": "residual",
            "ids_quantum_bottleneck": "ids_bottleneck",
        }.get(template, "standard")
        quantum_layer = self._quantum_layer(ir, model, backend, template=template)
        return self._hybrid_torch(ir, quantum_layer, model, architecture=architecture)

    @staticmethod
    def resolve_template(name: str) -> str | None:
        """Return the canonical route for a template/catalog name, if any."""

        key = normalize_key(name)
        key = _TEMPLATE_ALIASES.get(key, key)
        return key if key in _TEMPLATE_MODES else None

    @staticmethod
    def _resolve_template(model: ModelSpec) -> str:
        template = normalize_key(model.template)
        if template == "auto":
            try:
                return _AUTO_TEMPLATE[model.mode]
            except KeyError as exc:
                raise CapabilityError(
                    f"There is no automatic model template for {model.mode}; choose an explicit template."
                ) from exc
        template = _TEMPLATE_ALIASES.get(template, template)
        try:
            allowed_modes = _TEMPLATE_MODES[template]
        except KeyError as exc:
            raise CapabilityError(
                f"Model template not implemented by the core: {template}. "
                f"Executable templates: "
                f"{sorted(set(_TEMPLATE_MODES) | set(_TEMPLATE_ALIASES))}"
            ) from exc
        if model.mode not in allowed_modes:
            raise ConfigurationError(
                f"template={template} is not compatible with mode={model.mode}; "
                f"allowed modes: {sorted(map(str, allowed_modes))}"
            )
        return template

    @staticmethod
    def _validate_options(template: str, model: ModelSpec) -> None:
        """Reject options the selected template will never read.

        A misspelled or misplaced option used to be accepted and dropped, so a
        model quietly ran with defaults the user believed they had overridden.
        """

        allowed = _TEMPLATE_OPTIONS.get(template)
        if allowed is None:
            return
        unknown = sorted(set(model.options) - (allowed | _COMMON_MODEL_OPTIONS))
        if unknown:
            raise ConfigurationError(
                f"template={template} does not use these model options keys: {unknown}. "
                f"Valid ones for this template: {sorted(allowed)}"
            )

    @staticmethod
    def _kernel_fidelity(model: ModelSpec) -> str:
        """Pick the kernel implementation implied by the template and options.

        ``fidelity`` is a legitimate kernel setting, so ``ModelSpec.options``
        may carry it; the template name implies one too.  Both are read here so
        the two can be reconciled -- and disagreement reported -- instead of
        colliding as duplicate keyword arguments.
        """

        from .kernel_contract import normalize_fidelity

        requested = normalize_key(model.template)
        implied = "statevector" if requested == "statevector_quantum_kernel" else "auto"
        raw = model.options.get("fidelity")
        if raw is None:
            return implied
        override = normalize_fidelity(raw)
        if implied not in {"auto", override}:
            raise ConfigurationError(
                f"Kernel fidelity is requested by both the template ({implied!r}) and options ({override!r}) "
                "with different values; choose one."
            )
        return str(override)

    @staticmethod
    def _validate_quantum_only_output(
        ir: CircuitIR,
        model: ModelSpec,
        backend: BackendSpec,
        *,
        template: str,
    ) -> None:
        """Check that a quantum-only model really produces ``output_dimension``.

        A quantum-only model has no classical head to reshape anything, so its
        output width is fixed by the circuit: the number of observables for an
        expectation/variance measurement, or the sampler output shape for a
        probability measurement.  ``output_dimension`` was read only by the
        hybrid templates, so asking a quantum-only model for 7 outputs
        produced a 2-output layer and said nothing -- and the mismatch then
        surfaced as a shape error in whatever consumed the predictions.
        """

        if backend.engine == Engine.QISKIT:
            # Raise the measurement-contract error first when both apply: an
            # estimator asked to read a probability circuit is the more
            # fundamental mistake, and the width is a consequence of it.
            ModelFactory._resolve_qiskit_qnn_kind(ir, model, template=template)
        _, _, quantum_output_dim = ModelFactory._dimensions(ir, model)
        if model.output_dimension == quantum_output_dim:
            return
        measurement = ir.measurements[0] if ir.measurements else None
        kind = measurement.kind if measurement else "yok"
        if kind in {"expectation", "variance"}:
            remedy = (
                f"The measurement carries {quantum_output_dim} observables. Set output_dimension="
                f"{quantum_output_dim}, put {model.output_dimension} observables in the measurement, "
                "or use mode='hybrid' for a classical head."
            )
        elif kind == "probability":
            remedy = (
                f"The probability measurement produces {quantum_output_dim} outputs. Set output_dimension="
                f"{quantum_output_dim}, map the outputs with options['interpret'] and "
                "options['sampler_output_shape'], or use mode='hybrid' for a classical "
                "head."
            )
        else:
            remedy = (
                f"This measurement produces {quantum_output_dim} outputs; set output_dimension="
                f"{quantum_output_dim} or use mode='hybrid'."
            )
        raise ConfigurationError(
            f"quantum-only template={template} takes its output width from the circuit and "
            f"cannot reshape it: output_dimension={model.output_dimension} was "
            f"requested, the quantum layer produces {quantum_output_dim} outputs. {remedy}"
        )

    @staticmethod
    def _branch_count(model: ModelSpec) -> int:
        branches = int(model.options.get("branches", 2))
        if branches < 2:
            raise ConfigurationError("branches must be at least 2 in a parallel/ensemble template.")
        return branches

    @staticmethod
    def _quantum_layer(
        ir: CircuitIR,
        model: ModelSpec,
        backend: BackendSpec,
        *,
        template: str = "",
    ) -> Any:
        if backend.engine == Engine.QISKIT:
            qnn_kind = ModelFactory._resolve_qiskit_qnn_kind(ir, model, template=template)
            if qnn_kind == "sampler":
                qnn = build_sampler_qnn(
                    ir,
                    backend=backend,
                    output_shape=model.options.get("sampler_output_shape"),
                    interpret=model.options.get("interpret"),
                )
            elif qnn_kind == "estimator":
                qnn = build_estimator_qnn(
                    ir,
                    backend=backend,
                    observable=model.options.get("observable"),
                )
            else:
                raise ConfigurationError(f"Unknown Qiskit QNN kind: {qnn_kind}")
            return qiskit_torch_layer(qnn)
        if backend.engine == Engine.PENNYLANE:
            qiskit_only = sorted(
                key
                for key in {"qnn", "interpret", "sampler_output_shape", "observable"}
                if key in model.options
            )
            if qiskit_only:
                raise ConfigurationError(
                    f"Qiskit-specific options cannot be used in a PennyLane model: {qiskit_only}"
                )
            return pennylane_torch_layer(
                ir,
                device=backend.device if backend.device != "statevector" else "default.qubit",
                shots=backend.shots,
                diff_method=pennylane_diff_method(backend.gradient),
                noise=backend.noise,
                seed=backend.seed,
                device_options=dict(backend.options.get("device_options", {})),
                # gradient_options used to be validated by BackendSpec and then
                # never read, so a negative epsilon or an unknown key built a
                # model that quietly ran with PennyLane's defaults.
                gradient_options=dict(backend.options.get("gradient_options", {})),
            )
        raise CapabilityError("A Qiskit or PennyLane backend is required to build a model.")

    @staticmethod
    def _resolve_qiskit_qnn_kind(
        ir: CircuitIR,
        model: ModelSpec,
        *,
        template: str = "",
    ) -> str:
        measurement = ir.measurements[0].kind if ir.measurements else None
        # The estimator_qnn / sampler_qnn templates name the primitive
        # directly; anything else falls back to the qnn option or inference.
        pinned = {"estimator_qnn": "estimator", "sampler_qnn": "sampler"}.get(template)
        requested = pinned or normalize_key(str(model.options.get("qnn", "auto")))
        if requested == "auto":
            inferred = (
                {"expectation": "estimator", "probability": "sampler"}.get(measurement)
                if measurement is not None
                else None
            )
            if inferred is None:
                raise ConfigurationError(
                    "Automatic Qiskit QNN selection is possible only with an expectation (EstimatorQNN) "
                    "or probability (SamplerQNN) measurement; "
                    f"got {measurement!r}."
                )
            return inferred
        if requested not in {"estimator", "sampler"}:
            raise ConfigurationError(f"Unknown Qiskit QNN kind: {requested}")
        expected = "expectation" if requested == "estimator" else "probability"
        if measurement != expected:
            raise ConfigurationError(
                f"qnn='{requested}' requires measurement='{expected}'; "
                f"the circuit defines measurement='{measurement}'."
            )
        return requested

    @staticmethod
    def _dimensions(ir: CircuitIR, model: ModelSpec) -> tuple[int, int, int]:
        input_dim = model.input_dimension or ir.parameter_groups().get(
            "input", ir.num_data_qubits
        )
        quantum_input_dim = ir.parameter_groups().get("input", ir.num_data_qubits)
        measurement = ir.measurements[0] if ir.measurements else None
        if measurement and measurement.kind == "probability" and "sampler_output_shape" in model.options:
            quantum_output_dim = prod(
                validate_output_shape(
                    model.options["sampler_output_shape"], label="sampler_output_shape"
                )
            )
        elif measurement and measurement.kind in {"expectation", "variance"}:
            # An ``observable`` override replaces the IR's observable list, so
            # the classical head must follow the override -- not the IR.
            # Reading the IR here produced a head sized for two outputs in
            # front of a QNN emitting one, and the mismatch only surfaced as a
            # matmul error during the first forward pass.
            quantum_output_dim = observable_output_count(model.options.get("observable")) or len(
                measurement.observables
            )
        else:
            quantum_output_dim = 2 ** len(
                measurement.wires if measurement else ir.data_wires
            )
        return input_dim, quantum_input_dim, quantum_output_dim

    @staticmethod
    def _encoder(
        nn: Any,
        *,
        input_dim: int,
        quantum_input_dim: int,
        model: ModelSpec,
        layer_norm: bool = False,
    ) -> Any:
        hidden = model.hidden_dimensions or (max(input_dim, quantum_input_dim),)
        layers: list[Any] = []
        previous = input_dim
        for width in hidden:
            layers.append(nn.Linear(previous, width))
            if layer_norm:
                layers.append(nn.LayerNorm(width))
            layers.append(nn.GELU())
            if model.dropout:
                layers.append(nn.Dropout(model.dropout))
            previous = width
        layers.append(nn.Linear(previous, quantum_input_dim))
        return nn.Sequential(*layers)

    @staticmethod
    def _hybrid_torch(
        ir: CircuitIR,
        quantum_layer: Any,
        model: ModelSpec,
        *,
        architecture: str,
    ) -> Any:
        torch = require("torch", package="torch", extra="torch", feature="hybrid model")
        nn = require("torch.nn", package="torch", extra="torch", feature="hybrid model")
        input_dim, quantum_input_dim, quantum_output_dim = ModelFactory._dimensions(ir, model)
        encoder = ModelFactory._encoder(
            nn,
            input_dim=input_dim,
            quantum_input_dim=quantum_input_dim,
            model=model,
            layer_norm=architecture == "ids_bottleneck",
        )

        class HybridModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = encoder
                self.quantum = quantum_layer
                self.residual = (
                    nn.Linear(quantum_input_dim, quantum_output_dim, bias=False)
                    if architecture == "residual"
                    else None
                )
                self.normalizer = (
                    nn.LayerNorm(quantum_output_dim)
                    if architecture == "ids_bottleneck"
                    else nn.Identity()
                )
                self.head = nn.Linear(quantum_output_dim, model.output_dimension)

            def forward(self, inputs: Any) -> Any:
                single_input = inputs.ndim == 1
                batched_inputs = inputs.unsqueeze(0) if single_input else inputs
                encoded = torch.tanh(self.encoder(batched_inputs)) * torch.pi
                quantum_output = self.quantum(encoded)
                if quantum_output.ndim == 1:
                    quantum_output = quantum_output.unsqueeze(0)
                quantum_output = quantum_output.reshape(encoded.shape[0], -1)
                quantum_output = quantum_output.to(dtype=batched_inputs.dtype)
                if self.residual is not None:
                    quantum_output = quantum_output + self.residual(encoded)
                output = self.head(self.normalizer(quantum_output))
                return output.squeeze(0) if single_input else output

        return HybridModel()

    @staticmethod
    def _parallel_hybrid(ir: CircuitIR, quantum_layers: list[Any], model: ModelSpec) -> Any:
        torch = require("torch", package="torch", extra="torch", feature="parallel QNN")
        nn = require("torch.nn", package="torch", extra="torch", feature="parallel QNN")
        input_dim, quantum_input_dim, quantum_output_dim = ModelFactory._dimensions(ir, model)
        encoder = ModelFactory._encoder(
            nn,
            input_dim=input_dim,
            quantum_input_dim=quantum_input_dim,
            model=model,
        )
        reduction = str(model.options.get("reduction", "concat"))
        if reduction not in {"concat", "mean", "sum"}:
            raise ConfigurationError(f"Unknown parallel reduction: {reduction}")
        head_input = (
            quantum_output_dim * len(quantum_layers)
            if reduction == "concat"
            else quantum_output_dim
        )

        class ParallelHybrid(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = encoder
                self.quantum = nn.ModuleList(quantum_layers)
                self.head = nn.Linear(head_input, model.output_dimension)

            def forward(self, inputs: Any) -> Any:
                single_input = inputs.ndim == 1
                batched_inputs = inputs.unsqueeze(0) if single_input else inputs
                encoded = torch.tanh(self.encoder(batched_inputs)) * torch.pi
                outputs = [
                    layer(encoded).reshape(encoded.shape[0], -1).to(dtype=batched_inputs.dtype)
                    for layer in self.quantum
                ]
                if reduction == "concat":
                    merged = torch.cat(outputs, dim=-1)
                else:
                    merged = torch.stack(outputs, dim=0)
                    merged = merged.mean(dim=0) if reduction == "mean" else merged.sum(dim=0)
                output = self.head(merged)
                return output.squeeze(0) if single_input else output

        return ParallelHybrid()


__all__ = ["ModelFactory"]
