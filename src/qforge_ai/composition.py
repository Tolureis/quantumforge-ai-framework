"""Preflight for encoding x model x engine compositions.

The catalog verified every component *against a baseline chosen to suit it*:
basis encoding was checked with a plain circuit execution, amplitude encoding
with concrete inputs, and each model template with angle encoding.  Every
individual claim was true, and the combination of two true claims still failed:

* ``BASIS``/``STATE_PREP`` cannot be compiled with unbound symbolic parameters,
  so neither can appear in a Qiskit ``EstimatorQNN``/``SamplerQNN`` -- the
  translator raised deep inside the QNN builder with a message about "concrete
  inputs" that never mentioned the encoding the user had chosen.
* the hybrid templates put a classical encoder in front of the circuit and feed
  it ``tanh(...) * pi``, which basis encoding rejects (it takes bits) and
  un-normalized amplitude encoding rejects (it takes unit vectors) -- both only
  at the first forward pass, long after the model was built.
* an encoding with no input parameters at all (superposition) leaves the
  classical encoder's output connected to nothing: the model builds, trains,
  and is constant in its input.

Deriving these rules from the IR -- rather than from the encoding's *name* --
means they also cover hand-written circuits and plugin encodings.  The same
predicates drive the catalog's per-entry capability tags, so what the catalog
advertises and what the framework accepts cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import ConfigurationError
from .ir import CircuitIR, Operation, ParameterRef
from .specs import Engine, ModelMode

#: IR operations that cannot carry an unbound symbolic parameter.  Qiskit
#: compiles ``BASIS`` to a concrete ``X`` (or nothing) and ``STATE_PREP`` to a
#: concrete ``initialize``; neither has a parameterized form, so a Qiskit QNN
#: -- which binds inputs *after* compilation -- cannot contain them.
SYMBOLIC_INCOMPATIBLE_OPERATIONS = frozenset({"BASIS", "STATE_PREP"})

#: IR operations whose input must be a computational-basis bit.
DISCRETE_INPUT_OPERATIONS = frozenset({"BASIS"})

#: IR operations whose input must be an L2-normalized vector unless the
#: operation was built with ``normalize=True``.
NORMALIZED_INPUT_OPERATIONS = frozenset({"STATE_PREP"})

#: Model templates that place a trainable classical encoder in front of the
#: quantum circuit.  Their encoder emits ``tanh(...) * pi``: continuous, signed
#: and unnormalized.
CLASSICAL_ENCODER_MODES = frozenset(
    {
        ModelMode.HYBRID,
        ModelMode.QUANVOLUTION,
        ModelMode.ENSEMBLE,
        ModelMode.DISTRIBUTED_QNN,
    }
)


@dataclass(frozen=True, slots=True)
class EncodingProfile:
    """What an IR's encoding layer can actually compose with."""

    #: Whether any operation carries an ``input`` parameter reference.
    encodes_data: bool
    #: Whether every data-loading operation compiles with unbound parameters.
    symbolic_parameters: bool
    #: Whether the encoding accepts arbitrary real-valued features.
    continuous_input: bool
    #: Whether the encoding requires a unit-norm input it does not normalize.
    requires_normalized_input: bool
    #: Whether input gradients can flow through the encoding.
    differentiable: bool
    #: Whether the data-loading layer itself carries trainable weights.
    trainable_encoding: bool
    #: How many distinct input features the circuit consumes.
    input_features: int
    #: Operation names responsible for a restriction, for error messages.
    restricted_operations: tuple[str, ...]


def _input_operations(ir: CircuitIR) -> list[Operation]:
    return [
        operation
        for operation in ir.operations
        if any(
            isinstance(param, ParameterRef) and param.group == "input"
            for param in operation.params
        )
    ]


def encoding_profile(ir: CircuitIR) -> EncodingProfile:
    """Describe how *ir*'s data-loading layer constrains composition."""

    loaders = _input_operations(ir)
    names = {operation.name for operation in loaders}
    restricted = sorted(
        names
        & (
            SYMBOLIC_INCOMPATIBLE_OPERATIONS
            | DISCRETE_INPUT_OPERATIONS
            | NORMALIZED_INPUT_OPERATIONS
        )
    )
    needs_normalized = any(
        operation.name in NORMALIZED_INPUT_OPERATIONS
        and not bool(operation.metadata.get("normalize", False))
        for operation in loaders
    )
    discrete = bool(names & DISCRETE_INPUT_OPERATIONS)
    trainable_encoding = any(
        isinstance(param, ParameterRef) and param.group == "weight"
        for operation in ir.operations
        if operation.metadata.get("role") == "encoding"
        for param in operation.params
    )
    return EncodingProfile(
        encodes_data=bool(loaders),
        symbolic_parameters=not (names & SYMBOLIC_INCOMPATIBLE_OPERATIONS),
        continuous_input=not discrete,
        requires_normalized_input=needs_normalized,
        # A basis state is a step function of its bit: there is no gradient to
        # take, on either engine.  Everything else here is a rotation angle or
        # an amplitude, and both are differentiable.
        differentiable=bool(loaders) and not discrete,
        trainable_encoding=trainable_encoding,
        input_features=ir.parameter_groups().get("input", 0),
        restricted_operations=tuple(restricted),
    )


def validate_composition(
    ir: CircuitIR,
    *,
    engine: Engine,
    mode: ModelMode,
    template: str,
) -> None:
    """Reject encoding/model/engine combinations that cannot work.

    Raised at build time, naming the encoding *and* the remedy, instead of
    surfacing as a translator error inside a QNN primitive, as a value error on
    the first forward pass, or -- worse -- as a model that trains happily while
    ignoring its input.
    """

    profile = encoding_profile(ir)
    offenders = ", ".join(profile.restricted_operations) or "-"

    # A model whose circuit carries no input parameter cannot depend on its
    # data at all.  Qiskit reported this as "Invalid input dimension ...
    # expected input compatible to 0" from inside the QNN, and PennyLane simply
    # returned the same value for every sample.
    if not profile.encodes_data:
        raise ConfigurationError(
            f"template={template} builds a data-dependent model, but the circuit has no "
            "input parameter at all: the model would produce the same value for every input "
            "and the encoder's output would not be connected to any gate. Choose an "
            "encoding that loads the data (angle, phase, zz_feature_map, amplitude, etc.); "
            "superposition_encoding only prepares an initial state."
        )

    if mode == ModelMode.QUANTUM_KERNEL:
        _kernel_rules(profile, engine=engine, offenders=offenders)
        return

    if engine == Engine.QISKIT and not profile.symbolic_parameters:
        raise ConfigurationError(
            f"Qiskit QNN binds its inputs after compilation, so the circuit must be "
            f"compilable with symbolic parameters; the {offenders} operation "
            "requires a concrete value and has no parametric form. This combination is "
            "rejected here rather than blowing up inside the QNN setup with a "
            "'requires a concrete input' error. Options: use an angle/phase based encoding "
            "(angle, phase, zz_feature_map, iqp), switch to the PennyLane engine "
            f"(backend.engine='pennylane'), or prefer the quantum-kernel template for {offenders} "
            "instead."
        )

    if mode not in CLASSICAL_ENCODER_MODES:
        return

    if not profile.continuous_input:
        raise ConfigurationError(
            f"template={template} feeds the classical encoder's continuous output "
            f"(tanh(...)*pi) into the circuit; {offenders} accepts only 0/1 bits, "
            "so this combination would fail on the first forward pass. "
            "Choose an encoding that takes continuous values (angle, phase, amplitude), or "
            "use mode='quantum_only', which feeds the bits into the circuit directly."
        )

    if profile.requires_normalized_input:
        raise ConfigurationError(
            f"template={template} feeds the classical encoder's unnormalized output "
            f"into the circuit; with normalize=False, {offenders} requires an input whose "
            "L2 norm is 1. Use EncodingSpec(normalize=True) (which applies the same "
            "division on both engines), or choose an encoding that normalizes after the "
            "encoder."
        )


def validate_kernel_composition(ir: CircuitIR, *, engine: Engine) -> None:
    """Reject feature maps a fixed quantum kernel cannot be built from.

    Called by both kernel builders, so a direct
    ``build_quantum_kernel``/``build_fidelity_kernel`` call is checked as
    thoroughly as one routed through :class:`ModelFactory`.
    """

    profile = encoding_profile(ir)
    if not profile.encodes_data:
        raise ConfigurationError(
            "A quantum kernel feature map must depend on the data; the circuit has no input "
            "parameter at all, so the Gram matrix would be constant for every pair of "
            "samples. Choose an encoding that loads the data."
        )
    _kernel_rules(
        profile,
        engine=engine,
        offenders=", ".join(profile.restricted_operations) or "-",
    )


def _kernel_rules(
    profile: EncodingProfile,
    *,
    engine: Engine,
    offenders: str,
) -> None:
    """Shared kernel feature-map rules."""

    if profile.trainable_encoding:
        raise ConfigurationError(
            "A quantum kernel requires a fixed feature map; this encoding carries a trainable "
            "weight parameter, so the Gram matrix would change during training. Use a plugin "
            "for a trainable kernel, or choose an encoding that carries no weight "
            "(angle, phase, zz_feature_map, iqp)."
        )
    if not profile.continuous_input:
        # A kernel evaluates the feature map at arbitrary sample values, so a
        # bit-only encoding fails on the first evaluation -- as a translator
        # error on Qiskit and a value error on PennyLane.
        raise ConfigurationError(
            f"A quantum kernel evaluates its feature map at arbitrary sample values; "
            f"{offenders} accepts only 0/1 bits. Choose an encoding that takes continuous "
            "values (angle, phase, zz_feature_map, iqp, amplitude)."
        )
    if engine == Engine.QISKIT and not profile.symbolic_parameters:
        raise ConfigurationError(
            f"The Qiskit kernel path compiles the feature map with symbolic parameters; "
            f"{offenders} requires a concrete value and has no parametric form. "
            "For an analytic amplitude kernel use engine='pennylane', or choose an "
            "angle/phase based encoding."
        )


__all__ = [
    "SYMBOLIC_INCOMPATIBLE_OPERATIONS",
    "DISCRETE_INPUT_OPERATIONS",
    "NORMALIZED_INPUT_OPERATIONS",
    "CLASSICAL_ENCODER_MODES",
    "EncodingProfile",
    "encoding_profile",
    "validate_composition",
    "validate_kernel_composition",
]
