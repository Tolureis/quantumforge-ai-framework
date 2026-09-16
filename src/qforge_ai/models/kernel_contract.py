"""One quantum-kernel contract for both engines.

Each engine used to decide on its own what a kernel ``BackendSpec`` meant, and
the two decisions did not agree.  The same experiment definition produced:

===============================================  =====================  ==========================
definition                                       Qiskit                 PennyLane
===============================================  =====================  ==========================
``fidelity='sampler'``, ``shots=None``           explicit error         silent analytic kernel
``statevector_quantum_kernel`` + noise           explicit error         silent noisy "analytic"
``fidelity=...`` on the PennyLane route          honoured               never forwarded at all
``NoiseSpec(name='ideal')``                      treated as noise       treated as ideal
``readout_probability`` on an analytic kernel    explicit error         silently dropped
===============================================  =====================  ==========================

This module resolves the contract once, from the spec alone, before either
engine is touched.  ``resolve_kernel_contract`` answers three questions:

* which kernel does this definition actually ask for,
* which requested execution settings that kernel cannot honour, and
* what shot count it needs.

The engines then only implement what they were handed, and an engine that
cannot implement it says so by name.

Kernel kinds
------------
``statevector``
    Analytic, noiseless, ``|<phi(x)|phi(y)>|^2``: a genuine fidelity kernel
    between two pure states.  Rejects every execution setting it cannot honour
    rather than dropping it.
``hilbert_schmidt`` (spelled ``density_matrix`` after the simulation method)
    Analytic and noise-aware: ``Tr[rho(x) rho(y)]``, the Hilbert-Schmidt inner
    product of the two encoded states.  It is a valid Mercer kernel --
    symmetric and positive semi-definite by construction -- and it needs no
    shots, but it is **not** the mixed-state (Uhlmann) fidelity: the true
    fidelity of a state with itself is 1, while ``Tr[rho^2]`` is the purity and
    falls below 1 under any state-changing channel.  Both engines implement it.
``sampler``
    Finite-shot compute-uncompute.  Requires an explicit shot count on *both*
    engines: a kernel that silently invents 1024 shots is a different
    experiment from the one that was written down.  It is also the only kernel
    that measures, so it is the only one a readout channel can apply to.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import ConfigurationError
from ..noise import canonical_noise_name, is_ideal, is_terminal_only
from ..specs import BackendSpec, Engine, GradientMethod

#: Alternative spellings accepted for a kernel kind.  ``hilbert_schmidt`` is the
#: honest name for ``Tr[rho(x) rho(y)]``; ``density_matrix`` names the
#: simulation it needs and is kept because 1.2.x configurations use it.
KERNEL_FIDELITY_ALIASES: dict[str, str] = {
    "auto": "auto",
    "statevector": "statevector",
    "density_matrix": "density_matrix",
    "hilbert_schmidt": "density_matrix",
    "hilbert_schmidt_kernel": "density_matrix",
    "sampler": "sampler",
}

#: Every value ``ModelSpec.options['fidelity']`` and the kernel factories accept.
KERNEL_FIDELITIES = frozenset(KERNEL_FIDELITY_ALIASES)

#: Kernels that need no shots.
ANALYTIC_FIDELITIES = frozenset({"statevector", "density_matrix"})

#: Execution-setting keys that force a kernel off the analytic statevector path.
_OPTION_KEYS = ("backend_options", "run_options", "primitive_options", "device_options")


def normalize_fidelity(value: str) -> str:
    """Map a kernel-kind spelling onto its canonical name, or raise."""

    key = "_".join(str(value).strip().lower().replace("-", "_").split())
    try:
        return KERNEL_FIDELITY_ALIASES[key]
    except KeyError:
        raise ConfigurationError(
            f"Unknown kernel fidelity: {value!r}; "
            f"valid ones: {sorted(KERNEL_FIDELITIES)}."
        ) from None


@dataclass(frozen=True, slots=True)
class KernelContract:
    """What a kernel ``BackendSpec`` actually asks for, resolved once."""

    #: Resolved kernel kind: never ``"auto"``.
    fidelity: str
    #: Shot count the kernel will use; ``None`` for the analytic kinds.
    shots: int | None
    #: Whether the spec carries a noise channel of any kind.
    noisy: bool
    #: Requested settings a plain analytic statevector kernel cannot honour.
    unhonoured: tuple[str, ...]

    @property
    def analytic(self) -> bool:
        return self.fidelity in ANALYTIC_FIDELITIES

    @property
    def quantity(self) -> str:
        """The mathematical object this kernel computes, named honestly."""

        return {
            "statevector": "|<phi(x)|phi(y)>|^2",
            "density_matrix": "Tr[rho(x) rho(y)]",
            "sampler": "P(0...0) of U(y)† E U(x)",
        }[self.fidelity]

    @property
    def sampled(self) -> bool:
        return not self.analytic

    @property
    def enforce_psd(self) -> bool:
        """Whether the Gram matrix needs symmetrization and PSD projection.

        The analytic kinds are Gram matrices of explicit state vectors, so they
        are symmetric and positive semi-definite by construction.  A sampled
        compute-uncompute estimate is neither: ``K(x,y)`` and ``K(y,x)`` are
        two independent finite-shot experiments.
        """

        return self.sampled


def resolve_kernel_contract(
    backend: BackendSpec,
    *,
    fidelity: str = "auto",
) -> KernelContract:
    """Resolve *fidelity* and *backend* into the kernel they actually describe.

    Raises rather than silently downgrading: every branch here used to differ
    between the two engines, and each difference turned one written-down
    experiment into two different computations.
    """

    fidelity = normalize_fidelity(fidelity)
    if backend.gradient != GradientMethod.AUTO:
        # A kernel method has no differentiable parameters, so a gradient
        # setting here can only be a misplaced option.
        raise ConfigurationError(
            f"The quantum-kernel path uses no gradient; backend.gradient={backend.gradient} "
            "is rejected rather than ignored silently. For a kernel, leave "
            "gradient=GradientMethod.AUTO."
        )
    if backend.options.get("gradient_options"):
        raise ConfigurationError(
            "The quantum-kernel path uses no gradient_options; these keys are rejected "
            "rather than dropped silently."
        )

    # ``is_ideal`` rather than a hand-written name set: 'ideal', 'no_noise' and
    # 'shot_noise' are documented aliases for "the state is untouched", and
    # spelling the test by hand made ``NoiseSpec(name='ideal')`` behave like a
    # real channel on the Qiskit kernel path.
    noisy = not is_ideal(backend.noise)
    terminal = is_terminal_only(backend.noise)
    unhonoured: list[str] = []
    if backend.shots is not None:
        unhonoured.append(f"shots={backend.shots}")
    if noisy:
        unhonoured.append(f"noise={canonical_noise_name(backend.noise.name)}")
    if backend.method != "automatic":
        unhonoured.append(f"method={backend.method}")
    if backend.compute_device != "CPU":
        unhonoured.append(f"compute_device={backend.compute_device}")
    if backend.engine == Engine.PENNYLANE and backend.device == "default.mixed":
        # An explicitly mixed-state device is a request for noisy evolution
        # even when no channel is configured.
        unhonoured.append("device=default.mixed")
    for key in _OPTION_KEYS:
        if backend.options.get(key):
            unhonoured.append(key)
    frozen = tuple(dict.fromkeys(unhonoured))

    resolved = _resolve_fidelity(
        fidelity, noisy=noisy, terminal=terminal, backend=backend, unhonoured=frozen
    )
    if resolved == "statevector" and frozen:
        raise ConfigurationError(
            "The statevector kernel is an analytic, noiseless, single-CPU statevector "
            f"simulation; it cannot apply these settings: {sorted(frozen)}. "
            "For a noisy analytic kernel choose fidelity='hilbert_schmidt', for sampling "
            "choose fidelity='sampler' (with explicit shots)."
        )
    if terminal and resolved != "sampler":
        # A readout channel is defined at the measurement boundary, and the
        # analytic kernels never measure: they read the encoded state.  The
        # PennyLane route used to accept the combination and drop the channel
        # -- ``readout_probability=1.0`` produced a bit-identical noiseless
        # Gram matrix -- while Qiskit reported an unrelated capability error.
        raise ConfigurationError(
            f"Readout error is defined at the measurement boundary; the analytic {resolved} kernel "
            "reads the state directly and performs no measurement at all, so this channel is "
            "rejected rather than dropped silently. The only kernel that carries readout noise "
            "is the sampled compute-uncompute one: choose fidelity='sampler' and give an "
            "explicit BackendSpec.shots (for a gate channel use depolarizing, bit_flip, "
            "etc.)."
        )
    if resolved == "density_matrix" and backend.shots is not None:
        raise ConfigurationError(
            f"The hilbert_schmidt (density_matrix) kernel is analytic and uses no shots; "
            f"shots={backend.shots} was given. For sampling choose fidelity='sampler'."
        )
    if resolved == "sampler" and backend.shots is None:
        raise ConfigurationError(
            "A sampler-based kernel relies on sampling and does not fall back to 1024 shots "
            f"silently; give BackendSpec.shots for {sorted(frozen) or 'this path'}."
        )
    return KernelContract(
        fidelity=resolved,
        shots=backend.shots if resolved == "sampler" else None,
        noisy=noisy,
        unhonoured=frozen,
    )


def _resolve_fidelity(
    fidelity: str,
    *,
    noisy: bool,
    terminal: bool,
    backend: BackendSpec,
    unhonoured: tuple[str, ...],
) -> str:
    if fidelity != "auto":
        return fidelity
    # ``auto`` resolves identically on both engines, and both engines can now
    # run every kind it resolves to -- so ``auto`` no longer means "analytic
    # mixed state on PennyLane, CapabilityError on Qiskit".
    if backend.shots is not None:
        return "sampler"
    # Readout is a measurement-boundary channel: only the sampled kernel
    # measures, so ``auto`` resolves there and the missing shot count is
    # reported as the missing shot count.
    if terminal:
        return "sampler"
    if noisy or (backend.engine == Engine.PENNYLANE and backend.device == "default.mixed"):
        return "density_matrix"
    if unhonoured:
        return "sampler"
    return "statevector"


__all__ = [
    "KERNEL_FIDELITIES",
    "KERNEL_FIDELITY_ALIASES",
    "ANALYTIC_FIDELITIES",
    "KernelContract",
    "normalize_fidelity",
    "resolve_kernel_contract",
]
