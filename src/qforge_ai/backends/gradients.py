"""Shared gradient contract for both engines.

``BackendSpec.options["gradient_options"]`` was accepted on the PennyLane side
and then dropped on the floor: a model could be built with ``epsilon=-1.0`` and
an unknown key and would run happily with PennyLane's defaults, reporting a
finite-difference step it never used.  The Qiskit side validated the same
options properly, so the two engines disagreed about what a spec meant.

This module holds one vocabulary for both.  The framework's canonical keys are
``epsilon`` (step size), ``method`` (``central``/``forward``/``backward``) and
``batch_size`` (SPSA sampling directions).  Each engine's native spelling is
accepted too and normalized onto the canonical name, so the same spec produces
the same numerical differentiation on either backend.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..errors import ConfigurationError
from ..specs import GradientMethod

#: Canonical framework key -> PennyLane gradient-transform keyword.
_PENNYLANE_KEY_MAP: dict[str, str] = {
    "epsilon": "h",
    "method": "strategy",
    "batch_size": "num_directions",
}

#: PennyLane spellings accepted verbatim, per diff method.
_PENNYLANE_NATIVE_KEYS: dict[str, frozenset[str]] = {
    "finite-diff": frozenset({"h", "approx_order", "strategy", "validate_params"}),
    "spsa": frozenset(
        {"h", "approx_order", "strategy", "validate_params", "num_directions", "sampler_rng"}
    ),
}

#: Canonical keys each diff method can honour.
_PENNYLANE_CANONICAL_KEYS: dict[str, frozenset[str]] = {
    "finite-diff": frozenset({"epsilon", "method"}),
    "spsa": frozenset({"epsilon", "method", "batch_size"}),
}

#: ``method`` values and their PennyLane ``strategy`` equivalents.
_STRATEGY_MAP = {
    "central": "center",
    "center": "center",
    "forward": "forward",
    "backward": "backward",
}

#: QuantumForge gradient method -> PennyLane ``diff_method``.
_PENNYLANE_DIFF_METHODS: dict[GradientMethod, str] = {
    GradientMethod.AUTO: "best",
    GradientMethod.PARAMETER_SHIFT: "parameter-shift",
    GradientMethod.FINITE_DIFF: "finite-diff",
    GradientMethod.SPSA: "spsa",
    GradientMethod.ADJOINT: "adjoint",
    GradientMethod.BACKPROP: "backprop",
}


def pennylane_diff_method(gradient: GradientMethod | str) -> str:
    """Return the PennyLane ``diff_method`` for a framework gradient method."""

    try:
        resolved = GradientMethod(gradient)
    except ValueError as exc:
        raise ConfigurationError(
            f"Unknown gradient method: {gradient!r}. "
            f"Supported: {sorted(str(item) for item in _PENNYLANE_DIFF_METHODS)}"
        ) from exc
    try:
        return _PENNYLANE_DIFF_METHODS[resolved]
    except KeyError as exc:
        raise ConfigurationError(
            f"PennyLane does not implement this gradient method: {resolved}. "
            f"Supported: {sorted(str(item) for item in _PENNYLANE_DIFF_METHODS)}"
        ) from exc


def normalize_pennylane_gradient_options(
    options: dict[str, Any] | None,
    *,
    diff_method: str,
) -> dict[str, Any]:
    """Validate *options* and return PennyLane gradient keyword arguments.

    Raises rather than dropping anything: an option that reaches a method
    which cannot use it is a mistake in the experiment definition, and the
    whole point of this function is that it stops being invisible.
    """

    supplied = dict(options or {})
    if not supplied:
        return {}
    canonical = _PENNYLANE_CANONICAL_KEYS.get(diff_method)
    native = _PENNYLANE_NATIVE_KEYS.get(diff_method)
    if canonical is None or native is None:
        raise ConfigurationError(
            f"diff_method={diff_method!r} uses no gradient_options value; "
            f"the given keys are rejected rather than dropped silently: "
            f"{sorted(supplied)}. For epsilon/method/batch_size choose "
            "gradient='finite_diff' or 'spsa'."
        )
    unknown = sorted(set(supplied) - canonical - native)
    if unknown:
        raise ConfigurationError(
            f"Unknown PennyLane gradient_options keys: {unknown}. "
            f"Valid ones for this method: {sorted(canonical | native)}"
        )
    resolved: dict[str, Any] = {}
    for key, value in supplied.items():
        target = _PENNYLANE_KEY_MAP.get(key, key)
        if target in resolved and resolved[target] != value:
            raise ConfigurationError(
                f"gradient_options {key!r} and its equivalent cannot be given at the same time "
                f"with different values (PennyLane key: {target!r})."
            )
        resolved[target] = value
    if "h" in resolved:
        step = resolved["h"]
        if isinstance(step, bool) or not isinstance(step, (int, float)):
            raise ConfigurationError("gradient_options epsilon must be numeric.")
        step = float(step)
        if not np.isfinite(step) or step <= 0.0:
            raise ConfigurationError(
                f"gradient_options epsilon must be positive and finite; got {step!r}."
            )
        resolved["h"] = step
    if "strategy" in resolved:
        strategy = str(resolved["strategy"]).strip().lower()
        if strategy not in _STRATEGY_MAP:
            raise ConfigurationError(
                f"gradient_options method must be central, forward or backward; "
                f"got {resolved['strategy']!r}."
            )
        resolved["strategy"] = _STRATEGY_MAP[strategy]
    if "num_directions" in resolved:
        directions = resolved["num_directions"]
        if isinstance(directions, bool) or not isinstance(directions, int) or directions < 1:
            raise ConfigurationError("gradient_options batch_size must be an integer of at least 1.")
    if "approx_order" in resolved:
        order = resolved["approx_order"]
        if isinstance(order, bool) or not isinstance(order, int) or order < 1:
            raise ConfigurationError("gradient_options approx_order must be an integer of at least 1.")
    return resolved


__all__ = [
    "pennylane_diff_method",
    "normalize_pennylane_gradient_options",
]
