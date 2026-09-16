"""Runtime capability probes and reproducibility helpers."""

from __future__ import annotations

import os
import random
from collections.abc import Iterator
from contextlib import contextmanager
from functools import cache, lru_cache
from importlib import import_module
from importlib.metadata import entry_points
from typing import Any

import numpy as np


@lru_cache(maxsize=1)
def qiskit_aer_devices() -> frozenset[str]:
    """Return devices exposed by the installed Aer build on this machine."""

    try:
        aer = import_module("qiskit_aer")
        devices = aer.AerSimulator().available_devices()
    except Exception:
        # Capability probing is deliberately conservative: a broken/missing
        # optional runtime is equivalent to the device being unavailable.
        return frozenset()
    return frozenset(str(device).upper() for device in devices)


def qiskit_gpu_available() -> bool:
    """Report actual Aer GPU availability, not package-level potential."""

    return "GPU" in qiskit_aer_devices()


@lru_cache(maxsize=1)
def qiskit_aer_methods() -> frozenset[str]:
    """Return simulation methods the installed Aer build actually accepts."""

    try:
        aer = import_module("qiskit_aer")
        methods = aer.AerSimulator().available_methods()
    except Exception:
        return frozenset()
    return frozenset(str(method) for method in methods)


@lru_cache(maxsize=1)
def pennylane_registered_devices() -> frozenset[str]:
    """Return PennyLane device names actually registered on this machine.

    Reading the ``pennylane.plugins`` entry points is what separates a device
    the framework *knows about* from one that is *installed*.  Claiming
    ``lightning.gpu`` on a plain ``pennylane-lightning`` install produced a
    capability report that disagreed with ``qml.device`` at run time.
    """

    try:
        import_module("pennylane")
    except Exception:
        return frozenset()
    names: set[str] = set()
    for group in ("pennylane.plugins", "pennylane.devices"):
        try:
            names.update(entry.name for entry in entry_points(group=group))
        except Exception:  # pragma: no cover - importlib backend differences
            continue
    return frozenset(names)


@cache
def pennylane_device_constructible(name: str) -> bool:
    """Return whether ``qml.device(name, wires=1)`` actually succeeds.

    Entry-point presence is necessary but not sufficient: a GPU device can be
    registered while its CUDA backend is missing, in which case construction
    is the only honest test.
    """

    if name not in pennylane_registered_devices():
        return False
    try:
        qml = import_module("pennylane")
        qml.device(name, wires=1)
    except Exception:
        return False
    return True


def pennylane_gpu_available() -> bool:
    """Report whether a PennyLane GPU device can genuinely be created here."""

    return any(
        pennylane_device_constructible(name)
        for name in ("lightning.gpu", "lightning.kokkos")
    )


def seed_everything(seed: int, *, torch_module: Any | None = None) -> None:
    """Seed every RNG the framework can influence.

    This must run *before* a model is constructed.  Seeding inside the
    training loop was too late: PyTorch had already drawn the initial weights,
    so two runs of the same ``ExperimentSpec`` started from different points
    and training was not reproducible.
    """

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer.")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    torch = torch_module
    if torch is None:
        try:
            torch = import_module("torch")
        except ImportError:
            return
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - no GPU in core CI
        torch.cuda.manual_seed_all(seed)


@contextmanager
def deterministic_algorithms(
    enabled: bool,
    *,
    warn_only: bool = True,
    torch_module: Any | None = None,
) -> Iterator[None]:
    """Scope PyTorch's global determinism flag to a single block.

    ``torch.use_deterministic_algorithms`` is process-global state.  Setting
    it inside ``fit()`` and never restoring it leaked into every later
    workload in the same interpreter -- including unrelated libraries that
    then raised on non-deterministic kernels.  The flag is now restored on
    exit, exceptions included.
    """

    torch = torch_module
    if torch is None:
        try:
            torch = import_module("torch")
        except ImportError:
            yield
            return
    if not enabled:
        yield
        return
    previous = torch.are_deterministic_algorithms_enabled()
    warn_probe = getattr(torch, "is_deterministic_algorithms_warn_only_enabled", None)
    previous_warn = bool(warn_probe()) if callable(warn_probe) else False
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn)


__all__ = [
    "qiskit_aer_devices",
    "qiskit_aer_methods",
    "qiskit_gpu_available",
    "pennylane_registered_devices",
    "pennylane_device_constructible",
    "pennylane_gpu_available",
    "seed_everything",
    "deterministic_algorithms",
]
