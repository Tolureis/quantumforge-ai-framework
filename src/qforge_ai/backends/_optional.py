"""Helpers for optional dependency boundaries."""

from __future__ import annotations

import importlib
from types import ModuleType

from ..errors import MissingDependencyError


def require(module: str, *, package: str, extra: str, feature: str) -> ModuleType:
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise MissingDependencyError(package, extra, feature) from exc
