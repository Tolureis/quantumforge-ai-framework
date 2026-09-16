"""Third-party plugin discovery through Python entry points."""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Any

from .catalog import catalog


def load_plugins(group: str = "qforge_ai.plugins") -> list[str]:
    """Load and register every plugin published under *group*.

    A plugin may be a callable taking the registry, or an object, module or
    class exposing ``register(registry)``.  The ``register`` attribute is
    checked first because classes and modules are themselves callable:
    testing ``callable()`` first made the documented ``register()`` contract
    unreachable for class-based plugins, which failed with
    ``TypeError: Module() takes no arguments``.
    """

    loaded: list[str] = []
    for entry_point in entry_points(group=group):
        plugin = entry_point.load()
        register = getattr(plugin, "register", None)
        if callable(register):
            register(catalog)
        elif callable(plugin):
            plugin(catalog)
        else:
            raise TypeError(
                f"The plugin cannot be registered: {entry_point.name}. The entry point must be "
                "either a callable that takes a registry or an object that provides "
                "register(registry)."
            )
        loaded.append(entry_point.name)
    return loaded


def register_factory(key: str, factory: Any, *, replace: bool = False) -> None:
    catalog.register_factory(key, factory, replace=replace)


__all__ = ["load_plugins", "register_factory"]
