"""Thread-safe metadata and implementation registry."""

from __future__ import annotations

import builtins
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from dataclasses import replace as replace_dataclass
from threading import RLock
from typing import Any

from .errors import DuplicateComponentError, DuplicateFactoryError, RegistryError
from .specs import ComponentStatus, Engine, freeze_value

Factory = Callable[..., Any]


@dataclass(frozen=True, slots=True)
class ComponentDescriptor:
    key: str
    display_name: str
    category: str
    description: str
    status: ComponentStatus = ComponentStatus.DESCRIPTOR
    engines: frozenset[Engine] = frozenset({Engine.IR})
    capabilities: frozenset[str] = frozenset()
    aliases: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    references: tuple[str, ...] = ()
    implementation: str = "descriptor"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", freeze_value(self.metadata))


class ComponentRegistry:
    def __init__(self) -> None:
        self._descriptors: dict[str, ComponentDescriptor] = {}
        self._factories: dict[str, Factory] = {}
        self._aliases: dict[str, str] = {}
        self._lock = RLock()

    def register(
        self,
        descriptor: ComponentDescriptor,
        factory: Factory | None = None,
        *,
        replace: bool = False,
    ) -> None:
        if factory is not None:
            descriptor = _with_factory_contract(descriptor)
        key = normalize_key(descriptor.key)
        if not key:
            raise RegistryError("Component key cannot be empty.")
        normalized_aliases = tuple(normalize_key(alias) for alias in (key, *descriptor.aliases))
        if any(not alias for alias in normalized_aliases):
            raise RegistryError("Component alias values cannot be empty.")
        if len(set(normalized_aliases)) != len(normalized_aliases):
            raise DuplicateComponentError(
                f"Duplicate alias within the same descriptor: {descriptor.aliases}"
            )
        with self._lock:
            if key in self._descriptors and not replace:
                raise DuplicateComponentError(f"Component already registered: {key}")
            # Validate the entire prospective registration before mutating any
            # live map.  In particular, a late alias conflict must not leak the
            # aliases that happened to be visited first.
            for alias, normalized in zip(
                (key, *descriptor.aliases), normalized_aliases, strict=True
            ):
                owner = self._aliases.get(normalized)
                if owner is not None and owner != key:
                    raise DuplicateComponentError(f"Alias conflict: {alias} ({owner}, {key})")

            descriptors = dict(self._descriptors)
            factories = dict(self._factories)
            aliases = dict(self._aliases)
            if replace and key in descriptors:
                old = descriptors[key]
                for alias in (key, *old.aliases):
                    normalized = normalize_key(alias)
                    if aliases.get(normalized) == key:
                        aliases.pop(normalized, None)
                if factory is None:
                    factories.pop(key, None)
            aliases.update(dict.fromkeys(normalized_aliases, key))
            descriptors[key] = descriptor
            if factory is not None:
                factories[key] = factory
            self._descriptors = descriptors
            self._factories = factories
            self._aliases = aliases

    def register_factory(
        self,
        key: str,
        factory: Factory,
        *,
        replace: bool = False,
        ignore_existing: bool = False,
    ) -> None:
        with self._lock:
            canonical = self._resolve_key_unlocked(key)
            if canonical in self._factories and not replace:
                if ignore_existing:
                    return
                raise DuplicateFactoryError(f"Factory already registered: {canonical}")
            self._factories[canonical] = factory
            self._descriptors[canonical] = _with_factory_contract(
                self._descriptors[canonical]
            )

    def get_factory(self, key: str) -> Factory:
        """Return the registered implementation for a descriptor or fail explicitly."""

        with self._lock:
            canonical = self._resolve_key_unlocked(key)
            try:
                return self._factories[canonical]
            except KeyError as exc:
                status = self._descriptors[canonical].status
                raise RegistryError(
                    f"'{canonical}' is registered in the catalog as {status} but has no "
                    "executable factory."
                ) from exc

    def annotate(self, key: str, **metadata: Any) -> None:
        """Merge scientific/runtime metadata without replacing descriptor identity."""

        with self._lock:
            canonical = self._resolve_key_unlocked(key)
            descriptor = self._descriptors[canonical]
            self._descriptors[canonical] = replace_dataclass(
                descriptor,
                metadata={**dict(descriptor.metadata), **metadata},
            )

    def has_factory(self, key: str) -> bool:
        with self._lock:
            canonical = self._resolve_key_unlocked(key)
            return canonical in self._factories

    def is_runnable(self, key: str) -> bool:
        descriptor = self.get(key)
        return self.has_factory(key) or descriptor.implementation in {
            "builtin_path",
            "model_factory",
        }

    def resolve_key(self, key: str) -> str:
        with self._lock:
            return self._resolve_key_unlocked(key)

    def _resolve_key_unlocked(self, key: str) -> str:
        normalized = normalize_key(key)
        try:
            return self._aliases[normalized]
        except KeyError as exc:
            suggestions = self._search_unlocked(normalized, limit=5)
            hint = f" Close matches: {[s.key for s in suggestions]}" if suggestions else ""
            raise RegistryError(f"Unknown component: {key}.{hint}") from exc

    def get(self, key: str) -> ComponentDescriptor:
        with self._lock:
            return self._descriptors[self._resolve_key_unlocked(key)]

    def create(self, key: str, **kwargs: Any) -> Any:
        return self.get_factory(key)(**kwargs)

    def list(
        self,
        *,
        category: str | None = None,
        engine: Engine | str | None = None,
        status: ComponentStatus | str | None = None,
        capability: str | None = None,
    ) -> builtins.list[ComponentDescriptor]:
        engine_value = Engine(engine) if engine is not None else None
        status_value = ComponentStatus(status) if status is not None else None
        with self._lock:
            items = tuple(self._descriptors.values())
            return sorted(
                (
                    item
                    for item in items
                    if (category is None or item.category == category)
                    and (engine_value is None or engine_value in item.engines)
                    and (status_value is None or item.status == status_value)
                    and (capability is None or capability in item.capabilities)
                ),
                key=lambda item: (item.category, item.key),
            )

    def search(self, text: str, *, limit: int = 20) -> builtins.list[ComponentDescriptor]:
        with self._lock:
            return self._search_unlocked(text, limit=limit)

    def _search_unlocked(
        self, text: str, *, limit: int = 20
    ) -> builtins.list[ComponentDescriptor]:
        needle = normalize_key(text)
        ranked: builtins.list[tuple[int, ComponentDescriptor]] = []
        for item in self._descriptors.values():
            haystack = " ".join(
                (item.key, item.display_name, item.description, *item.aliases)
            ).lower()
            if needle in normalize_key(haystack):
                score = 0 if item.key.startswith(needle) else 1
                ranked.append((score, item))
        return [item for _, item in sorted(ranked, key=lambda pair: (pair[0], pair[1].key))[:limit]]

    def categories(self) -> builtins.list[str]:
        with self._lock:
            return sorted({item.category for item in self._descriptors.values()})

    def __len__(self) -> int:
        with self._lock:
            return len(self._descriptors)

    def __iter__(self) -> Iterable[ComponentDescriptor]:
        return iter(self.list())


def normalize_key(value: str) -> str:
    return "_".join(value.strip().lower().replace("-", "_").replace("/", "_").split())


def _with_factory_contract(descriptor: ComponentDescriptor) -> ComponentDescriptor:
    return replace_dataclass(
        descriptor,
        status=ComponentStatus.NATIVE,
        implementation="registry_factory",
        metadata={
            **dict(descriptor.metadata),
            "factory_registered": True,
            "implementation": "registry_factory",
            "runtime_contract": "runnable",
        },
    )
