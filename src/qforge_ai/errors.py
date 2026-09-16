"""Framework-specific exception hierarchy."""


class QuantumForgeError(Exception):
    """Base exception for all framework failures."""


class ConfigurationError(QuantumForgeError, ValueError):
    """Raised when a specification is internally inconsistent."""


class CapabilityError(QuantumForgeError):
    """Raised when a backend cannot satisfy a requested capability."""


class MissingDependencyError(QuantumForgeError, ImportError):
    """Raised only when an optional integration is actually used."""

    def __init__(self, package: str, extra: str, feature: str) -> None:
        super().__init__(
            f"The optional '{package}' package is required for {feature}. "
            f"Install it with: pip install 'quantumforge-ai[{extra}]'"
        )


class RegistryError(QuantumForgeError):
    """Raised for unknown, duplicate or invalid catalog entries."""


class DuplicateComponentError(RegistryError):
    """Raised when a component key or alias is already owned."""


class DuplicateFactoryError(RegistryError):
    """Raised when a component already has an implementation factory."""


class TranslationError(QuantumForgeError):
    """Raised when an IR operation cannot be translated to an engine."""


class ExecutionError(QuantumForgeError):
    """Raised when compilation or simulation fails."""
