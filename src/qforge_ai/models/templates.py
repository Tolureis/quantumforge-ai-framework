"""Engine-neutral composition helpers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ..backends._optional import require
from ..errors import ConfigurationError

#: Reductions ``QuantumParallel`` implements.
PARALLEL_REDUCTIONS = frozenset({"concat", "mean", "sum"})


class QuantumSequential:
    @staticmethod
    def build(*layers: Any) -> Any:
        nn = require("torch.nn", package="torch", extra="torch", feature="QuantumSequential")
        if not layers:
            raise ConfigurationError("QuantumSequential requires at least one layer.")
        return nn.Sequential(*layers)


class QuantumParallel:
    @staticmethod
    def build(branches: Iterable[Any], *, reduction: str = "concat") -> Any:
        torch = require("torch", package="torch", extra="torch", feature="QuantumParallel")
        nn = require("torch.nn", package="torch", extra="torch", feature="QuantumParallel")
        # An unrecognised reduction used to fall through to ``concat``, so a
        # typo silently changed the model's output width instead of failing.
        if reduction not in PARALLEL_REDUCTIONS:
            raise ConfigurationError(
                f"Unknown parallel reduction: {reduction}. "
                f"Supported: {sorted(PARALLEL_REDUCTIONS)}"
            )
        resolved_branches = list(branches)
        if not resolved_branches:
            raise ConfigurationError("QuantumParallel requires at least one branch.")

        class Parallel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.branches = nn.ModuleList(resolved_branches)

            def forward(self, inputs: Any) -> Any:
                outputs = [branch(inputs) for branch in self.branches]
                if reduction == "mean":
                    return torch.stack(outputs, dim=0).mean(dim=0)
                if reduction == "sum":
                    return torch.stack(outputs, dim=0).sum(dim=0)
                return torch.cat(outputs, dim=-1)

        return Parallel()


class QuantumResidual:
    @staticmethod
    def build(layer: Any, *, projection: Any | None = None) -> Any:
        nn = require("torch.nn", package="torch", extra="torch", feature="QuantumResidual")

        class Residual(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer = layer
                self.projection = projection or nn.Identity()

            def forward(self, inputs: Any) -> Any:
                return self.layer(inputs) + self.projection(inputs)

        return Residual()


__all__ = ["QuantumSequential", "QuantumParallel", "QuantumResidual"]
