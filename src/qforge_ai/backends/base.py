"""Backend adapter protocol and normalized result model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np

from ..ir import CircuitIR


@dataclass(slots=True)
class ExecutionResult:
    engine: str
    values: Any
    runtime_seconds: float
    shots: int | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_array(self) -> np.ndarray:
        if isinstance(self.values, Mapping):
            return np.asarray(list(self.values.values()), dtype=float)
        return np.asarray(self.values)


class BackendAdapter(ABC):
    engine: str

    @abstractmethod
    def compile(self, circuit: CircuitIR, **kwargs: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def run(
        self,
        circuit: CircuitIR,
        *,
        inputs: Iterable[float] = (),
        weights: Iterable[float] = (),
        **kwargs: Any,
    ) -> ExecutionResult:
        raise NotImplementedError

    def benchmark(self, circuit: CircuitIR, **kwargs: Any) -> ExecutionResult:
        started = perf_counter()
        result = self.run(circuit, **kwargs)
        result.metadata["wall_clock_with_overhead"] = perf_counter() - started
        return result
