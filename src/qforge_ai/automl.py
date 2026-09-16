"""Optional Optuna-based search orchestration for QuantumForge experiments."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .backends._optional import require
from .errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class SearchParameter:
    kind: str
    low: float | int | None = None
    high: float | int | None = None
    choices: tuple[Any, ...] = ()
    log: bool = False
    step: float | int | None = None

    def suggest(self, trial: Any, name: str) -> Any:
        if self.kind == "categorical":
            if not self.choices:
                raise ConfigurationError(f"{name} categorical choices cannot be empty.")
            return trial.suggest_categorical(name, list(self.choices))
        if self.low is None or self.high is None:
            raise ConfigurationError(f"low/high are required for {name}.")
        if self.kind == "int":
            return trial.suggest_int(
                name,
                int(self.low),
                int(self.high),
                step=int(self.step or 1),
                log=self.log,
            )
        if self.kind == "float":
            kwargs: dict[str, Any] = {"log": self.log}
            if self.step is not None:
                kwargs["step"] = float(self.step)
            return trial.suggest_float(name, float(self.low), float(self.high), **kwargs)
        raise ConfigurationError(f"Unknown search parameter kind: {self.kind}")


@dataclass(frozen=True, slots=True)
class QuantumSearchSpace:
    parameters: Mapping[str, SearchParameter] = field(default_factory=dict)

    def suggest(self, trial: Any) -> dict[str, Any]:
        return {name: parameter.suggest(trial, name) for name, parameter in self.parameters.items()}


class AutoMLStudy:
    """Thin, real Optuna integration with explicit objective/config boundary."""

    def __init__(
        self,
        search_space: QuantumSearchSpace,
        *,
        direction: str = "minimize",
        study_name: str | None = None,
        storage: str | None = None,
        seed: int = 42,
    ) -> None:
        if direction not in {"minimize", "maximize"}:
            raise ConfigurationError("Optuna direction must be minimize or maximize.")
        self.search_space = search_space
        self.direction = direction
        self.study_name = study_name
        self.storage = storage
        self.seed = seed

    def optimize(
        self,
        objective: Callable[[dict[str, Any], Any], float],
        *,
        n_trials: int = 20,
        timeout: float | None = None,
        callbacks: tuple[Callable[[Any, Any], None], ...] = (),
    ) -> Any:
        if n_trials < 1:
            raise ConfigurationError("n_trials must be at least 1.")
        optuna = require("optuna", package="optuna", extra="automl", feature="AutoML search")
        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(
            direction=self.direction,
            study_name=self.study_name,
            storage=self.storage,
            load_if_exists=self.storage is not None,
            sampler=sampler,
        )

        def wrapped(trial: Any) -> float:
            return float(objective(self.search_space.suggest(trial), trial))

        study.optimize(wrapped, n_trials=n_trials, timeout=timeout, callbacks=list(callbacks))
        return study


class OptunaSearch(AutoMLStudy):
    """Deprecated 1.1 name retained as a source-compatible facade."""

    def run(
        self,
        objective: Callable[[dict[str, Any], Any], float],
        *,
        n_trials: int = 20,
        timeout: float | None = None,
        callbacks: tuple[Callable[[Any, Any], None], ...] = (),
    ) -> Any:
        return self.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            callbacks=callbacks,
        )


__all__ = ["SearchParameter", "QuantumSearchSpace", "AutoMLStudy", "OptunaSearch"]
