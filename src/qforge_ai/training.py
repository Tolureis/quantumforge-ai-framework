"""Reproducible PyTorch training loop with callbacks and early stopping."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np

from .backends._optional import require
from .errors import ConfigurationError
from .runtime import deterministic_algorithms, seed_everything
from .specs import TrainingSpec, to_primitive


@dataclass(slots=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    validation_loss: float | None
    learning_rate: float
    runtime_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "validation_loss": self.validation_loss,
            "learning_rate": self.learning_rate,
            "runtime_seconds": self.runtime_seconds,
        }


@dataclass(slots=True)
class TrainingResult:
    history: list[EpochMetrics]
    best_epoch: int
    best_validation_loss: float | None
    stopped_early: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the serializable 1.1-compatible training payload."""

        return {
            "history": [metric.to_dict() for metric in self.history],
            "best_epoch": self.best_epoch,
            "best_validation_loss": self.best_validation_loss,
            "stopped_early": self.stopped_early,
            "metadata": to_primitive(self.metadata),
        }


@dataclass(slots=True)
class TrainingRun(TrainingResult):
    """Backward-compatible train return that is also a ``TrainingResult``.

    1.1 callers can access ``run.model`` and ``run.result`` while 1.2 callers
    continue to use ``history`` and the other ``TrainingResult`` attributes
    directly.
    """

    model: Any = field(default=None, repr=False)

    @property
    def result(self) -> TrainingResult:
        return self

    @classmethod
    def from_result(cls, result: TrainingResult, *, model: Any) -> TrainingRun:
        return cls(
            history=result.history,
            best_epoch=result.best_epoch,
            best_validation_loss=result.best_validation_loss,
            stopped_early=result.stopped_early,
            metadata=result.metadata,
            model=model,
        )


class TorchTrainer:
    def __init__(self, spec: TrainingSpec, *, seed: int = 42) -> None:
        self.spec = spec
        self.seed = seed

    def fit(
        self,
        model: Any,
        train_loader: Iterable[Any],
        *,
        validation_loader: Iterable[Any] | None = None,
        callbacks: Iterable[Callable[[EpochMetrics], None]] = (),
        device: str = "cpu",
    ) -> TrainingResult:
        torch = require("torch", package="torch", extra="torch", feature="TorchTrainer")
        # The determinism flag is process-global; scoping it here keeps a
        # QuantumForge run from silently reconfiguring the rest of the
        # interpreter.  Weight initialization is seeded earlier, at model
        # construction, because seeding here would already be too late.
        with deterministic_algorithms(self.spec.deterministic, torch_module=torch):
            return self._fit(
                model,
                train_loader,
                validation_loader=validation_loader,
                callbacks=callbacks,
                device=device,
                torch=torch,
            )

    def _fit(
        self,
        model: Any,
        train_loader: Iterable[Any],
        *,
        validation_loader: Iterable[Any] | None,
        callbacks: Iterable[Callable[[EpochMetrics], None]],
        device: str,
        torch: Any,
    ) -> TrainingResult:
        nn = require("torch.nn", package="torch", extra="torch", feature="TorchTrainer")
        self._seed_everything(torch)
        model.to(device)
        optimizer = self._optimizer(torch, model)
        criterion = self._loss(nn)
        history: list[EpochMetrics] = []
        best_loss = float("inf")
        best_epoch = 0
        stale_epochs = 0
        best_state: dict[str, Any] | None = None
        for epoch in range(1, self.spec.epochs + 1):
            started = perf_counter()
            model.train()
            losses: list[float] = []
            for features, labels in train_loader:
                features, labels = features.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                output = model(features)
                loss = criterion(output, labels)
                loss.backward()
                if self.spec.gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.spec.gradient_clip)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            if not losses:
                raise ConfigurationError("train_loader must produce at least one batch.")
            validation_loss = self._validate(model, validation_loader, criterion, device, torch)
            monitored = validation_loss if validation_loss is not None else float(np.mean(losses))
            if monitored < best_loss:
                best_loss, best_epoch, stale_epochs = monitored, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale_epochs += 1
            metrics = EpochMetrics(
                epoch=epoch,
                train_loss=float(np.mean(losses)),
                validation_loss=validation_loss,
                learning_rate=float(optimizer.param_groups[0]["lr"]),
                runtime_seconds=perf_counter() - started,
            )
            history.append(metrics)
            for callback in callbacks:
                callback(metrics)
            patience = self.spec.early_stopping_patience
            if patience is not None and stale_epochs >= patience:
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        return TrainingResult(
            history=history,
            best_epoch=best_epoch,
            best_validation_loss=None if validation_loader is None else best_loss,
            stopped_early=len(history) < self.spec.epochs,
            metadata={"seed": self.seed, "device": device},
        )

    def _optimizer(self, torch: Any, model: Any) -> Any:
        kwargs = {"lr": self.spec.learning_rate, "weight_decay": self.spec.weight_decay}
        name = self.spec.optimizer.lower()
        if name == "adam":
            return torch.optim.Adam(model.parameters(), **kwargs)
        if name == "adamw":
            return torch.optim.AdamW(model.parameters(), **kwargs)
        if name == "sgd":
            return torch.optim.SGD(model.parameters(), momentum=0.9, **kwargs)
        if name == "rmsprop":
            return torch.optim.RMSprop(model.parameters(), **kwargs)
        raise ConfigurationError(f"Unsupported optimizer: {self.spec.optimizer}")

    def _loss(self, nn: Any) -> Any:
        name = self.spec.loss.lower()
        if name in {"cross_entropy", "ce"}:
            return nn.CrossEntropyLoss()
        if name in {"mse", "mean_squared_error"}:
            return nn.MSELoss()
        if name in {"bce", "binary_cross_entropy"}:
            return nn.BCEWithLogitsLoss()
        raise ConfigurationError(f"Unsupported loss: {self.spec.loss}")

    @staticmethod
    def _validate(model: Any, loader: Any, criterion: Any, device: str, torch: Any) -> float | None:
        if loader is None:
            return None
        model.eval()
        losses: list[float] = []
        with torch.no_grad():
            for features, labels in loader:
                output = model(features.to(device))
                losses.append(float(criterion(output, labels.to(device)).cpu()))
        if not losses:
            raise ConfigurationError("validation_loader must produce at least one batch.")
        return float(np.mean(losses))

    def _seed_everything(self, torch: Any) -> None:
        """Re-seed data-order and dropout RNGs at the start of the loop.

        Determinism of the *global* torch flag is handled by ``fit`` through a
        scoped context manager, and weight initialization is seeded before the
        model exists -- see :func:`qforge_ai.runtime.seed_everything`.
        """

        seed_everything(self.seed, torch_module=torch)


__all__ = ["TorchTrainer", "TrainingResult", "TrainingRun", "EpochMetrics"]
