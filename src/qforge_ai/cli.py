"""Dependency-light command line interface."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .backends._optional import require
from .catalog import catalog
from .experiment import ExperimentRunner
from .specs import ExperimentSpec, to_primitive


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qforge", description="QuantumForge AI CLI")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    catalog_parser = sub.add_parser("catalog", help="List the catalog components")
    catalog_parser.add_argument("--category")
    catalog_parser.add_argument("--engine", choices=("qiskit", "pennylane", "ir"))
    catalog_parser.add_argument(
        "--status", choices=("native", "composite", "experimental", "descriptor")
    )
    catalog_parser.add_argument("--search")
    catalog_parser.add_argument("--json", action="store_true")
    catalog_parser.add_argument(
        "--report",
        action="store_true",
        help="Shows how many records are actually executable",
    )

    validate = sub.add_parser("validate", help="Validate an experiment YAML file")
    validate.add_argument("config", type=Path)

    inspect = sub.add_parser("inspect", help="Show the circuit resource report")
    inspect.add_argument("config", type=Path)
    inspect.add_argument("--json", action="store_true")

    run = sub.add_parser("run", help="Perform a single circuit execution")
    run.add_argument("config", type=Path)
    run.add_argument("--inputs", required=True, help="Comma-separated input values")
    run.add_argument("--weights", required=True, help="Comma-separated weight values")

    train = sub.add_parser("train", help="Train the hybrid model with an NPZ dataset")
    train.add_argument("config", type=Path)
    train.add_argument("dataset", type=Path, help="NPZ containing the X_train/y_train keys")
    train.add_argument("--validation-data", type=Path)
    train.add_argument("--device", default="cpu")

    sub.add_parser("doctor", help="Check the optional dependencies")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "catalog":
            if args.report:
                return _catalog_report(as_json=args.json)
            items = (
                catalog.search(args.search)
                if args.search
                else catalog.list(category=args.category, engine=args.engine, status=args.status)
            )
            if args.json:
                print(
                    json.dumps([_descriptor(item) for item in items], indent=2, ensure_ascii=False)
                )
            else:
                for item in items:
                    engines = ",".join(sorted(map(str, item.engines)))
                    print(f"{item.key:42} {item.category:22} {item.status:12} {engines}")
                print(f"Total: {len(items)}")
            return 0
        if args.command == "validate":
            spec = ExperimentSpec.from_yaml(args.config)
            prepared = ExperimentRunner().prepare(spec)
            print(f"Valid: {spec.name}")
            for warning in prepared.warnings:
                print(f"WARNING: {warning}")
            return 0
        if args.command == "inspect":
            prepared = ExperimentRunner().prepare(args.config)
            data = prepared.resources.to_dict()
            if args.json:
                print(json.dumps(data, indent=2, ensure_ascii=False))
            else:
                for key, value in data.items():
                    print(f"{key}: {value}")
            return 0
        if args.command == "run":
            prepared = ExperimentRunner().prepare(args.config)
            inputs = _floats(args.inputs)
            weights = _floats(args.weights)
            execution = ExperimentRunner().run_circuit(prepared, inputs=inputs, weights=weights)
            print(
                json.dumps(
                    {
                        "engine": execution.engine,
                        "values": _jsonable(execution.values),
                        "runtime_seconds": execution.runtime_seconds,
                    },
                    indent=2,
                )
            )
            return 0
        if args.command == "train":
            runner = ExperimentRunner()
            prepared = runner.prepare(args.config)
            train_loader, validation_loader = _load_npz_data(
                args.dataset,
                args.validation_data,
                batch_size=prepared.spec.training.batch_size,
                seed=prepared.spec.seed,
                loss=prepared.spec.training.loss,
            )
            training_result = runner.train(
                prepared,
                train_loader,
                validation_loader,
                device=args.device,
            )
            print(
                json.dumps(
                    {
                        "best_epoch": training_result.best_epoch,
                        "best_validation_loss": training_result.best_validation_loss,
                        "stopped_early": training_result.stopped_early,
                        "history": [asdict(item) for item in training_result.history],
                        "metadata": training_result.metadata,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return 0
        if args.command == "doctor":
            return _doctor()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 1


def _catalog_report(*, as_json: bool) -> int:
    """Print how much of the catalog the core framework can actually execute.

    The total record count is a discovery surface, not a capability claim; this
    view separates the two so the distinction is visible without reading the
    documentation.
    """

    from .catalog import capability_report

    report = capability_report()
    if as_json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    print(f"Total records            : {report['total']}")
    print(f"Executable               : {report['executable']}")
    print(f"Discovery/plugin contract: {report['discovery_only']}")
    print("\nBy status:")
    for status, count in report["by_status"].items():
        print(f"  {status:14} {count:4}")
    print("\nBy category (executable/total):")
    for category, bucket in report["by_category"].items():
        print(f"  {category:22} {bucket['executable']:3}/{bucket['total']:3}")
    return 0


def _descriptor(item: Any) -> dict[str, Any]:
    data = to_primitive(item)
    if not isinstance(data, dict):
        raise TypeError("ComponentDescriptor could not be converted to a JSON object.")
    data["engines"] = sorted(data["engines"])
    data["capabilities"] = sorted(data["capabilities"])
    return data


def _floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    return to_primitive(value)


def _doctor() -> int:
    import importlib.util

    modules = (
        "numpy",
        "yaml",
        "qiskit",
        "qiskit_aer",
        "qiskit_machine_learning",
        "pennylane",
        "torch",
    )
    for module in modules:
        print(f"{module:28} {'OK' if importlib.util.find_spec(module) else 'YOK'}")
    return 0


def _load_npz_data(
    train_path: Path,
    validation_path: Path | None,
    *,
    batch_size: int,
    seed: int,
    loss: str,
) -> tuple[Any, Any | None]:
    torch = require("torch", package="torch", extra="torch", feature="qforge train")
    data = require("torch.utils.data", package="torch", extra="torch", feature="qforge train")

    def arrays(path: Path, *, validation: bool) -> tuple[np.ndarray, np.ndarray] | None:
        with np.load(path, allow_pickle=False) as archive:
            feature_keys = (
                ("X_val", "x_val", "validation_inputs")
                if validation
                else ("X_train", "x_train", "train_inputs", "X", "x")
            )
            label_keys = (
                ("y_val", "Y_val", "validation_labels")
                if validation
                else ("y_train", "Y_train", "train_labels", "y", "Y")
            )
            feature_key = next((key for key in feature_keys if key in archive), None)
            label_key = next((key for key in label_keys if key in archive), None)
            if feature_key is None or label_key is None:
                if validation:
                    return None
                raise ValueError(
                    f"{path} must contain X_train/y_train (or train_inputs/train_labels)."
                )
            return np.asarray(archive[feature_key]), np.asarray(archive[label_key])

    def loader(pair: tuple[np.ndarray, np.ndarray], *, shuffle: bool) -> Any:
        features, labels = pair
        if len(features) != len(labels):
            raise ValueError(f"Sample/label counts are not equal: {len(features)} != {len(labels)}")
        feature_tensor = torch.as_tensor(features, dtype=torch.float32)
        label_dtype = torch.long if loss.lower() in {"cross_entropy", "ce"} else torch.float32
        label_tensor = torch.as_tensor(labels, dtype=label_dtype)
        generator = torch.Generator().manual_seed(seed)
        return data.DataLoader(
            data.TensorDataset(feature_tensor, label_tensor),
            batch_size=batch_size,
            shuffle=shuffle,
            generator=generator,
        )

    training = arrays(train_path, validation=False)
    assert training is not None
    validation = (
        arrays(validation_path, validation=True)
        if validation_path is not None
        else arrays(train_path, validation=True)
    )
    return loader(training, shuffle=True), loader(validation, shuffle=False) if validation else None


if __name__ == "__main__":
    raise SystemExit(main())
