import importlib.util
import json

import numpy as np
import pytest

from qforge_ai.cli import main
from qforge_ai.specs import ExperimentSpec


def test_catalog_json_serializes_immutable_descriptor_metadata(capsys) -> None:
    assert main(["catalog", "--category", "qnn_model", "--json"]) == 0
    output = json.loads(capsys.readouterr().out)
    sampler = next(item for item in output if item["key"] == "sampler_qnn")
    assert sampler["metadata"]["runtime_contract"] == "runnable"
    assert sampler["engines"] == sorted(sampler["engines"])


@pytest.mark.skipif(
    any(importlib.util.find_spec(module) is None for module in ("qiskit", "qiskit_aer")),
    reason="qiskit aer optional dependency is not installed",
)
def test_cli_run_with_qiskit_optional_backend(tmp_path, capsys) -> None:
    config = tmp_path / "qiskit.yaml"
    ExperimentSpec.from_dict(
        {
            "name": "cli-qiskit",
            "circuit": {
                "num_qubits": 1,
                "encoding": {"name": "angle", "feature_dimension": 1},
                "ansatz": {"name": "real_amplitudes", "layers": 1},
                "measurement": {"name": "expectation", "wires": [0]},
            },
            "backend": {"engine": "qiskit", "device": "aer"},
        }
    ).to_yaml(config)
    assert main(["run", str(config), "--inputs", "0.2", "--weights", "0.0,0.0"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["engine"] == "qiskit"


@pytest.mark.skipif(
    any(importlib.util.find_spec(module) is None for module in ("pennylane", "torch")),
    reason="pennylane and torch optional dependencies are not installed",
)
def test_cli_train_with_pennylane_optional_backend(tmp_path, capsys) -> None:
    config = tmp_path / "pennylane.yaml"
    dataset = tmp_path / "dataset.npz"
    ExperimentSpec.from_dict(
        {
            "name": "cli-pennylane-train",
            "circuit": {
                "num_qubits": 1,
                "encoding": {"name": "angle", "feature_dimension": 1},
                "ansatz": {"name": "real_amplitudes", "layers": 1},
                "measurement": {"name": "expectation", "wires": [0]},
            },
            "backend": {"engine": "pennylane", "device": "default.qubit"},
            "model": {
                "mode": "hybrid",
                "template": "classical_quantum_classical",
                "input_dimension": 1,
                "output_dimension": 2,
            },
            "training": {"epochs": 1, "batch_size": 2},
        }
    ).to_yaml(config)
    np.savez(
        dataset,
        X_train=np.asarray([[0.0], [0.2], [0.8], [1.0]], dtype=np.float32),
        y_train=np.asarray([0, 0, 1, 1], dtype=np.int64),
    )
    assert main(["train", str(config), str(dataset)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert len(output["history"]) == 1


def test_cli_catalog_report_separates_executable_from_discovery(capsys) -> None:
    from qforge_ai.catalog import capability_report

    assert main(["catalog", "--report"]) == 0
    output = capsys.readouterr().out
    report = capability_report()
    assert str(report["total"]) in output
    assert str(report["executable"]) in output
    assert "Discovery/plugin contract" in output


def test_cli_catalog_report_json_matches_the_api(capsys) -> None:
    import json as json_module

    from qforge_ai.catalog import capability_report

    assert main(["catalog", "--report", "--json"]) == 0
    assert json_module.loads(capsys.readouterr().out) == capability_report()
