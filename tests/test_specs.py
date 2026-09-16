import json

import pytest

from qforge_ai.backends import BackendRouter
from qforge_ai.capabilities import CapabilityRegistry
from qforge_ai.errors import CapabilityError, ConfigurationError
from qforge_ai.ir import CircuitIR
from qforge_ai.specs import AnsatzSpec, BackendSpec, EncodingSpec, Engine, ExperimentSpec, ModelSpec


def test_yaml_round_trip(tmp_path) -> None:
    config = {
        "name": "roundtrip",
        "circuit": {
            "num_qubits": 3,
            "encoding": {"name": "angle", "feature_dimension": 3},
            "ansatz": {"name": "real_amplitudes", "layers": 2},
            "measurement": {"name": "expectation", "wires": [0, 1, 2]},
        },
        "backend": {"engine": "pennylane", "gradient": "parameter_shift"},
        "model": {"mode": "hybrid", "hidden_dimensions": [8]},
    }
    spec = ExperimentSpec.from_dict(config)
    path = tmp_path / "config.yaml"
    spec.to_yaml(path)
    loaded = ExperimentSpec.from_yaml(path)
    assert loaded == spec
    assert loaded.circuit.measurement.wires == (0, 1, 2)


def test_specs_are_hashable_and_recursively_immutable() -> None:
    spec = AnsatzSpec(
        entanglement="user_defined", options={"custom_edges": [[0, 1], [1, 2]]}
    )
    assert isinstance(hash(spec), int)
    assert spec.options["custom_edges"] == ((0, 1), (1, 2))
    with pytest.raises(TypeError):
        spec.options["x"] = 1  # type: ignore[index]


def test_unknown_option_keys_are_rejected_and_plugins_can_declare_their_own() -> None:
    """A key nothing reads is a typo, not a forward-compatible extension."""

    from qforge_ai.specs import SPEC_OPTION_KEYS, register_option_keys

    with pytest.raises(ConfigurationError, match="does not use"):
        EncodingSpec(options={"scalling": 2.0})
    try:
        register_option_keys("encoding", "plugin_depth")
        assert EncodingSpec(options={"plugin_depth": 3}).options["plugin_depth"] == 3
    finally:
        SPEC_OPTION_KEYS["encoding"].discard("plugin_depth")


def test_layers_zero_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="at least 1"):
        AnsatzSpec(layers=0)


def test_experiment_spec_serialization_is_primitive() -> None:
    spec = ExperimentSpec.from_dict(
        {"name": "json", "circuit": {"num_qubits": 2}, "model": {"template": "auto"}}
    )
    json.dumps(spec.to_dict())
    assert hash(ModelSpec(options={"branches": 2}))


def test_backend_device_and_compute_device_are_distinct() -> None:
    backend = BackendSpec(engine=Engine.QISKIT, device="aer", compute_device="gpu")
    assert backend.device == "aer"
    assert backend.compute_device == "GPU"
    with pytest.raises(ConfigurationError, match="aer"):
        BackendSpec(engine=Engine.QISKIT, device="ignored_backend")


def test_qiskit_gpu_capability_is_runtime_detected(monkeypatch) -> None:
    import qforge_ai.capabilities as capabilities_module

    monkeypatch.setattr(capabilities_module, "qiskit_gpu_available", lambda: False)
    assert CapabilityRegistry().get(Engine.QISKIT).gpu is False
    with pytest.raises(CapabilityError, match="GPU was requested"):
        BackendRouter().recommend(CircuitIR(16), gpu=True)
