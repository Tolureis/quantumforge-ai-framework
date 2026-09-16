"""Regression tests for the defects reported against 1.2.1.

Every test here fails on 1.2.1 and passes on 1.2.2.  They are grouped by the
numbered finding so a future change that reintroduces one is attributed
immediately.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from qforge_ai import (
    AnsatzSpec,
    BackendSpec,
    CircuitSpec,
    EncodingSpec,
    MeasurementSpec,
    ModelSpec,
    NoiseSpec,
)
from qforge_ai.capabilities import CapabilityRegistry
from qforge_ai.catalog import capability_report, catalog, is_executable
from qforge_ai.circuits import CircuitFactory
from qforge_ai.errors import CapabilityError, ConfigurationError
from qforge_ai.ir import (
    CircuitIR,
    Measurement,
    Observable,
    OperationSignature,
    operation_signature,
    register_operation_signature,
)
from qforge_ai.specs import ComponentStatus, Engine, ExperimentSpec, ModelMode, to_primitive

HAS_QISKIT = importlib.util.find_spec("qiskit") is not None
HAS_AER = importlib.util.find_spec("qiskit_aer") is not None
HAS_QML = importlib.util.find_spec("qiskit_machine_learning") is not None
HAS_PENNYLANE = importlib.util.find_spec("pennylane") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None

requires_qiskit = pytest.mark.skipif(not HAS_QISKIT, reason="qiskit not installed")
requires_aer = pytest.mark.skipif(not (HAS_QISKIT and HAS_AER), reason="qiskit-aer not installed")
requires_qml = pytest.mark.skipif(
    not (HAS_QISKIT and HAS_AER and HAS_QML), reason="qiskit-machine-learning not installed"
)
requires_pennylane = pytest.mark.skipif(not HAS_PENNYLANE, reason="pennylane not installed")
requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")


# --------------------------------------------------------------------------
# 0. Operation signatures are validated
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,wires,params",
    [
        ("X", (0, 1), ()),          # extra wire was silently dropped by Qiskit
        ("H", (0, 1), ()),
        ("RX", (0,), ()),           # missing rotation angle
        ("RX", (0,), (0.1, 0.2)),   # extra angle
        ("U", (0,), (0.1, 0.2)),    # U3 needs exactly three
        ("CX", (0,), ()),           # two-qubit gate on one wire
        ("CRY", (0, 1), ()),        # controlled rotation without an angle
        ("RESET", (0, 1), ()),
        ("MEASURE", (0, 1), ()),
    ],
)
def test_operation_arity_is_rejected(name: str, wires: tuple[int, ...], params: tuple) -> None:
    ir = CircuitIR(num_qubits=2)
    with pytest.raises(ConfigurationError):
        ir.add(name, wires, *params)


def test_operation_arity_is_case_insensitive() -> None:
    with pytest.raises(ConfigurationError):
        CircuitIR(num_qubits=2).add("x", (0, 1))


@pytest.mark.parametrize(
    "name,wires,params",
    [
        ("X", (0,), ()),
        ("RX", (0,), (0.5,)),
        ("U", (0,), (0.1, 0.2, 0.3)),
        ("CX", (0, 1), ()),
        ("CRY", (0, 1), (0.5,)),
        ("BARRIER", (0, 1), ()),
        ("STATE_PREP", (0, 1), (1.0, 0.0, 0.0, 0.0)),
    ],
)
def test_valid_operations_still_accepted(name, wires, params) -> None:
    assert CircuitIR(num_qubits=2).add(name, wires, *params).validate()


def test_state_prep_amplitudes_bounded_by_wire_space() -> None:
    with pytest.raises(ConfigurationError, match="at most 2 parameter"):
        CircuitIR(num_qubits=2).add("STATE_PREP", (0,), 1.0, 0.0, 0.0, 0.0)


@requires_qiskit
def test_signature_violation_cannot_reach_the_compiler() -> None:
    """The original failure mode: compile() quietly emitted a 1-qubit X."""

    from qforge_ai.backends import QiskitAdapter

    ir = CircuitIR(num_qubits=2)
    with pytest.raises(ConfigurationError):
        ir.add("X", (0, 1))
    ir.add("X", 0)
    compiled = QiskitAdapter().compile(ir)
    assert [instruction.operation.name for instruction in compiled.data] == ["x"]


def test_plugin_operations_may_declare_a_signature() -> None:
    register_operation_signature(
        "QFORGE_TEST_GATE", OperationSignature(wires=2, params=1), replace=True
    )
    assert operation_signature("qforge_test_gate").wires == 2
    CircuitIR(num_qubits=2).add("QFORGE_TEST_GATE", (0, 1), 0.4)
    with pytest.raises(ConfigurationError):
        CircuitIR(num_qubits=2).add("QFORGE_TEST_GATE", (0,), 0.4)


def test_unregistered_operations_remain_permissive_at_ir_level() -> None:
    # Third-party gates keep working; the translator rejects them instead.
    assert CircuitIR(num_qubits=2).add("SOME_PLUGIN_GATE", (0, 1), 0.1, 0.2).validate()


# --------------------------------------------------------------------------
# 1. Quantum-kernel / QSVC path honours BackendSpec
# --------------------------------------------------------------------------


def _kernel_ir() -> CircuitIR:
    return CircuitFactory().build(
        CircuitSpec(num_qubits=2, encoding=EncodingSpec(name="angle"))
    )


@requires_qml
def test_kernel_uses_statevector_only_when_analytic() -> None:
    from qforge_ai.models.qiskit_qnn import build_quantum_kernel

    _, kind = build_quantum_kernel(
        _kernel_ir(), backend=BackendSpec(engine=Engine.QISKIT, device="statevector")
    )
    assert kind == "fidelity_statevector_kernel"


@requires_qml
@pytest.mark.parametrize(
    "backend",
    [
        BackendSpec(engine=Engine.QISKIT, device="aer", shots=256, seed=7),
        # A noisy kernel is sampler-based too, but the sampler needs an
        # explicit shot count: it no longer substitutes 1024 silently.
        BackendSpec(
            engine=Engine.QISKIT,
            device="aer",
            shots=256,
            noise=NoiseSpec(name="depolarizing_noise", probability=0.05),
        ),
    ],
)
def test_kernel_switches_to_sampler_when_backend_is_not_analytic(backend) -> None:
    from qforge_ai.models.qiskit_qnn import build_quantum_kernel

    _, kind = build_quantum_kernel(_kernel_ir(), backend=backend)
    assert kind.startswith("fidelity_quantum_kernel")


@requires_qml
def test_kernel_values_actually_change_under_noise() -> None:
    """Not just a different class: the noise must reach the kernel entries."""

    from qforge_ai.models.qiskit_qnn import build_quantum_kernel

    ir = _kernel_ir()
    samples = np.array([[0.1, 0.2], [0.9, 0.8], [0.4, 0.5]])
    analytic, _ = build_quantum_kernel(
        ir, backend=BackendSpec(engine=Engine.QISKIT, device="statevector")
    )
    noisy, _ = build_quantum_kernel(
        ir,
        backend=BackendSpec(
            engine=Engine.QISKIT,
            device="aer",
            shots=2048,
            seed=7,
            noise=NoiseSpec(name="depolarizing_noise", probability=0.15),
        ),
    )
    upper = np.triu_indices(len(samples), 1)
    difference = np.abs(analytic.evaluate(samples)[upper] - noisy.evaluate(samples)[upper])
    assert difference.max() > 0.02


@requires_qml
def test_kernel_seed_is_honoured_and_reproducible() -> None:
    from qforge_ai.models.qiskit_qnn import build_quantum_kernel

    ir = _kernel_ir()
    samples = np.array([[0.1, 0.2], [0.9, 0.8], [0.4, 0.5]])

    def evaluate(seed: int) -> np.ndarray:
        kernel, _ = build_quantum_kernel(
            ir, backend=BackendSpec(engine=Engine.QISKIT, device="aer", shots=2048, seed=seed)
        )
        return kernel.evaluate(samples)

    assert np.allclose(evaluate(1), evaluate(1))
    assert not np.allclose(evaluate(1), evaluate(2))


@requires_qml
def test_statevector_kernel_refuses_to_pretend_under_shots() -> None:
    from qforge_ai.models.qiskit_qnn import build_quantum_kernel

    with pytest.raises(ConfigurationError, match="analytic"):
        build_quantum_kernel(
            _kernel_ir(),
            backend=BackendSpec(engine=Engine.QISKIT, device="aer", shots=64),
            fidelity="statevector",
        )


@requires_qml
def test_qsvc_rejects_unknown_options() -> None:
    from qforge_ai.models.qiskit_qnn import build_qsvc

    with pytest.raises(ConfigurationError, match="does not accept"):
        build_qsvc(_kernel_ir(), not_an_svc_parameter=1)


# --------------------------------------------------------------------------
# 2. Seeding happens before the model exists
# --------------------------------------------------------------------------


def _seed_spec(seed: int) -> ExperimentSpec:
    return ExperimentSpec(
        name="seeded",
        circuit=CircuitSpec(num_qubits=2),
        backend=BackendSpec(engine=Engine.PENNYLANE, device="default.qubit"),
        model=ModelSpec(mode=ModelMode.HYBRID, input_dimension=4, output_dimension=2),
        seed=seed,
    )


@requires_pennylane
@requires_torch
def test_same_experiment_seed_gives_identical_initial_weights() -> None:
    import torch

    from qforge_ai.experiment import ExperimentRunner

    runner = ExperimentRunner()
    prepared = runner.prepare(_seed_spec(123))
    first = torch.cat([p.detach().reshape(-1) for p in runner.build_model(prepared).parameters()])
    second = torch.cat([p.detach().reshape(-1) for p in runner.build_model(prepared).parameters()])
    assert torch.equal(first, second)


@requires_pennylane
@requires_torch
def test_different_experiment_seed_gives_different_initial_weights() -> None:
    import torch

    from qforge_ai.experiment import ExperimentRunner

    runner = ExperimentRunner()
    first = torch.cat(
        [
            p.detach().reshape(-1)
            for p in runner.build_model(runner.prepare(_seed_spec(1))).parameters()
        ]
    )
    second = torch.cat(
        [
            p.detach().reshape(-1)
            for p in runner.build_model(runner.prepare(_seed_spec(2))).parameters()
        ]
    )
    assert not torch.equal(first, second)


@requires_torch
def test_deterministic_flag_is_restored_after_training_scope() -> None:
    import torch

    from qforge_ai.runtime import deterministic_algorithms

    before = torch.are_deterministic_algorithms_enabled()
    with deterministic_algorithms(True):
        assert torch.are_deterministic_algorithms_enabled()
    assert torch.are_deterministic_algorithms_enabled() == before

    with pytest.raises(RuntimeError):
        with deterministic_algorithms(True):
            raise RuntimeError("boom")
    assert torch.are_deterministic_algorithms_enabled() == before


# --------------------------------------------------------------------------
# 3. Unknown options and invalid methods are rejected
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"engine": Engine.QISKIT, "device": "aer", "options": {"unknown_key": 1}},
        {"engine": Engine.PENNYLANE, "device": "default.qubit", "options": {"run_options": {}}},
        {"engine": Engine.PENNYLANE, "device": "default.qubit", "method": "not_a_method"},
        {"engine": Engine.QISKIT, "device": "aer", "method": "not_a_method"},
        {"engine": Engine.PENNYLANE, "device": "default.qubit", "compute_device": "GPU"},
        {"engine": Engine.QISKIT, "device": "aer", "options": {"run_options": 5}},
    ],
)
def test_backend_spec_rejects_settings_it_would_have_dropped(kwargs) -> None:
    with pytest.raises(ConfigurationError):
        BackendSpec(**kwargs)


def test_valid_backend_settings_still_accepted() -> None:
    assert BackendSpec(
        engine=Engine.QISKIT,
        device="aer",
        method="density_matrix",
        options={"run_options": {"max_parallel_threads": 2}},
    )
    assert BackendSpec(engine=Engine.PENNYLANE, device="lightning.qubit")


def test_model_options_that_a_template_ignores_are_rejected() -> None:
    from qforge_ai.models import ModelFactory

    ir = CircuitFactory().build(CircuitSpec(num_qubits=2))
    with pytest.raises(ConfigurationError, match="does not use"):
        ModelFactory().build(
            ir,
            ModelSpec(mode=ModelMode.HYBRID, template="quantum_residual", options={"branches": 4}),
            BackendSpec(engine=Engine.PENNYLANE, device="default.qubit"),
        )


@requires_qml
def test_gradient_options_are_rejected_when_the_method_ignores_them() -> None:
    from qforge_ai.models.qiskit_qnn import build_estimator_qnn

    ir = CircuitFactory().build(
        CircuitSpec(num_qubits=2, measurement=MeasurementSpec(name="expectation"))
    )
    with pytest.raises(ConfigurationError, match="gradient_options"):
        build_estimator_qnn(
            ir,
            backend=BackendSpec(
                engine=Engine.QISKIT,
                device="statevector",
                options={"gradient_options": {"epsilon": 1e-3}},
            ),
        )


# --------------------------------------------------------------------------
# 4. Noise covers every translatable gate
# --------------------------------------------------------------------------


@requires_aer
@pytest.mark.parametrize(
    "noise",
    [
        NoiseSpec(name="depolarizing_noise", probability=0.05),
        NoiseSpec(name="bit_flip_noise", probability=0.05),
        NoiseSpec(name="phase_flip_noise", probability=0.05),
        NoiseSpec(name="amplitude_damping", probability=0.05),
        NoiseSpec(name="phase_damping", probability=0.05),
        NoiseSpec(name="thermal_relaxation", t1=50e3, t2=30e3, gate_time=100.0),
    ],
)
def test_noise_model_covers_every_translatable_gate(noise: NoiseSpec) -> None:
    from qforge_ai.backends.gates import NOISY_ONE_QUBIT_BASIS, NOISY_TWO_QUBIT_BASIS
    from qforge_ai.noise import build_qiskit_noise_model

    model = build_qiskit_noise_model(noise)
    expected = set(NOISY_ONE_QUBIT_BASIS) | set(NOISY_TWO_QUBIT_BASIS)
    assert expected <= set(model.noise_instructions)


def test_noise_basis_is_derived_from_the_translator_tables() -> None:
    """Guards against the two lists drifting apart again."""

    from qforge_ai.backends.gates import (
        NOISY_ONE_QUBIT_BASIS,
        NOISY_TWO_QUBIT_BASIS,
        QISKIT_SINGLE_QUBIT_GATES,
        QISKIT_TWO_QUBIT_GATES,
    )

    assert set(NOISY_ONE_QUBIT_BASIS) == set(QISKIT_SINGLE_QUBIT_GATES.values())
    assert set(NOISY_TWO_QUBIT_BASIS) == set(QISKIT_TWO_QUBIT_GATES.values())
    # The gate QCNN pooling depends on must be covered.
    assert "cry" in NOISY_TWO_QUBIT_BASIS


@requires_aer
def test_qcnn_controlled_rotation_is_actually_noisy() -> None:
    from qforge_ai.backends import QiskitAdapter

    ir = CircuitFactory().build(
        CircuitSpec(
            num_qubits=4,
            encoding=EncodingSpec(name="angle"),
            ansatz=AnsatzSpec(name="qcnn", layers=2),
            measurement=MeasurementSpec(name="density_matrix"),
        )
    )
    assert ir.gate_counts().get("CRY", 0) > 0
    weights = np.zeros(ir.parameter_groups().get("weight", 0))
    inputs = np.full(ir.parameter_groups().get("input", 0), 0.3)
    clean = QiskitAdapter(backend_name="aer").run(ir, inputs=inputs, weights=weights)
    noisy = QiskitAdapter(
        backend_name="aer", noise=NoiseSpec(name="depolarizing_noise", probability=0.1)
    ).run(ir, inputs=inputs, weights=weights)
    assert not np.allclose(np.asarray(clean.values), np.asarray(noisy.values))


# --------------------------------------------------------------------------
# 5. Data re-uploading no longer collapses
# --------------------------------------------------------------------------


def _reuploading_ir(repeats: int = 2, feature_dimension: int = 4) -> CircuitIR:
    return CircuitFactory().build(
        CircuitSpec(
            num_qubits=2,
            encoding=EncodingSpec(
                name="data_reuploading", feature_dimension=feature_dimension, repeats=repeats
            ),
            ansatz=AnsatzSpec(
                name="hardware_efficient",
                layers=repeats,
                entanglement="none",
                rotation_blocks=("RY",),
            ),
            measurement=MeasurementSpec(name="statevector"),
        )
    )


@requires_qiskit
def test_reuploading_distinguishes_vectors_with_equal_per_wire_sums() -> None:
    """The exact collapse from the report: identical statevectors."""

    from qforge_ai.backends import QiskitAdapter

    ir = _reuploading_ir()
    adapter = QiskitAdapter()
    weights = np.zeros(ir.parameter_groups().get("weight", 0))
    left = np.asarray(adapter.run(ir, inputs=[0.3, 0.4, 0.7, 0.6], weights=weights).values)
    right = np.asarray(adapter.run(ir, inputs=[0.5, 0.2, 0.5, 0.8], weights=weights).values)
    assert np.max(np.abs(left - right)) > 1e-6


@requires_qiskit
def test_reuploading_is_injective_over_a_random_sweep() -> None:
    from qforge_ai.backends import QiskitAdapter

    ir = _reuploading_ir()
    adapter = QiskitAdapter()
    weights = np.zeros(ir.parameter_groups().get("weight", 0))
    generator = np.random.default_rng(0)
    states = np.stack(
        [
            np.asarray(adapter.run(ir, inputs=list(generator.uniform(-1, 1, 4)), weights=weights).values)
            for _ in range(60)
        ]
    )
    distances = np.abs(states[:, None, :] - states[None, :, :]).max(axis=-1)
    np.fill_diagonal(distances, np.inf)
    assert distances.min() > 1e-6


@requires_qiskit
def test_angle_encoding_with_repeats_does_not_collapse() -> None:
    from qforge_ai.backends import QiskitAdapter

    ir = CircuitFactory().build(
        CircuitSpec(
            num_qubits=2,
            encoding=EncodingSpec(name="angle", feature_dimension=4, repeats=2),
            ansatz=AnsatzSpec(name="hardware_efficient", layers=1, entanglement="none"),
            measurement=MeasurementSpec(name="statevector"),
        )
    )
    adapter = QiskitAdapter()
    weights = np.zeros(ir.parameter_groups().get("weight", 0))
    left = np.asarray(adapter.run(ir, inputs=[0.3, 0.4, 0.7, 0.6], weights=weights).values)
    right = np.asarray(adapter.run(ir, inputs=[0.5, 0.2, 0.5, 0.8], weights=weights).values)
    assert np.max(np.abs(left - right)) > 1e-6


def test_single_load_encoding_keeps_the_declared_rotation_axis() -> None:
    """repeats=1 must be untouched by the axis-alternation fix."""

    ir = CircuitFactory().build(
        CircuitSpec(num_qubits=3, encoding=EncodingSpec(name="angle", rotation="Y"))
    )
    encoding_gates = [
        operation.name
        for operation in ir.operations
        if operation.metadata.get("role") == "encoding"
    ]
    assert encoding_gates == ["RY", "RY", "RY"]


def test_reuploading_alternates_axes_per_wire() -> None:
    ir = _reuploading_ir(repeats=2, feature_dimension=4)
    per_wire: dict[int, list[str]] = {}
    for operation in ir.operations:
        if operation.metadata.get("role") == "encoding":
            per_wire.setdefault(operation.wires[0], []).append(operation.name)
    for gates in per_wire.values():
        assert all(left != right for left, right in zip(gates, gates[1:], strict=False))


# --------------------------------------------------------------------------
# 6. Catalog claims match the implementation
# --------------------------------------------------------------------------


def test_every_catalog_entanglement_name_is_accepted_by_ansatz_spec() -> None:
    for descriptor in catalog.list(category="entanglement"):
        assert AnsatzSpec(entanglement=descriptor.key)


def test_canonical_entanglement_names_still_work() -> None:
    assert AnsatzSpec(entanglement="linear").entanglement == "linear"
    assert AnsatzSpec(entanglement="linear_entanglement").entanglement == "linear"
    with pytest.raises(ConfigurationError):
        AnsatzSpec(entanglement="not_a_topology")


def test_every_composite_catalog_entry_has_a_model_factory_route() -> None:
    from qforge_ai.models import ModelFactory

    unrouted = [
        descriptor.key
        for descriptor in catalog.list()
        if descriptor.status == ComponentStatus.COMPOSITE
        and ModelFactory.resolve_template(descriptor.key) is None
    ]
    assert unrouted == []


@pytest.mark.parametrize(
    "observable",
    ["pauli_x", "pauli_y", "pauli_z", "identity", "hadamard_observable"],
)
@requires_qiskit
@requires_pennylane
def test_catalog_observable_names_run_on_both_backends(observable: str) -> None:
    from qforge_ai.backends import PennyLaneAdapter, QiskitAdapter

    ir = CircuitFactory().build(
        CircuitSpec(
            num_qubits=2,
            measurement=MeasurementSpec(name="expectation", observable=observable),
        )
    )
    weights = np.zeros(ir.parameter_groups().get("weight", 0))
    inputs = np.zeros(ir.parameter_groups().get("input", 0))
    for adapter in (
        QiskitAdapter(),
        QiskitAdapter(backend_name="statevector"),
        PennyLaneAdapter(),
    ):
        assert adapter.run(ir, inputs=inputs, weights=weights) is not None


def test_unsupported_observable_is_rejected_with_guidance() -> None:
    with pytest.raises(ConfigurationError, match="hermitian"):
        MeasurementSpec(name="expectation", observable="pauli_hamiltonian")


def test_hermitian_observable_requires_a_matrix() -> None:
    with pytest.raises(ConfigurationError, match="observable_matrix"):
        MeasurementSpec(name="expectation", observable="hermitian_observable")


def test_every_entry_claimed_runnable_is_actually_executable() -> None:
    """The catalog must not count capacity the framework does not have."""

    claimed = [
        descriptor.key
        for descriptor in catalog.list()
        if descriptor.metadata.get("runtime_contract") == "runnable"
    ]
    assert claimed, "expected some runnable entries"
    assert [key for key in claimed if not is_executable(key)] == []


def test_capability_report_is_self_consistent() -> None:
    report = capability_report()
    assert report["total"] == len(catalog.list())
    assert report["executable"] + report["discovery_only"] == report["total"]
    assert report["executable"] == sum(
        bucket["executable"] for bucket in report["by_category"].values()
    )
    # Executable entries are exactly the native + composite ones.
    assert report["executable"] == (
        report["by_status"]["native"] + report["by_status"]["composite"]
    )


# --------------------------------------------------------------------------
# 7. Multiple terminal measurements are rejected, not dropped
# --------------------------------------------------------------------------


def test_second_terminal_measurement_is_rejected() -> None:
    ir = CircuitIR(num_qubits=2)
    ir.add("H", 0)
    ir.measure(Measurement("expectation", (0,), (Observable("Z", (0,)),)))
    with pytest.raises(ConfigurationError, match="single terminal measurement"):
        ir.measure(Measurement("probability", (0, 1)))


def test_rejected_measurement_does_not_corrupt_the_circuit() -> None:
    ir = CircuitIR(num_qubits=2)
    ir.measure(Measurement("expectation", (0,), (Observable("Z", (0,)),)))
    with pytest.raises(ConfigurationError):
        ir.measure(Measurement("probability", (0, 1)))
    assert len(ir.measurements) == 1
    assert ir.validate()


def test_multi_output_is_expressed_with_multiple_observables() -> None:
    ir = CircuitIR(num_qubits=2)
    ir.measure(
        Measurement(
            "expectation", (0, 1), (Observable("Z", (0,)), Observable("Z", (1,)))
        )
    )
    assert len(ir.measurements[0].observables) == 2


def test_validate_catches_measurements_appended_directly_to_the_list() -> None:
    ir = CircuitIR(num_qubits=2)
    ir.measurements.append(Measurement("expectation", (0,), (Observable("Z", (0,)),)))
    ir.measurements.append(Measurement("probability", (0, 1)))
    with pytest.raises(ConfigurationError, match="single terminal measurement"):
        ir.validate()


# --------------------------------------------------------------------------
# 8. Estimator observable override keeps the head consistent
# --------------------------------------------------------------------------


@requires_qml
@requires_torch
@pytest.mark.parametrize("count", [1, 2, 3])
def test_observable_override_sizes_the_classical_head(count: int) -> None:
    import torch
    from qiskit.quantum_info import SparsePauliOp

    from qforge_ai.models import ModelFactory

    ir = CircuitFactory().build(
        CircuitSpec(num_qubits=2, measurement=MeasurementSpec(name="expectation"))
    )
    labels = ["ZI", "IZ", "ZZ"][:count]
    override = SparsePauliOp(labels[0]) if count == 1 else [SparsePauliOp(x) for x in labels]
    model = ModelFactory().build(
        ir,
        ModelSpec(
            mode=ModelMode.HYBRID,
            input_dimension=2,
            output_dimension=2,
            options={"observable": override},
        ),
        BackendSpec(engine=Engine.QISKIT, device="statevector"),
    )
    assert model(torch.zeros(1, 2)).shape == (1, 2)


@requires_qml
def test_observable_override_width_is_validated_early() -> None:
    from qiskit.quantum_info import SparsePauliOp

    from qforge_ai.models.qiskit_qnn import build_estimator_qnn

    ir = CircuitFactory().build(
        CircuitSpec(num_qubits=2, measurement=MeasurementSpec(name="expectation"))
    )
    with pytest.raises(ConfigurationError, match="qubit"):
        build_estimator_qnn(ir, observable=SparsePauliOp("Z"))


def test_observable_output_count_treats_multi_term_operators_as_one() -> None:
    from qforge_ai.models.qiskit_qnn import observable_output_count

    pytest.importorskip("qiskit")
    from qiskit.quantum_info import SparsePauliOp

    assert observable_output_count(None) == 0
    assert observable_output_count(SparsePauliOp(["ZI", "IZ"])) == 1
    assert observable_output_count([SparsePauliOp("ZI"), SparsePauliOp("IZ")]) == 2


# --------------------------------------------------------------------------
# 9. PennyLane device/GPU capability is probed, not asserted
# --------------------------------------------------------------------------


@requires_pennylane
def test_pennylane_capabilities_only_list_installed_devices() -> None:
    from qforge_ai.runtime import pennylane_registered_devices

    capability = CapabilityRegistry().get(Engine.PENNYLANE)
    registered = pennylane_registered_devices()
    for device in capability.devices:
        assert device == "statevector" or device in registered


@requires_pennylane
def test_pennylane_gpu_claim_matches_reality() -> None:
    from qforge_ai.runtime import pennylane_device_constructible

    capability = CapabilityRegistry().get(Engine.PENNYLANE)
    assert capability.gpu == any(
        pennylane_device_constructible(name)
        for name in ("lightning.gpu", "lightning.kokkos")
    )


@requires_pennylane
def test_uninstallable_pennylane_device_fails_preflight_not_at_runtime() -> None:
    from qforge_ai.runtime import pennylane_device_constructible

    if pennylane_device_constructible("lightning.gpu"):
        pytest.skip("lightning.gpu genuinely available on this machine")
    ir = CircuitFactory().build(CircuitSpec(num_qubits=2))
    with pytest.raises(CapabilityError):
        CapabilityRegistry().validate(
            ir, BackendSpec(engine=Engine.PENNYLANE, device="lightning.gpu")
        )


@requires_aer
def test_qiskit_method_is_checked_against_the_installed_build() -> None:
    from qforge_ai.runtime import qiskit_aer_methods

    installed = qiskit_aer_methods()
    missing = sorted({"tensor_network", "superop", "unitary"} - installed)
    if not missing:
        pytest.skip("installed Aer build offers every probed method")
    ir = CircuitFactory().build(CircuitSpec(num_qubits=2))
    with pytest.raises(CapabilityError, match="method"):
        CapabilityRegistry().validate(
            ir, BackendSpec(engine=Engine.QISKIT, device="aer", method=missing[0])
        )


# --------------------------------------------------------------------------
# 10. Serialization contract is narrow and enforced
# --------------------------------------------------------------------------


def test_metadata_with_path_and_array_round_trips_through_json() -> None:
    spec = ExperimentSpec(
        name="serialization",
        circuit=CircuitSpec(num_qubits=2),
        metadata={"path": Path("/tmp/data.npz"), "array": np.array([1.0, 2.0])},
    )
    payload = spec.to_dict()
    assert json.loads(json.dumps(payload))["metadata"] == {
        "path": "/tmp/data.npz",
        "array": [1.0, 2.0],
    }


def test_spec_with_path_and_array_metadata_is_hashable() -> None:
    spec = ExperimentSpec(
        name="hashable",
        circuit=CircuitSpec(num_qubits=2),
        metadata={"path": Path("/tmp/data.npz"), "array": np.array([1.0, 2.0])},
    )
    assert isinstance(hash(spec), int)


def test_metadata_rejects_objects_that_are_not_data() -> None:
    class Opaque:
        pass

    with pytest.raises(ConfigurationError, match="JSON-compatible"):
        ExperimentSpec(
            name="bad", circuit=CircuitSpec(num_qubits=2), metadata={"engine": Opaque()}
        )


def test_yaml_round_trip_preserves_normalized_metadata(tmp_path) -> None:
    spec = ExperimentSpec(
        name="yaml",
        circuit=CircuitSpec(num_qubits=2),
        metadata={"path": Path("/tmp/x.npz"), "array": np.array([[1.0, 2.0], [3.0, 4.0]])},
    )
    target = tmp_path / "spec.yaml"
    spec.to_yaml(target)
    assert ExperimentSpec.from_yaml(target).metadata == spec.metadata


@requires_qml
def test_strict_serialization_names_the_offending_field() -> None:
    from qiskit.quantum_info import SparsePauliOp

    spec = ExperimentSpec(
        name="strict",
        circuit=CircuitSpec(num_qubits=2),
        model=ModelSpec(options={"observable": SparsePauliOp("ZI")}),
    )
    with pytest.raises(ConfigurationError, match="model.options.observable"):
        spec.to_dict()
    lenient = to_primitive(spec, strict=False)
    assert lenient["model"]["options"]["observable"] == "<SparsePauliOp>"
    json.dumps(lenient)


@requires_qml
def test_tracking_manifest_survives_unserializable_options() -> None:
    from qiskit.quantum_info import SparsePauliOp

    from qforge_ai.tracking import build_manifest

    spec = ExperimentSpec(
        name="manifest",
        circuit=CircuitSpec(num_qubits=2),
        model=ModelSpec(options={"observable": SparsePauliOp("ZI")}),
    )
    manifest = build_manifest(spec, CircuitFactory().build(spec.circuit))
    json.dumps(manifest.experiment)


# ===========================================================================
# Defects reported against 1.2.2
#
# Each test below fails on 1.2.2 and passes on 1.2.3.
# ===========================================================================


# --------------------------------------------------------------- finding 4
@requires_qml
def test_kernel_fidelity_option_no_longer_raises_a_bare_type_error() -> None:
    """``ModelSpec.options={'fidelity': ...}`` collided with the factory kwarg.

    The user saw ``TypeError: got multiple values for keyword argument
    'fidelity'`` with nothing pointing at their own configuration.
    """

    from qforge_ai.models import ModelFactory

    ir = CircuitFactory().build(
        CircuitSpec(
            num_qubits=2,
            encoding=EncodingSpec("angle", 2),
            ansatz=AnsatzSpec("real_amplitudes", layers=1),
            measurement=MeasurementSpec("expectation"),
        )
    )
    model = ModelFactory().build(
        ir,
        ModelSpec(
            mode=ModelMode.QUANTUM_KERNEL,
            template="quantum_kernel_model",
            options={"fidelity": "sampler"},
        ),
        BackendSpec(engine=Engine.QISKIT, device="aer", shots=256),
    )
    assert model.qforge_kernel_kind.startswith("fidelity_quantum_kernel")

    with pytest.raises(ConfigurationError, match="fidelity"):
        ModelFactory().build(
            ir,
            ModelSpec(
                mode=ModelMode.QUANTUM_KERNEL,
                template="quantum_kernel_model",
                options={"fidelity": "nonsense"},
            ),
            BackendSpec(engine=Engine.QISKIT, device="aer", shots=256),
        )


@requires_qml
@pytest.mark.parametrize(
    "backend",
    [
        BackendSpec(engine=Engine.QISKIT, device="aer", method="matrix_product_state"),
        BackendSpec(
            engine=Engine.QISKIT,
            device="aer",
            options={"backend_options": {"max_parallel_threads": 1}},
        ),
        BackendSpec(
            engine=Engine.QISKIT,
            device="aer",
            options={"run_options": {"memory": True}},
        ),
    ],
)
def test_analytic_kernel_refuses_settings_it_cannot_honour(backend) -> None:
    """The statevector kernel silently dropped every execution setting.

    ``fidelity='statevector'`` is a single analytic CPU simulation; a spec
    asking for MPS, custom backend options or run options used to be accepted
    and produce a plain statevector run that reported the request back.
    """

    from qforge_ai.models.qiskit_qnn import build_quantum_kernel

    ir = _kernel_ir()
    with pytest.raises(ConfigurationError, match="cannot apply"):
        build_quantum_kernel(ir, backend=backend, fidelity="statevector")
    # 'auto' must route such a backend to the sampler instead of pretending.
    with pytest.raises(ConfigurationError, match="shots"):
        build_quantum_kernel(ir, backend=backend, fidelity="auto")


@requires_qml
def test_kernel_rejects_a_gradient_contract_it_never_uses() -> None:
    from qforge_ai.models.qiskit_qnn import build_quantum_kernel
    from qforge_ai.specs import GradientMethod

    with pytest.raises(ConfigurationError, match="gradient"):
        build_quantum_kernel(
            _kernel_ir(),
            backend=BackendSpec(
                engine=Engine.QISKIT,
                device="statevector",
                gradient=GradientMethod.SPSA,
            ),
        )


# --------------------------------------------------------------- finding 5
@pytest.mark.skipif(not HAS_PENNYLANE, reason="pennylane required")
@pytest.mark.parametrize(
    "options",
    [
        {"epsilon": -1.0},
        {"epsilon": 0.0},
        {"epsilon": float("inf")},
        {"bogus_key": 3},
        {"method": "sideways"},
        {"batch_size": 0},
    ],
)
def test_pennylane_gradient_options_are_no_longer_ignored(options) -> None:
    """PennyLane accepted gradient_options and dropped them on the floor.

    A model could be built with a negative epsilon and an unknown key and ran
    happily with PennyLane's defaults, reporting a step size it never used.
    """

    from qforge_ai.backends import PennyLaneAdapter

    adapter = PennyLaneAdapter(gradient_options=options)
    ir = CircuitFactory().build(
        CircuitSpec(
            num_qubits=1,
            encoding=EncodingSpec("angle", 1),
            ansatz=AnsatzSpec("real_amplitudes", layers=1),
            measurement=MeasurementSpec("expectation", wires=(0,)),
        )
    )
    with pytest.raises(ConfigurationError):
        adapter.compile(ir, diff_method="finite-diff")


@pytest.mark.skipif(not HAS_PENNYLANE, reason="pennylane required")
def test_pennylane_gradient_options_actually_reach_the_qnode() -> None:
    from qforge_ai.backends.gradients import normalize_pennylane_gradient_options

    resolved = normalize_pennylane_gradient_options(
        {"epsilon": 0.01, "method": "central"}, diff_method="finite-diff"
    )
    # The framework's canonical keys map onto PennyLane's own spelling.
    assert resolved == {"h": 0.01, "strategy": "center"}
    assert normalize_pennylane_gradient_options(
        {"epsilon": 0.02, "batch_size": 4}, diff_method="spsa"
    ) == {"h": 0.02, "num_directions": 4}


@pytest.mark.skipif(not HAS_PENNYLANE, reason="pennylane required")
def test_parameter_shift_rejects_options_it_cannot_use_on_both_engines() -> None:
    """Qiskit already rejected these; PennyLane accepted and ignored them."""

    from qforge_ai.backends.gradients import normalize_pennylane_gradient_options

    with pytest.raises(ConfigurationError, match="gradient_options"):
        normalize_pennylane_gradient_options(
            {"epsilon": 1e-4}, diff_method="parameter-shift"
        )


# --------------------------------------------------------------- finding 6
@requires_qml
@pytest.mark.skipif(not HAS_PENNYLANE, reason="pennylane required")
def test_quantum_kernel_model_really_runs_on_pennylane() -> None:
    """The catalog advertised PennyLane support the factory did not have."""

    pytest.importorskip("sklearn")
    from qforge_ai.models import ModelFactory

    ir = _kernel_ir()
    model = ModelFactory().build(
        ir,
        ModelSpec(mode=ModelMode.QUANTUM_KERNEL, template="quantum_kernel_model"),
        BackendSpec(engine=Engine.PENNYLANE, device="default.qubit"),
    )
    samples = np.array([[0.1, 0.2], [1.3, 1.4], [0.15, 0.25], [1.35, 1.45]])
    labels = np.array([0, 1, 0, 1])
    model.fit(samples, labels)
    assert np.asarray(model.predict(samples)).shape == (4,)
    gram = model.qforge_quantum_kernel(samples)
    assert np.allclose(np.diag(gram), 1.0, atol=1e-8)
    assert np.allclose(gram, gram.T, atol=1e-10)


def test_catalog_executability_is_verified_by_execution_not_by_routing() -> None:
    """Executable claims are now backed by a build-and-run sweep."""

    from qforge_ai.verification import verify_component

    if not (HAS_QISKIT and HAS_PENNYLANE):
        pytest.skip("both engines required")
    for engine in (Engine.QISKIT, Engine.PENNYLANE):
        assert verify_component("angle_encoding", engine=engine).ok
        assert verify_component("thermal_relaxation", engine=engine).ok
    # A discovery-only entry is reported as skipped, never as verified.
    assert verify_component("surface_code", engine=Engine.QISKIT).status == "skipped"


# --------------------------------------------------------------- finding 7
@pytest.mark.skipif(not HAS_TORCH, reason="torch required")
@requires_qml
def test_quantum_only_output_dimension_is_enforced_not_ignored() -> None:
    """Asking a quantum-only model for 7 outputs produced 2 and said nothing."""

    from qforge_ai.models import ModelFactory

    ir = CircuitFactory().build(
        CircuitSpec(
            num_qubits=2,
            encoding=EncodingSpec("angle", 2),
            ansatz=AnsatzSpec("real_amplitudes", layers=1),
            measurement=MeasurementSpec("expectation"),
        )
    )
    with pytest.raises(ConfigurationError, match="output_dimension"):
        ModelFactory().build(
            ir,
            ModelSpec(
                mode=ModelMode.QUANTUM_ONLY,
                template="quantum_classifier",
                output_dimension=7,
            ),
            BackendSpec(engine=Engine.QISKIT, device="statevector"),
        )
    # The width the circuit really produces is accepted.
    model = ModelFactory().build(
        ir,
        ModelSpec(
            mode=ModelMode.QUANTUM_ONLY,
            template="quantum_classifier",
            output_dimension=2,
        ),
        BackendSpec(engine=Engine.QISKIT, device="statevector"),
    )
    import torch

    assert model(torch.zeros(1, 2)).shape == (1, 2)


@requires_qml
def test_sampler_qnn_does_not_silently_substitute_1024_shots() -> None:
    """Qiskit fell back to 1024 shots while PennyLane ran analytically."""

    from qforge_ai.models.qiskit_qnn import build_sampler_qnn

    ir = CircuitFactory().build(
        CircuitSpec(
            num_qubits=2,
            encoding=EncodingSpec("angle", 2),
            ansatz=AnsatzSpec("real_amplitudes", layers=1),
            measurement=MeasurementSpec("probability"),
        )
    )
    with pytest.raises(ConfigurationError, match="shot"):
        build_sampler_qnn(ir, backend=BackendSpec(engine=Engine.QISKIT))
    qnn = build_sampler_qnn(ir, backend=BackendSpec(engine=Engine.QISKIT, shots=512))
    assert qnn.qforge_shots == 512


# --------------------------------------------------------------- finding 8
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_metadata_is_rejected_instead_of_producing_invalid_json(value) -> None:
    with pytest.raises(ConfigurationError, match="finite"):
        ExperimentSpec(
            name="nonfinite",
            circuit=CircuitSpec(num_qubits=1),
            metadata={"bad": value},
        )


def test_complex_hermitian_matrix_survives_a_yaml_round_trip(tmp_path: Path) -> None:
    """``to_primitive`` wrote complex as real/imag and nothing read it back.

    Any Hermitian observable with an imaginary part could be written to YAML
    and never loaded again; ``from_yaml`` failed with "matrix must contain
    finite numbers", pointing at the wrong problem.
    """

    matrix = ((1 + 0j, 0 + 1j), (0 - 1j, 1 + 0j))
    spec = ExperimentSpec(
        name="hermitian",
        circuit=CircuitSpec(
            num_qubits=1,
            measurement=MeasurementSpec(
                "expectation", observable="hermitian", observable_matrix=matrix
            ),
        ),
    )
    path = tmp_path / "experiment.yaml"
    spec.to_yaml(path)
    restored = ExperimentSpec.from_yaml(path)
    assert restored.circuit.measurement.observable_matrix == matrix
    assert restored == spec
    json.dumps(restored.to_dict())


def test_metadata_keys_that_collide_after_stringification_are_rejected() -> None:
    """``1`` and ``"1"`` both became ``"1"`` and one value vanished."""

    with pytest.raises(ConfigurationError, match="key"):
        ExperimentSpec(
            name="collide",
            circuit=CircuitSpec(num_qubits=1),
            metadata={1: "int", "1": "str"},
        )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: CircuitSpec(num_qubits=True),
        lambda: ModelSpec(output_dimension=True),
        lambda: ModelSpec(input_dimension=True),
        lambda: BackendSpec(engine=Engine.QISKIT, device="aer", shots=True),
        lambda: MeasurementSpec("counts", shots=True),
        lambda: Measurement(kind="counts", wires=(0,), shots=True),
        lambda: EncodingSpec("angle", feature_dimension=True),
        lambda: EncodingSpec("angle", repeats=True),
        lambda: AnsatzSpec("real_amplitudes", layers=True),
        lambda: CircuitIR(num_qubits=True),
    ],
)
def test_boolean_is_not_a_valid_integer_field(factory) -> None:
    """``bool`` subclasses ``int``: ``epochs=True`` trained for one epoch."""

    with pytest.raises(ConfigurationError):
        factory()


def test_training_spec_integer_fields_reject_booleans() -> None:
    from qforge_ai.specs import TrainingSpec

    for kwargs in ({"epochs": True}, {"batch_size": True}, {"early_stopping_patience": True}):
        with pytest.raises(ConfigurationError):
            TrainingSpec(**kwargs)


# --------------------------------------------------------------- finding 9
def test_single_clbit_condition_rejects_values_it_can_never_match() -> None:
    """``value=2`` on one classical bit produced a permanently dead gate."""

    ir = CircuitIR(num_qubits=1, name="conditional")
    ir.measure_wire(0, 0)
    with pytest.raises(ConfigurationError, match="0 or 1"):
        ir.add_conditional("X", 0, clbit=0, value=2)
    # 0 and 1 stay valid.
    ir.add_conditional("X", 0, clbit=0, value=1)
    ir.validate()


def test_measure_without_a_target_clbit_is_rejected_by_the_ir() -> None:
    """It used to pass validate() and fail as ``KeyError: 'clbit'`` in Qiskit."""

    from qforge_ai.ir import Operation

    with pytest.raises(ConfigurationError, match="clbit"):
        Operation("MEASURE", (0,))

    ir = CircuitIR(num_qubits=1, name="measure")
    ir.num_clbits = 1
    ir.measure_wire(0, 0)
    ir.validate()
    # A clbit outside the declared register is caught too.
    ir.num_clbits = 0
    with pytest.raises(ConfigurationError, match="clbit"):
        ir.validate()


def test_named_observable_cannot_claim_wires_it_does_not_act_on() -> None:
    """PennyLane read ``wires[0]`` and dropped the rest; Qiskit raised."""

    with pytest.raises(ConfigurationError, match="a single wire"):
        Observable("Z", (0, 1))
    # A matrix observable is exactly how a multi-wire observable is expressed.
    Observable("HERMITIAN", (0, 1), matrix=tuple(
        tuple(complex(1.0 if r == c else 0.0) for c in range(4)) for r in range(4)
    ))


# -------------------------------------------------------------- finding 11
def test_coverage_targets_in_ci_are_paths_not_dotted_modules() -> None:
    """Dotted ``--cov`` targets broke test collection outright.

    coverage.py resolves a dotted target by importing it, which re-executed
    NumPy's C extension under the tracer: every test module then failed to
    collect with "cannot load module more than once per process", so the
    coverage gate could never run.
    """

    workflow = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"
    if not workflow.exists():  # pragma: no cover - sdist without CI config
        pytest.skip("CI workflow not packaged")
    text = workflow.read_text(encoding="utf-8")
    targets = [
        line.split("--cov=", 1)[1].strip().rstrip("\\").strip()
        for line in text.splitlines()
        if "--cov=" in line
    ]
    assert targets, "no coverage targets found in CI"
    for target in targets:
        if target.startswith("-") or "=" in target:
            continue
        assert "/" in target, f"coverage target {target!r} must be a path, not a module"
        assert not target.startswith("qforge_ai."), target


# -------------------------------------------------------------- finding 12
@pytest.mark.skipif(not HAS_PENNYLANE, reason="pennylane required")
def test_provenance_reports_the_device_that_actually_ran() -> None:
    """A run on default.mixed used to claim it came from default.qubit."""

    from qforge_ai.backends import PennyLaneAdapter

    ir = CircuitIR(num_qubits=1, name="provenance")
    ir.add("X", 0)
    ir.measure(
        Measurement(kind="expectation", wires=(0,), observables=(Observable("Z", (0,)),))
    )
    noisy = PennyLaneAdapter(
        device="default.qubit", noise=NoiseSpec(name="bit_flip_noise", probability=0.2)
    ).run(ir)
    assert noisy.metadata["device"] == "default.mixed"
    assert noisy.metadata["requested_device"] == "default.qubit"
    assert noisy.metadata["noise_channel"] == "bit_flip"

    clean = PennyLaneAdapter(device="default.qubit").run(ir)
    assert clean.metadata["device"] == "default.qubit"
    assert clean.metadata["requested_device"] == "default.qubit"


@requires_qiskit
def test_qiskit_provenance_reports_the_simulation_method_that_actually_ran() -> None:
    """A noisy run is forced onto density_matrix; the record said 'automatic'."""

    from qforge_ai.backends import QiskitAdapter

    ir = CircuitIR(num_qubits=1, name="provenance")
    ir.add("X", 0)
    ir.measure(
        Measurement(kind="expectation", wires=(0,), observables=(Observable("Z", (0,)),))
    )
    result = QiskitAdapter(
        noise=NoiseSpec(name="bit_flip_noise", probability=0.2)
    ).run(ir)
    assert result.metadata["method"] == "automatic"
    assert result.metadata["simulation_method"] == "density_matrix"
    assert result.metadata["noise_placement"] == "circuit"
