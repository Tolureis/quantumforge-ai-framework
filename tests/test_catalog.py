from qforge_ai.catalog import catalog
from qforge_ai.errors import DuplicateComponentError, DuplicateFactoryError, RegistryError
from qforge_ai.registry import ComponentDescriptor, ComponentRegistry
from qforge_ai.specs import ComponentStatus, Engine


def test_catalog_is_comprehensive() -> None:
    assert len(catalog) >= 380
    assert len(catalog.categories()) >= 20


def test_aliases_resolve() -> None:
    assert catalog.get("SamplerQNN").key == "sampler_qnn"
    assert catalog.get("vqc").key == "variational_quantum_classifier"
    assert catalog.get("angle").key == "angle_encoding"
    assert catalog.get("hea").key == "hardware_efficient_ansatz"


def test_filtering() -> None:
    native_qiskit = catalog.list(engine=Engine.QISKIT, status=ComponentStatus.NATIVE)
    assert native_qiskit
    assert all(Engine.QISKIT in item.engines for item in native_qiskit)


def test_catalog_metadata_and_status_contract_are_complete() -> None:
    from qforge_ai.circuits import CircuitFactory

    CircuitFactory()
    assert all(item.dependencies for item in catalog)
    assert all(item.references for item in catalog)
    assert all(item.metadata for item in catalog)
    assert all(catalog.is_runnable(item.key) for item in catalog.list(status="native"))


def test_kernel_and_cv_engine_claims_are_honest() -> None:
    """Every engine a kernel entry claims must have a real route behind it.

    The catalog used to advertise ``quantum_kernel_model`` on PennyLane while
    ``ModelFactory`` accepted Qiskit only.  PennyLane now has a genuine
    compute-uncompute fidelity kernel, so the routed kernel entries are
    dual-engine and the un-routed kernel families stay Qiskit-shaped.
    """

    from qforge_ai.models.factory import ModelFactory

    for key in ("fidelity_quantum_kernel", "statevector_quantum_kernel", "qsvc"):
        descriptor = catalog.get(key)
        assert {Engine.QISKIT, Engine.PENNYLANE} <= descriptor.engines
        assert ModelFactory.resolve_template(key) == "quantum_kernel_model"
    # A kernel family with no core route must not claim PennyLane support.
    assert Engine.PENNYLANE not in catalog.get("quantum_rbf_kernel").engines
    # Continuous-variable components remain descriptor-only on every engine.
    assert catalog.get("cv_quantum_neural_network").engines == frozenset({Engine.IR})


def test_replace_clears_old_aliases_and_duplicate_errors_are_typed() -> None:
    import pytest

    registry = ComponentRegistry()
    first = ComponentDescriptor("first", "First", "test", "First", aliases=("legacy",))
    registry.register(first, lambda: 1)
    assert registry.get("first").status == ComponentStatus.NATIVE
    assert registry.get("first").metadata["factory_registered"] is True
    with pytest.raises(DuplicateComponentError):
        registry.register(first)
    with pytest.raises(DuplicateFactoryError):
        registry.register_factory("first", lambda: 2)
    registry.register(
        ComponentDescriptor("first", "First v2", "test", "Second", aliases=("current",)),
        replace=True,
    )
    with pytest.raises(RegistryError, match="Unknown component"):
        registry.get("legacy")
    assert registry.get("current").key == "first"


def test_approximated_scientific_names_are_explicitly_annotated() -> None:
    from qforge_ai.circuits import CircuitFactory

    CircuitFactory()
    expected = {
        "ising_ansatz": "efficient_su2",
        "pauli_rotation_ansatz": "xyz_rotation_ansatz",
        "random_layers": "xyz_rotation_ansatz",
        "quantum_convolution": "qcnn",
        "quantum_pooling": "qcnn",
    }
    for key, alias_of in expected.items():
        descriptor = catalog.get(key)
        assert descriptor.metadata["approximation"] is True
        assert descriptor.metadata["alias_of"] == alias_of


def test_runnable_claims_are_verified_not_declared() -> None:
    """Guard against the catalog counting capacity the framework lacks."""

    from qforge_ai.catalog import capability_report, is_executable

    claimed = [
        descriptor.key
        for descriptor in catalog.list()
        if descriptor.metadata.get("runtime_contract") == "runnable"
    ]
    assert [key for key in claimed if not is_executable(key)] == []
    report = capability_report()
    assert report["executable"] == len(claimed)


def test_documented_catalog_numbers_match_the_implementation() -> None:
    """The published counts are generated, never hand-maintained."""

    import subprocess
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parent.parent / "scripts" / "check_catalog_docs.py"
    result = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
