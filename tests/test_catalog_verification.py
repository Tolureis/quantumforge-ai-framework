"""The catalog's executability claims must survive an actual execution.

``is_executable`` answers a static question -- does a route exist? -- and the
catalog was reporting that answer as capability.  These tests build a complete
experiment around every entry the catalog calls executable and run it on every
engine the entry claims, so a published claim that stops being true fails the
build instead of failing a user.
"""

from __future__ import annotations

import collections
import importlib.util

import pytest

from qforge_ai.catalog import executable_components, is_executable
from qforge_ai.specs import Engine
from qforge_ai.verification import (
    verify_catalog,
    verify_component,
    verify_composition,
    verify_compositions,
)

requires_qiskit = pytest.mark.skipif(
    importlib.util.find_spec("qiskit") is None or importlib.util.find_spec("qiskit_aer") is None,
    reason="qiskit and qiskit-aer required",
)
requires_pennylane = pytest.mark.skipif(
    importlib.util.find_spec("pennylane") is None, reason="pennylane required"
)


@requires_qiskit
@requires_pennylane
def test_every_executable_catalog_entry_actually_runs() -> None:
    results = verify_catalog()
    assert results, "verification sweep produced no results"
    failures = [item for item in results if item.failed]
    assert not failures, "catalog claims capability it does not have:\n" + "\n".join(
        f"  {item.engine} {item.category}/{item.key}: {item.detail}" for item in failures
    )


@requires_qiskit
@requires_pennylane
def test_verification_covers_every_declared_engine_of_every_entry() -> None:
    """No entry may pass the sweep merely by being skipped everywhere."""

    results = verify_catalog()
    verified: dict[str, set[str]] = collections.defaultdict(set)
    for item in results:
        if item.ok:
            verified[item.key].add(item.engine)
    for descriptor in executable_components():
        declared = {
            str(engine) for engine in descriptor.engines if engine != Engine.IR
        }
        assert declared <= verified[descriptor.key], (
            f"{descriptor.key} claims the {sorted(declared)} engines in the catalog "
            f"but was only run on {sorted(verified[descriptor.key])}"
        )


@requires_qiskit
@requires_pennylane
@pytest.mark.parametrize(
    "key",
    [
        "angle_encoding",
        "amplitude_encoding",
        "zz_feature_map",
        "data_reuploading_feature_map",
        "real_amplitudes",
        "qcnn",
        "hermitian_observable",
        "thermal_relaxation",
        "depolarizing_noise",
        "readout_error",
        "quantum_kernel_model",
        "quantum_classifier",
        "classical_quantum_classical_network",
    ],
)
@pytest.mark.parametrize("engine", [Engine.QISKIT, Engine.PENNYLANE])
def test_named_components_run_on_both_engines(key: str, engine: Engine) -> None:
    """Spot-check the entries a user is most likely to reach for."""

    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch required for model templates")
    result = verify_component(key, engine=engine)
    assert result.status == "ok", f"{key} on {engine}: {result.detail}"


@requires_qiskit
@requires_pennylane
def test_composition_claims_survive_a_real_build_and_run() -> None:
    """Components were verified individually; combinations never were.

    Each entry was checked against a baseline chosen to suit it -- basis
    encoding with a plain circuit run, every model template with angle
    encoding -- so two individually-true claims could still combine into a
    failure.  This sweep takes the cross product and requires that a
    composition the catalog's capabilities support actually runs, and that one
    they do not is refused by name rather than crashing inside an engine.
    """

    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch required for model templates")
    if importlib.util.find_spec("sklearn") is None:
        pytest.skip("scikit-learn required for kernel templates")
    results = verify_compositions()
    assert results, "composition sweep produced no results"
    failures = [item for item in results if item.failed]
    assert not failures, "composition claims are not backed by execution:\n" + "\n".join(
        f"  {item.engine} {item.key}: {item.detail}" for item in failures
    )


@requires_qiskit
@requires_pennylane
@pytest.mark.parametrize(
    "encoding,template,engine,expect_refusal",
    [
        # Individually executable, and impossible together.
        ("basis_encoding", "quantum_classifier", Engine.QISKIT, True),
        ("basis_encoding", "classical_quantum_classical", Engine.PENNYLANE, True),
        ("amplitude_encoding", "quantum_classifier", Engine.QISKIT, True),
        ("superposition_encoding", "classical_quantum_classical", Engine.PENNYLANE, True),
        # ...and the combinations that must keep working.
        ("basis_encoding", "quantum_classifier", Engine.PENNYLANE, False),
        ("amplitude_encoding", "classical_quantum_classical", Engine.PENNYLANE, False),
        ("zz_feature_map", "quantum_kernel_model", Engine.QISKIT, False),
        ("angle_encoding", "classical_quantum_classical", Engine.QISKIT, False),
    ],
)
def test_named_compositions_behave_as_the_catalog_declares(
    encoding: str, template: str, engine: Engine, expect_refusal: bool
) -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch required for model templates")
    if importlib.util.find_spec("sklearn") is None:
        pytest.skip("scikit-learn required for kernel templates")
    result = verify_composition(encoding, template, engine=engine)
    assert result.status == "ok", result.detail
    assert ("refused" in result.detail) is expect_refusal, result.detail


def test_is_executable_still_answers_the_static_question() -> None:
    assert is_executable("angle_encoding")
    assert is_executable("quantum_kernel_model")
    # A discovery-only research family stays non-executable.
    assert not is_executable("quantum_diffusion_model")
    assert not is_executable("surface_code")
