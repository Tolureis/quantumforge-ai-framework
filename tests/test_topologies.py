"""Every entanglement topology the catalog advertises must actually build.

The catalog marks all twelve entanglement entries as executable.  Backing that
claim with a coverage-bearing test is what keeps it from becoming another
number that overstates capacity: each topology is generated, checked for
structural validity, and compiled on both engines.
"""

from __future__ import annotations

import importlib.util

import pytest

from qforge_ai import AnsatzSpec, CircuitSpec, MeasurementSpec
from qforge_ai.catalog import catalog
from qforge_ai.circuits import CircuitFactory
from qforge_ai.circuits.topology import entanglement_edges
from qforge_ai.errors import ConfigurationError
from qforge_ai.specs import ENTANGLEMENT_TOPOLOGIES, normalize_entanglement

HAS_QISKIT = importlib.util.find_spec("qiskit") is not None
HAS_PENNYLANE = importlib.util.find_spec("pennylane") is not None

PARAMETERLESS = sorted(ENTANGLEMENT_TOPOLOGIES - {"hardware", "user_defined"})


@pytest.mark.parametrize("topology", PARAMETERLESS)
@pytest.mark.parametrize("num_qubits", [2, 3, 4, 5])
def test_topology_edges_are_structurally_valid(topology: str, num_qubits: int) -> None:
    edges = entanglement_edges(num_qubits, topology, layer=0)
    for control, target in edges:
        assert 0 <= control < num_qubits
        assert 0 <= target < num_qubits
        assert control != target, f"{topology} produced a self-loop"
    if topology != "none":
        assert len(edges) == len(set(edges)), f"{topology} repeated an edge"


@pytest.mark.parametrize("topology", PARAMETERLESS)
def test_topology_connects_qubits_where_it_should(topology: str) -> None:
    edges = entanglement_edges(4, topology, layer=0)
    if topology == "none":
        assert edges == []
        return
    touched = {wire for edge in edges for wire in edge}
    assert touched, f"{topology} produced no entanglement at all"


def test_single_qubit_circuits_have_no_entanglement() -> None:
    for topology in PARAMETERLESS:
        assert entanglement_edges(1, topology) == []


def test_layer_dependent_topologies_alternate() -> None:
    for topology in ("pairwise", "alternating"):
        even = entanglement_edges(5, topology, layer=0)
        odd = entanglement_edges(5, topology, layer=1)
        assert even != odd, f"{topology} ignored the layer index"


def test_random_topology_is_seed_reproducible() -> None:
    first = entanglement_edges(6, "random", layer=0, seed=17)
    second = entanglement_edges(6, "random", layer=0, seed=17)
    third = entanglement_edges(6, "random", layer=0, seed=18)
    assert first == second
    assert first != third


def test_full_topology_is_complete() -> None:
    edges = entanglement_edges(4, "full")
    assert len(edges) == 4 * 3 // 2


def test_ring_closes_the_loop_but_pairs_do_not_duplicate() -> None:
    assert entanglement_edges(2, "ring") == [(0, 1)]
    assert (3, 0) in entanglement_edges(4, "ring")


def test_star_topology_centres_on_wire_zero() -> None:
    assert entanglement_edges(4, "star") == [(0, 1), (0, 2), (0, 3)]


def test_custom_topologies_require_and_validate_edges() -> None:
    for topology in ("hardware", "user_defined"):
        with pytest.raises(ConfigurationError, match="custom_edges"):
            entanglement_edges(3, topology)
        with pytest.raises(ConfigurationError, match="Invalid custom edge"):
            entanglement_edges(3, topology, custom_edges=[(0, 7)])
        assert entanglement_edges(3, topology, custom_edges=[(0, 2)]) == [(0, 2)]


def test_unknown_topology_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="Unknown entanglement"):
        entanglement_edges(3, "spaghetti")


@pytest.mark.parametrize("descriptor", catalog.list(category="entanglement"), ids=lambda d: d.key)
def test_every_catalog_topology_builds_a_circuit(descriptor) -> None:
    """Catalog names must be usable verbatim, including the custom ones."""

    topology = normalize_entanglement(descriptor.key)
    assert topology is not None
    options = (
        {"custom_edges": [(0, 1), (1, 2)]}
        if topology in {"hardware", "user_defined"}
        else {}
    )
    ir = CircuitFactory().build(
        CircuitSpec(
            num_qubits=3,
            ansatz=AnsatzSpec(
                name="hardware_efficient",
                layers=2,
                entanglement=descriptor.key,
                options=options,
            ),
            measurement=MeasurementSpec(name="expectation"),
        )
    )
    assert ir.validate()
    if topology != "none":
        assert ir.two_qubit_gate_count() > 0


@pytest.mark.skipif(not HAS_QISKIT, reason="qiskit not installed")
@pytest.mark.parametrize("topology", PARAMETERLESS)
def test_every_topology_compiles_on_qiskit(topology: str) -> None:
    from qforge_ai.backends import QiskitAdapter

    ir = CircuitFactory().build(
        CircuitSpec(
            num_qubits=4,
            ansatz=AnsatzSpec(name="hardware_efficient", layers=1, entanglement=topology),
            measurement=MeasurementSpec(name="expectation"),
        )
    )
    assert QiskitAdapter().compile(ir) is not None


@pytest.mark.skipif(not HAS_PENNYLANE, reason="pennylane not installed")
@pytest.mark.parametrize("topology", PARAMETERLESS)
def test_every_topology_compiles_on_pennylane(topology: str) -> None:
    from qforge_ai.backends import PennyLaneAdapter

    ir = CircuitFactory().build(
        CircuitSpec(
            num_qubits=4,
            ansatz=AnsatzSpec(name="hardware_efficient", layers=1, entanglement=topology),
            measurement=MeasurementSpec(name="expectation"),
        )
    )
    assert PennyLaneAdapter().compile(ir) is not None


@pytest.mark.skipif(not (HAS_QISKIT and HAS_PENNYLANE), reason="both engines required")
@pytest.mark.parametrize("topology", PARAMETERLESS)
def test_topologies_agree_across_engines(topology: str) -> None:
    """The point of a shared IR: the same topology means the same circuit."""

    import numpy as np

    from qforge_ai.benchmark import compare_engines

    ir = CircuitFactory().build(
        CircuitSpec(
            num_qubits=3,
            ansatz=AnsatzSpec(name="real_amplitudes", layers=1, entanglement=topology),
            measurement=MeasurementSpec(name="statevector"),
        )
    )
    groups = ir.parameter_groups()
    report = compare_engines(
        ir,
        inputs=np.linspace(0.1, 0.6, groups.get("input", 0)),
        weights=np.linspace(0.2, 0.9, groups.get("weight", 0)),
        tolerance=1e-6,
    )
    assert report.passed, f"{topology}: {report.notes} err={report.maximum_absolute_error}"
