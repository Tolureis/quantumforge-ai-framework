"""Backend recommendation, adapter dispatch and composite template plumbing.

``BackendRouter.recommend`` and ``models.templates`` were the least-covered
runtime modules.  A recommendation that cannot be executed on the machine that
produced it is a capability claim like any other, so these tests require the
router's output to survive its own preflight.
"""

from __future__ import annotations

import importlib.util

import pytest

from qforge_ai import AnsatzSpec, CircuitSpec, EncodingSpec, MeasurementSpec
from qforge_ai.backends import BackendRouter
from qforge_ai.capabilities import CapabilityRegistry
from qforge_ai.circuits import CircuitFactory
from qforge_ai.errors import CapabilityError, MissingDependencyError
from qforge_ai.ir import CircuitIR
from qforge_ai.specs import BackendSpec, Engine

HAS_QISKIT = importlib.util.find_spec("qiskit") is not None
HAS_PENNYLANE = importlib.util.find_spec("pennylane") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None


def _ir(num_qubits: int = 3) -> CircuitIR:
    return CircuitFactory().build(
        CircuitSpec(
            num_qubits=num_qubits,
            encoding=EncodingSpec(name="angle"),
            ansatz=AnsatzSpec(name="real_amplitudes", layers=1),
            measurement=MeasurementSpec(name="expectation"),
        )
    )


# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------


def test_router_rejects_the_ir_engine_for_execution() -> None:
    with pytest.raises(CapabilityError, match="direct simulation"):
        BackendRouter().create(BackendSpec(engine=Engine.IR))


@pytest.mark.skipif(not HAS_QISKIT, reason="qiskit not installed")
def test_router_creates_a_qiskit_adapter() -> None:
    adapter = BackendRouter().create(BackendSpec(engine=Engine.QISKIT, device="aer"))
    assert adapter.engine == "qiskit"


@pytest.mark.skipif(not HAS_PENNYLANE, reason="pennylane not installed")
def test_router_creates_a_pennylane_adapter() -> None:
    adapter = BackendRouter().create(
        BackendSpec(engine=Engine.PENNYLANE, device="default.qubit")
    )
    assert adapter.engine == "pennylane"


def test_router_recommends_aer_for_dynamic_circuits() -> None:
    ir = CircuitIR(num_qubits=2)
    ir.add("H", 0).measure_wire(0, 0)
    recommendation = BackendRouter().recommend(ir)
    assert recommendation.engine == Engine.QISKIT
    assert recommendation.device == "aer"


def test_router_recommends_density_matrix_for_noise() -> None:
    recommendation = BackendRouter().recommend(_ir(), noisy=True)
    assert recommendation.method == "density_matrix"


def test_router_recommends_matrix_product_state_for_wide_shallow_circuits() -> None:
    ir = CircuitFactory().build(
        CircuitSpec(
            num_qubits=22,
            encoding=EncodingSpec(name="angle"),
            ansatz=AnsatzSpec(name="real_amplitudes", layers=1, entanglement="linear"),
            measurement=MeasurementSpec(name="expectation"),
        )
    )
    assert BackendRouter().recommend(ir).method == "matrix_product_state"


def test_router_refuses_a_gpu_recommendation_it_cannot_back() -> None:
    if CapabilityRegistry().get(Engine.QISKIT).gpu:
        pytest.skip("this machine genuinely has an Aer GPU build")
    with pytest.raises(CapabilityError, match="GPU"):
        BackendRouter().recommend(_ir(), gpu=True)


@pytest.mark.skipif(not (HAS_QISKIT and HAS_PENNYLANE), reason="both engines required")
@pytest.mark.parametrize("noisy", [False, True])
def test_recommendation_survives_its_own_preflight(noisy: bool) -> None:
    """A recommendation the framework cannot then validate is worthless."""

    ir = _ir()
    if noisy:
        ir = CircuitFactory().build(
            CircuitSpec(
                num_qubits=3,
                encoding=EncodingSpec(name="angle"),
                ansatz=AnsatzSpec(name="real_amplitudes", layers=1),
                measurement=MeasurementSpec(name="density_matrix"),
            )
        )
    recommendation = BackendRouter().recommend(ir, noisy=noisy)
    CapabilityRegistry().validate(ir, recommendation)
    assert BackendRouter().create(recommendation) is not None


@pytest.mark.skipif(not HAS_PENNYLANE, reason="pennylane not installed")
def test_default_recommendation_names_an_installed_device() -> None:
    from qforge_ai.runtime import pennylane_device_constructible

    recommendation = BackendRouter().recommend(_ir())
    if recommendation.engine == Engine.PENNYLANE:
        assert pennylane_device_constructible(recommendation.device)


# --------------------------------------------------------------------------
# Composite templates
# --------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
@pytest.mark.parametrize("reduction", ["mean", "sum", "concat"])
def test_quantum_parallel_reductions(reduction: str) -> None:
    import torch
    from torch import nn

    from qforge_ai.models.templates import QuantumParallel

    branches = [nn.Linear(4, 3), nn.Linear(4, 3)]
    model = QuantumParallel.build(branches, reduction=reduction)
    output = model(torch.randn(5, 4))
    expected = 6 if reduction == "concat" else 3
    assert output.shape == (5, expected)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_quantum_parallel_rejects_an_unknown_reduction() -> None:
    from torch import nn

    from qforge_ai.errors import ConfigurationError
    from qforge_ai.models.templates import QuantumParallel

    with pytest.raises(ConfigurationError):
        QuantumParallel.build([nn.Linear(2, 2), nn.Linear(2, 2)], reduction="median")


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_quantum_parallel_requires_branches() -> None:
    from qforge_ai.errors import ConfigurationError
    from qforge_ai.models.templates import QuantumParallel

    with pytest.raises(ConfigurationError):
        QuantumParallel.build([], reduction="mean")


# --------------------------------------------------------------------------
# Optional-dependency boundary
# --------------------------------------------------------------------------


def test_missing_dependency_error_names_the_extra() -> None:
    from qforge_ai.backends._optional import require

    with pytest.raises(MissingDependencyError) as info:
        require(
            "qforge_definitely_not_installed",
            package="ghost-package",
            extra="ghost",
            feature="ghost feature",
        )
    message = str(info.value)
    assert "ghost-package" in message
    assert "quantumforge-ai[ghost]" in message
    assert "ghost feature" in message


def test_require_returns_an_installed_module() -> None:
    from qforge_ai.backends._optional import require

    module = require("json", package="json", extra="core", feature="test")
    assert module.dumps({"a": 1}) == '{"a": 1}'


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_quantum_residual_adds_the_skip_connection() -> None:
    import torch
    from torch import nn

    from qforge_ai.models.templates import QuantumResidual

    layer = nn.Linear(3, 3)
    model = QuantumResidual.build(layer)
    sample = torch.randn(2, 3)
    assert torch.allclose(model(sample), layer(sample) + sample)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_quantum_residual_projects_mismatched_widths() -> None:
    import torch
    from torch import nn

    from qforge_ai.models.templates import QuantumResidual

    model = QuantumResidual.build(nn.Linear(3, 5), projection=nn.Linear(3, 5, bias=False))
    assert model(torch.randn(4, 3)).shape == (4, 5)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_quantum_sequential_requires_layers() -> None:
    from qforge_ai.errors import ConfigurationError
    from qforge_ai.models.templates import QuantumSequential

    with pytest.raises(ConfigurationError):
        QuantumSequential.build()


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_quantum_sequential_chains_layers() -> None:
    import torch
    from torch import nn

    from qforge_ai.models.templates import QuantumSequential

    model = QuantumSequential.build(nn.Linear(3, 4), nn.GELU(), nn.Linear(4, 2))
    assert model(torch.randn(6, 3)).shape == (6, 2)
