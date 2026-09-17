# QuantumForge AI

QuantumForge AI is a Python framework for building, simulating, training, and
comparing quantum, hybrid quantum–classical, and quantum-kernel models on
Qiskit and PennyLane using the same experiment definition.

Version **1.2.5** establishes a shared numerical contract between the two engines:
state preparation, multi-qubit observables, and results such as `counts` use
the same wire-major ordering. Every noise channel has one canonical Kraus
definition, applied in the same way by both engines. Noisy analytical execution
uses density-matrix simulation throughout the direct execution, `EstimatorQNN`,
and kernel paths. The **quantum kernel has a single implementation shared by both
engines** (`qforge_ai.models.kernel`), including its mathematics, noise placement,
input-width contract, and capability preflight. The engines provide only two
primitives: the encoded state of one sample and the measured overlap of a pair.

The IR validates physical constraints, including real-valued rotation angles
and observable coefficients. CI exercises runnable catalog entries and
**encoding × template × engine combinations**. Unsupported settings are intended
to raise explicit errors naming the option, rather than being silently discarded.
The YAML/dictionary path applies the same configuration checks as direct
dataclass construction.

Cross-engine equivalence is covered by
[`tests/test_cross_engine_equivalence.py`](tests/test_cross_engine_equivalence.py).
Analytical comparisons use floating-point tolerances; finite-shot results are
compared within sampling uncertainty rather than requiring identical samples.

The framework has two layers:

1. **Executable core:** shared `CircuitIR`, built-in encoding and ansatz builders,
   Qiskit and PennyLane adapters, PyTorch hybrid models, noise, training,
   benchmarking, experiment management, and a CLI.
2. **Extensible catalog:** circuit and model families across sections 1–21,
   registered with maturity, engine support, capabilities, and dependencies.
   Entries include `native`, `composite`, `experimental`, and `descriptor` levels.

> Even in “fully quantum” models, data loading, optimization, and loss evaluation
> usually involve a classical runtime. The framework therefore uses the name
> `quantum_only` and explicitly distinguishes classical and quantum resources
> in its reports.

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Custom Circuit](#custom-circuit)
- [Experiments with YAML](#experiments-with-yaml)
- [Quantum-kernel execution and BackendSpec](#quantum-kernel-execution-and-backendspec)
- [State preparation, ancillas, and wire ordering](#state-preparation-ancillas-and-wire-ordering)
- [Physical constraints in the IR](#physical-constraints-in-the-ir)
- [Noise: one canonical definition for both engines](#noise-one-canonical-definition-for-both-engines)
- [Training](#training)
- [Catalog coverage: sections 1–21](#catalog-coverage-sections-121)
- [Support levels](#support-levels)
- [Design principles](#design-principles)
- [Development](#development)
- [License](#license)

## Installation

Requires **Python 3.11 or later**. Run these commands from the extracted project
directory containing `pyproject.toml`.

Install the basic catalog and shared IR:

```bash
pip install -e .
```

Install the full framework, including Qiskit and PennyLane:

```bash
pip install -e ".[all]"
```

Alternatively, select only the required engine and PyTorch integration:

```bash
pip install -e ".[qiskit,torch]"
pip install -e ".[pennylane,torch]"
```

## Quick start

Build a circuit without importing either quantum engine:

```python
from qforge_ai import AnsatzSpec, CircuitSpec, EncodingSpec, MeasurementSpec
from qforge_ai.circuits import CircuitFactory

spec = CircuitSpec(
    num_qubits=4,
    encoding=EncodingSpec(name="angle", feature_dimension=4, rotation="Y"),
    ansatz=AnsatzSpec(name="hardware_efficient", layers=2, entanglement="ring"),
    measurement=MeasurementSpec(name="expectation", wires=(0, 1, 2, 3)),
)

ir = CircuitFactory().build(spec)
print(ir.summary())
```

Convert the same IR into a Qiskit circuit and execute it:

```python
from qforge_ai.backends import QiskitAdapter

adapter = QiskitAdapter()
circuit = adapter.compile(ir)
weights = [0.0] * ir.parameter_groups()["weight"]
result = adapter.run(ir, inputs=[0.1, 0.2, 0.3, 0.4], weights=weights)
print(result.values)
```

Create a PennyLane QNode from the same IR:

```python
from qforge_ai.backends import PennyLaneAdapter

adapter = PennyLaneAdapter(device="default.qubit")
qnode = adapter.compile(ir, interface="torch", diff_method="parameter-shift")
```

## Custom Circuit

QuantumForge AI 1.2.5 supports two practical ways to create a custom circuit:

| Approach | Use it when | Integration |
| --- | --- | --- |
| Build `CircuitIR` directly | You need exact control over gate order, parameters, ancillas, and measurements | Pass the IR to an adapter, `ModelFactory`, or a kernel builder |
| Register an encoding or ansatz builder | You want a reusable component selectable by name in `CircuitSpec` or YAML | Register a `ComponentDescriptor` and a callable in the catalog |

The base installation is sufficient to construct, validate, and inspect the IR.
Execution requires the relevant quantum-engine extra; the training example also
requires PyTorch. The examples below target the API in the supplied **1.2.5**
source release.

### 1. Build a parameterized circuit directly

This complete example defines three data qubits, three input features, two
trainable layers, and three Pauli-Z expectation outputs. Each layer applies
`RY` and `RZ` rotations to every data wire, followed by a linear CNOT chain.

```python
from qforge_ai import CircuitIR, Measurement, Observable, ParameterRef


def build_custom_circuit() -> CircuitIR:
    ir = CircuitIR(num_qubits=3, name="custom_three_qubit_qnn")

    # Input features: x[0], x[1], x[2]. Angles are in radians.
    for wire in ir.data_wires:
        ir.add("RY", wire, ParameterRef("input", wire), role="encoding")

    weight_index = 0
    for layer in range(2):
        for wire in ir.data_wires:
            for gate in ("RY", "RZ"):
                ir.add(
                    gate,
                    wire,
                    ParameterRef("weight", weight_index),
                    role="ansatz",
                    layer=layer,
                )
                weight_index += 1
        ir.add("CX", (0, 1), role="ansatz", layer=layer)
        ir.add("CX", (1, 2), role="ansatz", layer=layer)

    ir.measure(
        Measurement(
            kind="expectation",
            wires=ir.data_wires,
            observables=tuple(Observable("Z", (wire,)) for wire in ir.data_wires),
        )
    )
    return ir.validate()


custom_ir = build_custom_circuit()
assert custom_ir.parameter_groups() == {"input": 3, "weight": 12}
print(custom_ir.summary())
```

This circuit has **19 operations**, including **4 two-qubit gates**, and
**12 trainable parameters**. The terminal expectation measurement returns
`[⟨Z₀⟩, ⟨Z₁⟩, ⟨Z₂⟩]`.

| API | Meaning |
| --- | --- |
| `ir.add("H", 0)` | Append a fixed single-qubit gate |
| `ir.add("CX", (0, 1))` | Append a CNOT with control 0 and target 1 |
| `ParameterRef("input", i)` | Read `inputs[i]` at execution time |
| `ParameterRef("weight", i)` | Read the trainable or externally supplied `weights[i]` |
| `ParameterRef("input", i, scale=s, offset=b)` | Resolve the angle as `s * inputs[i] + b` |
| `role="encoding"` | Mark an operation as part of the data-loading feature map |
| `role="ansatz"` | Mark an operation as part of the variational circuit |
| `ir.validate()` | Validate the IR structure and declared operation/measurement contracts |

Use the parameter groups **`input`** and **`weight`** for the built-in adapters.
Keep indices contiguous and zero-based. Reusing the same group/index pair
shares a parameter across gates; it does not create another independent weight.
`parameter_groups()` returns each group's required vector width
(`max_index + 1`), while `parameter_count()` counts distinct references.

Tag every feature-map operation, including fixed gates and entanglers, with
`role="encoding"` if the circuit will be used as a kernel. Kernel extraction
uses these tags; it does not infer the feature map from the circuit name.

### 2. Execute the same custom IR on either engine

Run this block after creating `custom_ir` above. It requires both engine extras.
An application using one engine can retain just its corresponding backend.

```python
import numpy as np

from qforge_ai import BackendSpec, CapabilityRegistry, Engine
from qforge_ai.backends import BackendRouter

inputs = np.array([0.1, 0.2, 0.3], dtype=float)
weights = np.linspace(-0.2, 0.2, custom_ir.parameter_groups()["weight"])
backends = (
    BackendSpec(engine=Engine.QISKIT, device="aer", shots=None, seed=7),
    BackendSpec(engine=Engine.PENNYLANE, device="default.qubit", shots=None, seed=7),
)

outputs = {}
for backend in backends:
    warnings = CapabilityRegistry().validate(custom_ir, backend)
    adapter = BackendRouter().create(backend)
    result = adapter.run(custom_ir, inputs=inputs, weights=weights)
    outputs[backend.engine] = result.as_array()
    print(backend.engine, result.values, warnings)

np.testing.assert_allclose(
    outputs[Engine.QISKIT], outputs[Engine.PENNYLANE], atol=1e-9, rtol=0,
)
```

Create adapters from `BackendSpec` when settings such as shots, noise, seed,
or simulator method must be carried through. The direct workflow calls
`CapabilityRegistry.validate()` explicitly: `BackendRouter.create()` constructs
an adapter but does not itself run circuit capability preflight.

For compilation only, use `QiskitAdapter.from_spec(backend).compile(custom_ir)`
with a Qiskit spec, or `PennyLaneAdapter.from_spec(backend).compile(custom_ir, ...)`
with a PennyLane spec. Qiskit compilation returns a `QuantumCircuit`; PennyLane
compilation returns a QNode. A raw native circuit or QNode is not a substitute
for the `CircuitIR` argument expected by `ModelFactory.build()`.

### 3. Change the measurement or add ancillas

An IR can contain **one terminal `Measurement`**. For multiple expectation
outputs, put several observables inside that measurement. To request a different
terminal result, copy the IR and replace its measurement:

```python
from qforge_ai import Measurement

counts_ir = custom_ir.copy(name="custom_counts")
counts_ir.measurements.clear()
counts_ir.measure(Measurement(kind="counts", wires=(0, 1, 2), shots=2048))
counts_ir.validate()
```

Run `counts_ir` with the same input and weight vectors. A backend with
`shots=None` uses the measurement's 2048 shots here. Counts use wire-major keys:
`"101"` means wire 0 is 1, wire 1 is 0, and wire 2 is 1.

For a joint observable such as `Z⊗Z`, supply its Hermitian matrix. A named
`Observable("Z", ...)` accepts exactly one wire:

```python
from qforge_ai import Measurement, Observable

zz_matrix = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, -1.0, 0.0, 0.0),
    (0.0, 0.0, -1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)
zz_ir = custom_ir.copy(name="custom_zz_expectation")
zz_ir.measurements.clear()
zz_ir.measure(
    Measurement(
        kind="expectation",
        wires=(0, 1),
        observables=(Observable("HERMITIAN", (0, 1), matrix=zz_matrix),),
    )
)
zz_ir.validate()
```

Ancilla counts have different meanings at the two construction levels:

| Constructor | `num_qubits` means | Example with 3 data qubits and 1 ancilla |
| --- | --- | --- |
| `CircuitSpec` | Number of data qubits; the factory adds ancillas | `CircuitSpec(num_qubits=3, ancilla_qubits=1)` |
| `CircuitIR` | Total number of qubits, already including ancillas | `CircuitIR(num_qubits=4, ancilla_qubits=1)` |

This separate circuit explicitly uses the reserved ancilla:

```python
from qforge_ai import CircuitIR, Measurement, ParameterRef

ancilla_ir = CircuitIR(num_qubits=4, ancilla_qubits=1, name="custom_ancilla")
assert ancilla_ir.data_wires == (0, 1, 2)
assert ancilla_ir.ancilla_wires == (3,)

for wire in ancilla_ir.data_wires:
    ancilla_ir.add("RY", wire, ParameterRef("input", wire), role="encoding")

ancilla = ancilla_ir.ancilla_wires[0]
ancilla_ir.add("CX", (0, ancilla))
ancilla_ir.add("CX", (2, ancilla))
ancilla_ir.measure(Measurement(kind="probability", wires=(ancilla,)))
ancilla_ir.validate()
```

Use `ir.data_wires` for data-loading and ansatz loops so that reserved ancillas
are not encoded or trained accidentally. For a full statevector, measure
`tuple(range(ir.num_qubits))`. For a subsystem density matrix in a manually
built IR, use `Measurement(kind="density_matrix", wires=(...))`; the factory
maps `MeasurementSpec(name="reduced_density_matrix", ...)` to that IR kind.

### 4. Train a hybrid model with the custom circuit

The next block uses `custom_ir` from step 1 and requires `[pennylane,torch]`.
It is a small training demonstration, not a model-quality benchmark.

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

from qforge_ai import (
    BackendSpec, CapabilityRegistry, Engine, GradientMethod,
    ModelMode, ModelSpec, TorchTrainer, TrainingSpec,
)
from qforge_ai.models import ModelFactory

backend = BackendSpec(
    engine=Engine.PENNYLANE,
    device="default.qubit",
    shots=None,
    gradient=GradientMethod.PARAMETER_SHIFT,
    seed=7,
)
CapabilityRegistry().validate(custom_ir, backend)
model = ModelFactory().build(
    custom_ir,
    ModelSpec(
        mode=ModelMode.HYBRID,
        template="classical_quantum_classical",
        input_dimension=3,
        hidden_dimensions=(8,),
        output_dimension=2,
    ),
    backend,
    seed=7,
)

X = torch.tensor(
    [[0.1, 0.2, 0.3], [0.2, 0.1, 0.4], [0.8, 0.7, 0.6], [0.9, 0.8, 0.7]],
    dtype=torch.float32,
)
y = torch.tensor([0, 0, 1, 1], dtype=torch.long)
loader = DataLoader(TensorDataset(X, y), batch_size=2, shuffle=False)
trainer = TorchTrainer(
    TrainingSpec(epochs=3, optimizer="adam", learning_rate=0.01, loss="cross_entropy"),
    seed=7,
)
training_result = trainer.fit(model, loader, device="cpu")
print(training_result.history[-1].train_loss)
```

The classical encoder maps three input features to the circuit's three input
parameters. The quantum layer returns three expectations, and the classical
head maps them to two class logits. `ModelFactory.build(..., seed=7)` seeds
initialization before constructing the layers.

The same rotation-based IR can use a Qiskit backend with `device="aer"` and the
`[qiskit,torch]` extra; expectation measurement selects `EstimatorQNN` by default.
For `mode=ModelMode.QUANTUM_ONLY` and `template="quantum_classifier"`, this IR
requires `output_dimension=3`, because no classical head changes its output width.

### 5. Register a reusable custom ansatz

An ansatz builder has the signature `(ir: CircuitIR, spec: AnsatzSpec) -> None`.
It appends operations to the supplied IR. `CircuitFactory` calls the encoding
builder first, then the ansatz builder, tags their operations with the appropriate
roles, and adds the terminal measurement.

The following custom builder uses configurable rotations and **CZ entanglers**.
It respects the standard layer, topology, final-rotation, and topology-option
fields. Save this registration block as `my_custom_components.py` if you want
to reuse it from a separate script:

```python
from qforge_ai import (
    AnsatzSpec, CircuitIR, ComponentDescriptor, ComponentStatus, Engine, ParameterRef,
)
from qforge_ai.catalog import catalog
from qforge_ai.circuits import entanglement_edges
from qforge_ai.errors import ConfigurationError


def custom_cz_ansatz(ir: CircuitIR, spec: AnsatzSpec) -> None:
    options = dict(spec.options)
    unknown = set(options) - {"layer_offset", "seed", "custom_edges"}
    if unknown:
        raise ConfigurationError(f"custom_cz_ansatz does not support: {sorted(unknown)}")

    weight_index = ir.parameter_groups().get("weight", 0)

    def add_rotations() -> None:
        nonlocal weight_index
        for wire in ir.data_wires:
            for gate in spec.rotation_blocks:
                ir.add(gate, wire, ParameterRef("weight", weight_index))
                weight_index += 1

    for layer in range(spec.layers):
        add_rotations()
        edges = entanglement_edges(
            ir.num_data_qubits,
            spec.entanglement,
            layer=layer + int(options.get("layer_offset", 0)),
            seed=int(options.get("seed", 42)),
            custom_edges=options.get("custom_edges"),
        )
        for control, target in edges:
            ir.add("CZ", (ir.data_wires[control], ir.data_wires[target]))

    if not spec.skip_final_rotation:
        add_rotations()


catalog.register(
    ComponentDescriptor(
        key="my_custom_cz_ansatz",
        display_name="Custom rotation-CZ ansatz",
        category="ansatz",
        description="Parameterized rotations with configurable CZ entanglement.",
        status=ComponentStatus.NATIVE,
        engines=frozenset({Engine.IR, Engine.QISKIT, Engine.PENNYLANE}),
    ),
    factory=custom_cz_ansatz,
)
```

Register once per Python process. Duplicate registration raises an error; use
`replace=True` only when you explicitly intend to replace that entry during
development. Supplying a descriptor without a callable does not implement a
new circuit builder.

After running or importing the registration block, build the custom component
through the standard factory:

```python
from qforge_ai import AnsatzSpec, CircuitSpec, EncodingSpec, MeasurementSpec
from qforge_ai.circuits import CircuitFactory

registered_spec = CircuitSpec(
    name="registered_custom_circuit",
    num_qubits=3,
    ancilla_qubits=1,
    encoding=EncodingSpec(name="angle", feature_dimension=3, rotation="Y"),
    ansatz=AnsatzSpec(
        name="my_custom_cz_ansatz",
        layers=2,
        entanglement="ring",
        rotation_blocks=("RY", "RZ"),
        skip_final_rotation=True,
    ),
    measurement=MeasurementSpec(name="expectation", wires=(0, 1, 2)),
)

registered_ir = CircuitFactory().build(registered_spec)
assert registered_ir.parameter_groups() == {"input": 3, "weight": 12}
assert registered_ir.ancilla_wires == (3,)
print(registered_ir.summary())
```

`skip_final_rotation=True` gives `2 × 3 × 2 = 12` weights here. Setting it to
`False` adds one final rotation block and gives 18 weights. Starting from the
IR's existing weight width prevents collisions with earlier trainable blocks.
Respecting `layer_offset` also allows topology scheduling across factory-managed
data-reuploading cycles.

To register a custom **encoding**, use `category="encoding"` and a callable with
signature `(ir: CircuitIR, spec: EncodingSpec) -> None`. Append data-loading gates
using `ParameterRef("input", ...)`, define and enforce the feature-capacity
contract, and implement or explicitly reject each configurable option.
`CircuitFactory` assigns the encoding role automatically.

If your plugin introduces a new option key, declare it **before constructing
the spec or loading YAML** using
`qforge_ai.specs.register_option_keys("encoding", "my_option")` or the
corresponding `"ansatz"` section. Declaring a key only allows it through spec
validation; the builder must still validate and consume its value.

### 6. Select a registered component from YAML

Save the following configuration as `custom_experiment.yaml`:

```yaml
name: custom_cz_experiment
seed: 7
circuit:
  name: registered_custom_circuit
  num_qubits: 3
  ancilla_qubits: 1
  encoding:
    name: angle
    feature_dimension: 3
    rotation: Y
  ansatz:
    name: my_custom_cz_ansatz
    layers: 2
    entanglement: ring
    rotation_blocks: [RY, RZ]
    skip_final_rotation: true
  measurement:
    name: expectation
    wires: [0, 1, 2]
backend:
  engine: pennylane
  device: default.qubit
  shots: null
  seed: 7
model:
  mode: hybrid
  template: classical_quantum_classical
  input_dimension: 3
  hidden_dimensions: [8]
  output_dimension: 2
training:
  epochs: 3
  optimizer: adam
  learning_rate: 0.01
```

Place `my_custom_components.py` beside the script below, then load the
registration before preparing the experiment:

```python
import my_custom_components  # Registers my_custom_cz_ansatz in this process.

from qforge_ai import ExperimentRunner

runner = ExperimentRunner()
prepared = runner.prepare("custom_experiment.yaml")
print(prepared.circuit.summary())
model = runner.build_model(prepared)
```

For training, call `runner.train(prepared, loader, model=model)` with a data
loader such as the one in step 4. The YAML file describes the component by name;
it does not embed or import the Python builder. In 1.2.5, the stock CLI does not
automatically import `my_custom_components` or call `load_plugins()`. Use a
Python entry script that performs registration first. Installed entry-point
plugins can be registered explicitly with `qforge_ai.plugins.load_plugins()`
before preparing the experiment.

### 7. Use a custom feature map as a quantum kernel

The `custom_ir` from step 1 already tags its three input rotations with
`role="encoding"`. The kernel uses those operations and excludes the tagged
ansatz and its 12 trainable weights:

```python
import numpy as np

from qforge_ai import BackendSpec, Engine
from qforge_ai.models import build_quantum_kernel

kernel, implementation = build_quantum_kernel(
    custom_ir,
    backend=BackendSpec(engine=Engine.PENNYLANE, device="default.qubit", seed=7),
    fidelity="statevector",
)
X_kernel = np.array([[0.1, 0.2, 0.3], [0.3, 0.1, 0.4]], dtype=float)
gram = kernel.evaluate(X_kernel)
assert gram.shape == (2, 2)
assert kernel.feature_dimension == 3
print(implementation, gram)
```

For a registered custom encoding, `CircuitFactory().build_feature_map(spec)`
builds only the encoding, without the ansatz. The core fixed-kernel path requires
an encoding with input parameters, no trainable weight references, and no
measurement-dependent state preparation. Its sampled compute–uncompute route
also requires invertible feature-map operations. See the
[kernel contract](#quantum-kernel-execution-and-backendspec) for shots and noise.

### 8. Validation and extension boundaries

| Requirement | Correct usage |
| --- | --- |
| Gate arity | `X` has one wire and no parameters; `RY` has one wire and one angle; `CX` has two distinct wires and no parameters |
| Parameter values | Use finite real rotation angles, scales, and offsets; use complex amplitudes only where supported by state preparation |
| Terminal measurements | Use one `Measurement`; place multiple observables inside it or execute separately copied circuits |
| Multi-qubit observables | Supply a Hermitian matrix of size `2**len(wires)` in wire-major order |
| Ancillas | Reserve them explicitly and use `data_wires` for ordinary encoding/ansatz loops |
| Dynamic circuits | Use `measure_wire(wire, clbit)`, `add_conditional(..., clbit=..., value=0 or 1)`, and `reset(wire)`; validate the selected backend and measurement combination |
| Model compatibility | Direct simulation support does not automatically imply differentiable QNN or kernel support |
| Reproducibility | Set backend and model/training seeds explicitly; equal seeds do not imply identical finite-shot samples across engines |

A **custom circuit** usually combines operations already supported by the
adapters. A genuinely new **gate type** additionally requires backend translation
and capability support. `register_operation_signature()` in `qforge_ai.ir` can
declare its wire/parameter arity, but does not implement that gate in Qiskit or
PennyLane. Unknown operation names may pass the generic IR layer and still be
rejected during capability checks or compilation. Decompose custom gates into
supported operations when possible.

Relevant implementation references:

- [`src/qforge_ai/ir.py`](src/qforge_ai/ir.py): IR, parameters, operations, observables, and validation.
- [`src/qforge_ai/circuits/factory.py`](src/qforge_ai/circuits/factory.py): builder invocation, role tagging, and measurements.
- [`src/qforge_ai/registry.py`](src/qforge_ai/registry.py): component and factory registration.
- [`src/qforge_ai/specs.py`](src/qforge_ai/specs.py): configuration objects and option-key registration.
- [`src/qforge_ai/models/factory.py`](src/qforge_ai/models/factory.py): quantum and hybrid model construction.
- [`src/qforge_ai/models/kernel.py`](src/qforge_ai/models/kernel.py): shared feature-map extraction and kernel execution.

## Experiments with YAML

The following example matches the built-in IDS hybrid experiment:

```yaml
name: ids_hybrid_qnn
seed: 42
circuit:
  num_qubits: 6
  encoding:
    name: data_reuploading
    feature_dimension: 12
    repeats: 2
  ansatz:
    name: hardware_efficient
    layers: 2
    entanglement: ring
  measurement:
    name: expectation
    wires: [0, 1, 2, 3, 4, 5]
backend:
  engine: pennylane
  device: default.qubit
  shots: null
model:
  mode: hybrid
  template: classical_quantum_classical
  output_dimension: 6
training:
  epochs: 20
  optimizer: adam
  learning_rate: 0.001
```

```bash
qforge validate examples/configs/ids_hybrid.yaml
qforge inspect examples/configs/ids_hybrid.yaml
qforge catalog --category encoding
```

`data_reuploading` reloads **the same complete feature vector** in every cycle,
followed by one ansatz layer. `encoding.repeats` must be at least 2 and must
match `ansatz.layers`.

When the feature count exceeds the data-qubit count, features are assigned to
wires cyclically. If multiple features share a wire, the rotation axis cycles
through **X → Y → Z**, starting from the configured axis. Consecutive rotations
around the same axis add (`RY(a) RY(b) == RY(a+b)`), so loading all features on
one axis can make different feature vectors produce the same state. Alternating
axes avoids this particular collapse.

This does **not** guarantee lossless or invertible encoding. A single-qubit state
has a limited number of independent degrees of freedom, and repeated loading
does not guarantee that every original feature can be recovered. Choose enough
qubits and encoding slots for the intended feature dimension. Builders with an
explicit capacity contract reject excess features instead of silently omitting
them; `repeats` must follow the semantics of the selected encoding.

## Quantum-kernel execution and BackendSpec

As of 1.2.5, the quantum kernel has **one shared implementation** in
`qforge_ai.models.kernel`. The framework defines what the kernel computes,
how the Gram matrix is constructed, and which inputs are valid. The contract is
resolved centrally in `qforge_ai.models.kernel_contract`. Each supported kernel
kind has a path for both engines, subject to the selected backend's capabilities.

| `fidelity` | Computation | Shots | Noise |
| --- | --- | --- | --- |
| `statevector` | Squared absolute inner product of the two encoded pure states | None | None |
| `hilbert_schmidt` (alias: `density_matrix`) | `Tr[ρ(x) ρ(y)]`, the Hilbert–Schmidt inner product | None | Supported by both engines |
| `sampler` | Compute–uncompute sampling | **Required** | Supported, including readout noise |
| `auto` | `sampler` when shots or readout noise are present; otherwise `hilbert_schmidt` for gate noise, or `statevector` for an ideal backend | Depends on the resolved kind | Depends on the resolved kind |

The pure-state fidelity kernel is:

$$
K(x,y)=\left|\langle\phi(x)\mid\phi(y)\rangle\right|^2.
$$

- `statevector` is analytical, noiseless, and uses CPU statevector simulation.
  Incompatible requests involving `shots`, non-ideal `noise`, `method`,
  `compute_device`, or execution options such as `backend_options`,
  `run_options`, and `primitive_options` produce an error naming the settings
  that cannot be applied. `NoiseSpec(name="ideal")` is not an actual channel.
- `sampler` requires an explicit shot count **on both engines**; it does not
  silently fall back to 1024 shots. Both engines insert the same Kraus channels
  at the IR-defined positions. **Every Gram entry, including diagonal entries,
  is measured.** Replacing a noisy diagonal measurement with 1 would mix results
  from different experimental definitions. Square training Gram matrices are
  symmetrized and projected onto the positive semidefinite cone with
  `enforce_psd`, since `K(x,y)` and `K(y,x)` are separate sampling experiments.
- `hilbert_schmidt` remains a valid Mercer kernel under noise: it is the inner
  product of the encoded density matrices and is therefore symmetric and
  positive semidefinite by construction. It is not mixed-state Uhlmann fidelity.
  Uhlmann fidelity satisfies `F(ρ, ρ) = 1`, whereas this kernel has diagonal
  `Tr[ρ(x)²]`, the state's purity, which is below 1 for mixed states.
- **Readout noise applies only to `sampler` kernels.** A readout confusion channel
  acts at the measurement boundary, which analytical kernels do not use.
  Unsupported combinations are rejected on both engines.
- **The kernel knows the feature width it actually reads** through
  `kernel.feature_dimension`. Data with too many or too few columns is rejected
  by both engines.
- **The kernel path also runs capability preflight:** device registration,
  Aer method support, and GPU availability are checked before construction.
  `kernel.qforge_device` reports the device selected for execution.

Example backend configurations for `fidelity="auto"`:

```python
from qforge_ai import BackendSpec, Engine, NoiseSpec

sampled_backend = BackendSpec(
    engine=Engine.QISKIT,
    device="aer",
    shots=2048,
    seed=7,
    noise=NoiseSpec(name="depolarizing_noise", probability=0.05),
)
# -> fidelity_quantum_kernel[aer_sampler_v2]
# Kernel values reflect the specified noise.

ideal_backend = BackendSpec(engine=Engine.PENNYLANE, device="default.qubit")
# -> pennylane_fidelity_kernel[statevector]
# Can be used with the scikit-learn-compatible PennyLaneQSVC.

mixed_backend = BackendSpec(
    engine=Engine.PENNYLANE,
    device="default.mixed",
    noise=NoiseSpec(name="amplitude_damping", probability=0.2),
)
# -> hilbert_schmidt_kernel[pennylane_density_matrix]
# The corresponding Qiskit route is
# hilbert_schmidt_kernel[qiskit_density_matrix].
```

Analytical kernels evaluate one encoded state per sample: `n` circuit
evaluations for a training set of size `n`, rather than `n²` pair circuits.
The classical Gram-matrix computation still involves pairs of samples.
The documented analytical equivalence checks use a tolerance of `1e-9`;
sampled values are compared within shot noise.

`PennyLaneQSVC` implements the scikit-learn estimator interface, including
`get_params`, `set_params`, `clone`, `pickle`, `GridSearchCV`, `Pipeline`, and
`cross_val_score`. It exposes `n_features_in_`, raises `NotFittedError` before
fitting, and makes `predict_proba` available when the installed `SVC` supports
and enables `probability=True`. The estimator checks are compared with those
passed by `sklearn.svm.SVC`. Hyperparameter names are read from the installed
`SVC` rather than duplicated, accommodating API changes such as the
`probability` removal anticipated for scikit-learn 1.11 in the source comments.
The `[pennylane]` extra also installs scikit-learn.

The built-in kernel path rejects encodings restricted to 0/1 inputs
(basis/binary), as well as feature maps containing trainable weights.
Amplitude/state-preparation encoding works in the **analytical** PennyLane
kernel because that path does not invert the preparation circuit; it is not
available in the sampled compute–uncompute path.

## State preparation, ancillas, and wire ordering

In a `CircuitSpec`, `ancilla_qubits` adds reserved wires to the total IR register.
Built-in encoding and ansatz builders operate only on `ir.data_wires`.
Default measurements also use data wires, except for a full statevector.
Select an ancilla explicitly in `MeasurementSpec.wires` when it must be measured.

For amplitude encoding, `normalize=True` applies the same normalization on both
engines at the adapter boundary. This avoids differences caused by the native
libraries' normalization tolerances: a vector with norm `1 + 2.6e-7` previously
produced different prepared states. With `normalize=False`, inputs outside the
accepted unit-norm tolerance are rejected; the option is not silently changed.

Amplitude vectors and multi-qubit observable matrices in the IR use
**wire-major order**: `wires[0]` is the most significant bit, followed by
`wires[1]`, and so on. The Qiskit adapter handles conversion to its little-endian
`qargs`/`initialize` convention. Without this conversion in 1.2.2, the same vector
could prepare orthogonal states across engines, and an observable such as
`diag(0,1,2,3)` could read the two qubits in the opposite order.

**The same ordering applies to `counts` dictionaries.** Character `j` of a
bitstring corresponds to `measurement.wires[j]`. For example, when measuring
wires `(0, 1)`, a state with wire 0 equal to 0 and wire 1 equal to 1 has key
`"01"` on both engines. In 1.2.3, `sample` and `probability` were reordered but
raw Qiskit counts were still exposed in little-endian form, allowing bitstring
analyses and class mappings to change when switching engines. Both adapters now
return plain `dict[str, int]` counts in the shared order.

## Physical constraints in the IR

Rotation angles, parameter `scale`/`offset` values, `global_phase`, and observable
coefficients must be **finite real numbers**. In 1.2.3, `RX(1j)` passed IR
validation and was later rejected by Qiskit, while PennyLane could produce an
unphysical statevector with norm 1.24. Similarly,
`Observable("Z", coefficient=1j)` produced a non-Hermitian error in Qiskit
but an imaginary expectation value in PennyLane.

Complex amplitudes remain valid for `STATE_PREP`. Custom observable matrices may
also contain complex entries, provided the matrix is Hermitian and has the
correct dimension.

## Noise: one canonical definition for both engines

Each channel is defined **once** as Kraus operators in `qforge_ai.noise`.
Both adapters insert the same operators at the same circuit positions.
The Qiskit adapter uses in-circuit `kraus` instructions for this shared contract,
rather than relying on an Aer `NoiseModel` to infer gate placement. Analytical
noisy results can therefore be compared at floating-point precision.

| Channel | Meaning of `probability` or channel parameters |
| --- | --- |
| `bit_flip`, `phase_flip`, `bit_phase_flip` | Probability of applying the Pauli operator; correlated `P⊗P` on two wires |
| `depolarizing` | Probability of replacing the state with the maximally mixed state, using the Nielsen & Chuang / Qiskit convention |
| `amplitude_damping`, `phase_damping` | Damping parameter (`gamma` / `lambda`); independent on two wires |
| `thermal_relaxation` | Derived from `t1`, `t2`, and `gate_time`; valid throughout `t2 <= 2*t1` |
| `readout` | Symmetric confusion probability on each measured wire |

In the documented 1.2.2 regression, `depolarizing(p=0.3)` produced `-0.7` in
Qiskit and `-0.6` in PennyLane. Qiskit interpreted the parameter as complete
depolarization probability; PennyLane interpreted it as total Pauli-error
probability. The framework now declares and implements one convention.

`thermal_relaxation` is supported on both engines, whereas it was previously
Qiskit-only, because the channel is constructed independently of the engine
using NumPy.

**State preparation is noiseless under this framework's noise contract.**
`BASIS` and `STATE_PREP` define the initial state and do not receive gate noise.
This provides consistent semantics despite different decompositions: PennyLane
can use one `AmplitudeEmbedding` operation, whereas Qiskit's `initialize` expands
into a state-dependent number of elementary gates. In the documented 1.2.2
regression, `BASIS(0)` under `bit_flip(p=1)` produced `+1` in Qiskit and `-1` in
PennyLane.

## Training

Run end-to-end training with an NPZ dataset containing `X_train` and `y_train`,
and optionally `X_val` and `y_val`:

```bash
qforge train examples/configs/ids_hybrid.yaml dataset.npz --device cpu
```

Programmatic training, using previously created PyTorch data loaders:

```python
from qforge_ai import ExperimentRunner

runner = ExperimentRunner()
prepared = runner.prepare("examples/configs/ids_hybrid.yaml")
result = runner.train(prepared, train_loader, validation_loader)
```

See [Custom Circuit](#custom-circuit) for a complete small example that builds
the data loader and trains a model from a manually constructed IR.

## Catalog coverage: sections 1–21

The table describes catalog coverage. Check each entry's support level and
`catalog.is_runnable(key)` before treating it as an executable implementation.

| No. | Catalog family | Contents |
| ---: | --- | --- |
| 1 | Data encoding | Basis, angle, amplitude, phase, feature maps, graph, image, sequence, molecular, and QRAM encodings |
| 2 | Ansatz | Hardware-efficient and rotation ansatzes, plus the separate `entanglement` programmatic category |
| 3 | Physics and chemistry | UCC, Hartree–Fock, Hamiltonians, and time evolution |
| 4 | Tensor networks | MPS, TTN, MERA, QCNN, pooling, and hierarchical circuits |
| 5 | Dynamic circuits | Mid-circuit measurement, conditional gates, reset/reuse, and feed-forward |
| 6 | Measurement | Separate `measurement` and `observable` programmatic categories; core expectation, probability, sample, counts, variance, full statevector, density matrix, and reduced density matrix outputs |
| 7 | QNN | EstimatorQNN, SamplerQNN, VQC/VQR, and multi-output QNNs |
| 8 | Quantum vision | QCNN, quanvolution, quantum ViT, and patch models |
| 9 | Time series | QRNN, QLSTM, QGRU, reservoirs, and quantum transformers |
| 10 | Graph QML | QGNN, graph convolution/attention, and link prediction |
| 11 | Kernels | Fidelity, trainable/projected kernels, QSVC/QSVR, and QGP |
| 12 | Unsupervised learning | QKMeans, QPCA, autoencoders, and anomaly detection |
| 13 | Generative models | QGAN, Born machines, QVAE, QBM, flow, and diffusion |
| 14 | Reinforcement learning | Quantum policies, actor–critic, DQN, PPO, and multi-agent methods |
| 15 | Optimization | QAOA, VQE, QUBO, Ising, and domain-specific optimization |
| 16 | Fundamental algorithms | QFT, QPE, Grover, HHL, swap/Hadamard tests, and QSVT |
| 17 | Noise | Pauli, damping, relaxation, readout, crosstalk, and leakage |
| 18 | Error mitigation | ZNE, PEC, symmetry verification, twirling, and dynamical decoupling |
| 19 | Quantum error correction | Repetition, Shor, Steane, surface, and LDPC codes |
| 20 | Continuous variables | Gaussian circuits, squeezing, interferometers, and CV-QNNs |
| 21 | Model templates | Sequential, parallel, residual, ensemble, and domain templates |

The full list of names is in [`CATALOG_EN.md`](CATALOG_EN.md).
Programmatic access:

```python
from qforge_ai.catalog import catalog

all_items = catalog.list()
qnn_items = catalog.list(category="qnn_model")
sampler = catalog.get("sampler_qnn")
print(sampler.engines, sampler.status, sampler.capabilities)
```

The 21 documented families correspond to 23 programmatic categories: family 2
is split into `ansatz` and `entanglement`, and family 6 into `measurement` and
`observable`. The other families map one-to-one.

## Support levels

- `native`: a registered `factory` or a verified `builtin_path` is available.
- `composite`: constructed through an implemented `ModelFactory` template.
- `experimental`: the API and metadata are registered; a custom implementation
  may be required for the selected hardware or plugin.
- `descriptor`: a catalog entry for discovery, validation, and plugin development.

Descriptors carry `dependencies`, `references`, `implementation`, and section
metadata. A catalog entry alone does not promise executability;
`catalog.is_runnable(key)` reports the implementation status explicitly.

`CapabilityRegistry` checks operation, measurement, noise-channel, device, shot,
dynamic-circuit, gradient, and GPU compatibility before execution when invoked
through the experiment/preflight workflow. IR mode supports analysis and
serialization and produces a warning instead of executing a circuit. Qiskit GPU
support is checked at runtime against the installed Aer build's
`available_devices()` result.

## Design principles

- Optional dependencies do not break the base package at import time.
- Seeds, circuits, transpilation, shots, noise, and version information are recorded.
- Circuit execution and Qiskit/PennyLane QNN training consume the same
  `BackendSpec` contract for shots, noise, method, compute device, and seed.
  Option support depends on the primitive path, as shown below.
- Statevector, probability, and density-matrix results are normalized to
  wire-major order at the adapter boundary. Global phase is normalized only
  when comparing statevectors.
- Finite-shot probability, expectation, and variance results come from actual
  sampling. Comparison reports include maximum error, total variation distance
  (TVD), Hellinger distance, confidence intervals, and a confidence threshold.
- Dataset manifests store the content SHA-256 separately from the source path,
  so moving a dataset does not change its content identity.
- Reports include depth, two-qubit gates, shots, runtime, and memory alongside
  model performance.
- User registrations cannot silently overwrite built-in entries.
- Real-hardware drivers can be added through plugins.

### Qiskit option-consumption contract

| `BackendSpec` field | Supported path |
| --- | --- |
| `options.backend_options` | `AerSimulator` construction for direct runs and Aer-based QNN primitives |
| `options.run_options` | Direct Aer execution, analytical Aer `EstimatorV2`, and Aer `SamplerV2` |
| `options.primitive_options.abelian_grouping` | Finite-shot `BackendEstimatorV2` only |
| `options.gradient_options` | QNN `parameter_shift`, `finite_diff`, and `spsa` gradient construction |

Finite-shot `BackendEstimatorV2` does not accept general `run_options`.
Precision derived from `shots` and `BackendSpec.seed` are managed by the
framework. Unsupported options produce explicit errors instead of being ignored.

Qiskit QNN selection defaults to `qnn: auto`: expectation measurement selects
`EstimatorQNN`, and probability measurement selects `SamplerQNN`. An explicitly
selected incompatible QNN/measurement pair is rejected during construction.
A custom `SamplerQNN` `interpret` function requires `sampler_output_shape`,
which is propagated to the input width of the hybrid model's classical head.

A statevector describes the full system only. For subsystem output, use
`MeasurementSpec(name="reduced_density_matrix", wires=(...))`.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

Install the relevant optional engine extras to run their integration tests.

## License

Apache License 2.0. See [LICENSE](LICENSE).
