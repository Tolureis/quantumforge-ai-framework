# Complete Component Catalog: Sections 1–21

## Document Sections ↔ Programmatic Categories

The framework's 21 conceptual families are represented by 23 programmatic
categories for more precise filtering. Section 2 is split into `ansatz` and
`entanglement`; section 6 is split into `measurement` and `observable`.
The other sections map one-to-one.

| Section | Programmatic category |
| ---: | --- |
| 1 | `encoding` |
| 2 | `ansatz`, `entanglement` |
| 3 | `physics_chemistry` |
| 4 | `tensor_network` |
| 5 | `dynamic_circuit` |
| 6 | `measurement`, `observable` |
| 7–21 | Respectively: `qnn_model`, `vision_model`, `sequence_model`, `graph_model`, `kernel`, `unsupervised_model`, `generative_model`, `reinforcement_model`, `optimization`, `algorithm`, `noise`, `error_mitigation`, `error_correction`, `continuous_variable`, `model_template` |

`native` means a verified factory/built-in path is available; `composite` means
an implemented ModelFactory route is available. Other entries are discovery
or plugin contracts.

This document describes the framework's conceptual scope. For the
machine-readable, up-to-date list, use `qforge catalog --json` or
`qforge_ai.catalog.catalog`. Each entry carries dependency, reference,
implementation, and document-section metadata.

### What the Catalog Counts Mean

The catalog contains <!-- qforge:count total -->382<!-- /qforge:count --> entries
in total. Of these, <!-- qforge:count executable -->93<!-- /qforge:count --> are
**executable** through the core;
<!-- qforge:count discovery_only -->289<!-- /qforge:count --> are
**discovery/plugin contracts** (`descriptor` or `experimental`): their APIs
and metadata are registered, but execution requires a plugin.

The total count does not represent implementation capacity; the executable
count is the relevant measure. These figures are not maintained manually:
they are derived from the code through `qforge_ai.catalog.capability_report()`,
and `scripts/check_catalog_docs.py` requires them to match this document in CI.
`tests/test_catalog.py` also verifies that every entry marked as executable
can actually be resolved.

For a breakdown by category:

```bash
python -c "import json,qforge_ai.catalog as c; print(json.dumps(c.capability_report(), indent=2))"
```

## 1. Data Encoding Circuits

Basic encodings: Basis, Binary, One Hot, Angle, Rotation, Phase, Amplitude,
Complex Amplitude, Probability, Dense Angle, and Superposition Encoding.

Feature maps: Z, ZZ, Pauli, IQP, Hamiltonian, Fourier, Chebyshev, Polynomial,
Kernel, Trainable, Data Re-uploading, and Higher-order Feature Map.

Domain encodings: Image Patch, Quanvolution, Time Series, Sequence, Token,
Graph Node/Edge, Adjacency Matrix, Molecular, Fermionic, Portfolio, QUBO,
State Preparation, Oracle, and experimental QRAM Encoding.

Angle/rotation encoding has a low preparation cost. Amplitude encoding uses
few qubits, but general state preparation may have an exponential gate cost.
Data re-uploading produces an encoding block followed by a single ansatz layer
in each cycle. The framework raises an error when
`feature_dimension > num_qubits * repeats` instead of losing data, and requires
`repeats == ansatz.layers`. When multiple features share a wire, the rotation
axis cycles through X → Y → Z. Otherwise, rotations around the same axis would
add together and cause different feature vectors to collapse to the same state
(see the README discussion of data re-uploading).

### Encoding Capabilities Are Declared Per Entry

Through version 1.2.3, capability labels were assigned **per section**, so all
38 encoding entries claimed to be `differentiable`. This was incorrect even for
entries implemented by the core: a basis-state encoding is a step function of
its input bit and is not differentiable on either engine, while superposition
encoding takes no input at all. Since 1.2.4, labels are assigned per entry and
include three additional labels that determine composition:

| Label | Meaning |
| --- | --- |
| `data_encoding` | Loads input features into the circuit |
| `differentiable` | Allows gradients to flow with respect to input parameters |
| `symbolic_parameters` | Can be compiled with unbound symbolic parameters, allowing use in Qiskit `EstimatorQNN`/`SamplerQNN` |
| `continuous_input` | Accepts arbitrary real values, allowing use after the classical encoder of a hybrid template |
| `state_preparation` | Constructs the initial state instead of loading data into an existing state |

| Entry | `differentiable` | `symbolic_parameters` | `continuous_input` |
| --- | :--: | :--: | :--: |
| angle / rotation / phase / dense angle / Z, ZZ, Pauli, IQP, trainable, data re-uploading | ✅ | ✅ | ✅ |
| amplitude / complex amplitude / probability / state preparation | ✅ | ❌ | ✅ |
| basis / binary / one-hot / oracle | ❌ | ❌ | ❌ |
| superposition | ❌ (does not load data) | — | — |

These labels are operational: `qforge_ai.composition` derives the same
predicates from the IR, and `ModelFactory.build` explicitly rejects unsupported
combinations at construction time, before they fail inside an engine.
`tests/test_catalog_verification.py` actually builds and runs the
encoding × template × engine cross-product to verify that every combination
advertised as supported works and every combination advertised as unsupported
is rejected.

## 2. Parameterized Circuits and Ansatzes/Topologies

Real Amplitudes, Efficient SU2, Two/N Local, Hardware Efficient, Alternating
Layer, Strongly Entangling, Basic Entangler, Random Layers, Simplified Two
Design, Excitation/Symmetry Preserving; RX/RY/RZ, XYZ, Euler, U3, Controlled
Rotation, Ising, and Pauli Rotation Ansatz.

Scientific names implemented through another core template are not silently
treated as synonyms. For example, `ising_ansatz`, `pauli_rotation_ansatz`,
`random_layers`, `quantum_convolution`, and `quantum_pooling` carry
`approximation=true` and `alias_of=<actual factory>` metadata. In contrast,
`u3_ansatz` produces a genuine three-parameter `U` gate.

Topologies: Linear, Reverse Linear, Circular/Ring, Full, Pairwise,
Alternating, Star, Grid, Random, Hardware Topology, and User Defined.

`AnsatzSpec.entanglement` accepts both the core name (`linear`) and the catalog
key (`linear_entanglement`), allowing names from `qforge catalog` output to be
copied directly into a configuration.

## 3. Physics and Chemistry Circuits

UCCSD, UCC Singles/Doubles, k-UpCCGSD, Gate Fabric, Particle Conserving,
Fermionic Excitation, Bogoliubov, Hartree–Fock, Hamiltonian Variational,
Trotter/Suzuki–Trotter Evolution, Time Evolution, ADAPT-VQE, Unitary Coupled
Cluster, Quantum Chemistry, Lattice, Heisenberg, and Hubbard Ansatz.

For these families, fermion-to-qubit mapping, active-space selection, and
Hamiltonian construction are provided by a domain plugin.

## 4. Tensor Networks and Hierarchical Circuits

Matrix Product State Circuit, Tree Tensor Network, Tensor Tree Classifier,
MERA, Hierarchical Quantum Circuit, QCNN, Quantum Convolution/Pooling,
Residual Block, Bottleneck, Patch Circuit, Tensor Ring, and Tensor Network
Classifier.

## 5. Dynamic and Adaptive Circuits

Dynamic Circuit, Mid-circuit Measurement, Conditional Gate, Measurement
Feedback, Reset-and-Reuse, Ancilla Assisted, Adaptive Ansatz/Measurement,
Post-selection, Repeat-until-success, Teleportation Based, Classically
Controlled, and Feed-forward Circuit.

Support must be checked during capability preflight together with the device,
shot count, and mid-circuit measurement method.

## 6. Measurement, Observables, and Readout

Computational Basis, Probability, Sample, Counts, Expectation, Variance,
Statevector, Density/Reduced Density Matrix, Entropy, Mutual Information,
Classical Shadow, and Tomography measurements.

In the core, statevector output is supported only for the full system.
`density_matrix` with a subset of wires and the backward-compatible
`reduced_density_matrix_measurement` (`rdm`) use the same native
reduced-density-matrix execution path.

Observables: Pauli X/Y/Z, Identity, Hadamard, tensor product, Pauli/Sparse
Hamiltonian, Hermitian, Projector, custom, and trainable observables.

Readout: single/multi-qubit, parity, probability/expectation vector, bitstring
class mapping, weighted/trainable, and pooling readout.

## 7. QNNs and Hybrid Models

EstimatorQNN, SamplerQNN, VQC, VQR, quantum-only classifier/regressor,
multi-class/label/output QNN, quantum perceptron/feed-forward/deep VQN;
classical→quantum, quantum→classical, classical→quantum→classical,
multi/parallel/residual quantum layer, bottleneck, mixture-of-experts,
classical ensemble, and Qiskit–PennyLane ensemble.

## 8. Quantum Vision Models

QCNN, Quanvolutional NN, Hybrid CNN-QNN, Image/Patch Classifier, Pooling
Network, Residual CNN, Multi-scale QCNN, Feature Pyramid, Quantum Vision
Transformer, and Hybrid ViT-QNN.

## 9. Time-Series and Sequence Models

QRNN, QLSTM, QGRU, Quantum Reservoir/Echo State, Temporal Convolution,
Sequence Classifier/Regressor, Time-series Forecaster, Attention, Transformer,
Hybrid Transformer-QNN, Memory Network, and Quantum State-space Model.

## 10. Graph Quantum ML

QGNN, Quantum Graph Convolution/Attention, Message Passing, Graph/Node
Classifier, Link Predictor, Graph Autoencoder, Molecular Graph Network, and
Hybrid GNN-QNN.

## 11. Quantum Kernel Models

Fidelity, Statevector, Trainable, Projected, RBF, Polynomial, Alignment,
Graph, Sequence, and Noise-aware Kernel; QSVC, QSVR, Kernel Ridge/Logistic,
Kernel PCA/Clustering, Quantum Gaussian Process, One-class SVM, and Kernel
Anomaly Detector.

Kernel entries **routed by the core**, namely `fidelity_quantum_kernel`,
`statevector_quantum_kernel`, and `qsvc`, work on both engines. Since 1.2.5,
the kernel itself has **one implementation** (`qforge_ai.models.kernel`):
the framework defines the quantity computed, how the Gram matrix is
constructed, which inputs are valid, and where noise is placed.
Engines provide only two primitives: the encoded state of one sample and the
measured overlap of a pair. Analytical values agree across engines within
`1e-9`; sampled values agree within shot noise. The other kernel families
listed above remain discovery-oriented descriptor entries.

The kernel contract is resolved in one place
(`qforge_ai.models.kernel_contract`). The `fidelity` value (`auto`, `statevector`,
`hilbert_schmidt`/`density_matrix`, `sampler`), the list of settings that the
selected kernel cannot apply, and the rule requiring explicit shots for a
sampled kernel are identical on both engines. Every resolved kind works on
both engines. For the detailed table, see the README section on quantum-kernel
execution.

Under noise, the analytical kernel computes `Tr[ρ(x) ρ(y)]`, the
**Hilbert–Schmidt inner product**. As the inner product of two encoded states,
it is symmetric and positive semidefinite by construction. It is not
mixed-state Uhlmann fidelity: true fidelity satisfies `F(ρ, ρ) = 1`, whereas
this kernel has diagonal `Tr[ρ²]`, the state's purity. In the sampled kernel,
every Gram entry, including the diagonal, is measured, and the matrix is
symmetrized and projected onto the PSD cone on both engines. Readout noise is
also applied only on this path, at the measurement boundary. The kernel knows
the number of columns read by its feature map (`kernel.feature_dimension`)
and rejects data with a mismatched width.

Encodings restricted to 0/1 inputs (basis/binary) cannot be used in the kernel
because the kernel evaluates its feature map at arbitrary sample values.
A feature map containing trainable weights cannot define a fixed kernel either.
Amplitude/state-preparation encoding works in the analytical PennyLane kernel,
but not in the sampled compute–uncompute path, where the circuit's inverse is
not defined.

## 12. Unsupervised Models

Quantum K-Means, Hierarchical/Spectral/Kernel Clustering, QPCA/VQPCA,
Autoencoder, Denoising/Sparse Autoencoder, Density Estimator, Anomaly
Detector, One-class Classifier, Isolation Model, and Self-organizing Map.

## 13. Generative Models

QGAN, Hybrid/Conditional/Wasserstein QGAN, Quantum/Circuit Born Machine,
QVAE, Quantum/Restricted Boltzmann Machine, Normalizing Flow, Diffusion,
Energy-based/Generative Model, and Sequence/Image Generator.

## 14. Quantum Reinforcement Learning

Variational Quantum Policy, Policy Gradient, Actor–Critic, Quantum Q Network,
Hybrid DQN-QNN, Quantum PPO/A2C/SAC, Multi-agent RL, Contextual Bandit,
Reinforcement Agent, and QAOA Policy Optimizer.

## 15. Optimization Models

QAOA, Recursive/Warm-start/Multi-angle/Constraint-preserving QAOA; VQE,
Subspace-search VQE, Variational Deflation, Grover Adaptive Search, Minimum
Finding, Quantum Annealing, QUBO/Ising, Portfolio, Scheduling, Routing, and
Feature-selection QAOA.

## 16. Fundamental Quantum Algorithms

QFT/Inverse QFT, QPE, Amplitude/Iterative Estimation, Grover, Amplitude
Amplification, HHL, Swap/Hadamard Test, Distance/Inner-product/Fidelity
Estimation, Counting, Walk/Random Walk, QSVT, Block Encoding, LCU, Quantum
Signal Processing, and State/Process Tomography.

## 17. Noise Circuits

Bit/Phase/Bit-phase Flip, Pauli, Depolarizing, Amplitude/Generalized Amplitude
Damping, Phase Damping, Thermal Relaxation, Readout/Reset/Coherent Error,
Kraus, Mixed-unitary, Crosstalk, Leakage, Shot, and Custom Noise.

Noise can be applied at the gate, qubit, layer, time, hardware-topology, or
training-epoch level.

## 18. Error Mitigation

Readout Mitigation, Zero-noise Extrapolation, Probabilistic Error
Cancellation, Clifford Data Regression, Virtual Distillation, Symmetry
Verification, Dynamical Decoupling, Pauli Twirling, Randomized Compiling,
Measurement Mitigation, Shot-frugal Estimation, Error-aware, and Noise-adaptive
Training.

## 19. Quantum Error Correction

Three-qubit Bit/Phase-flip, Shor, Steane, Repetition, Stabilizer, Surface,
Toric, Color, Bacon–Shor, and Quantum LDPC Code; Syndrome Measurement, Logical
Qubit Circuit, and Fault-tolerant Gate Simulation.

## 20. Continuous-Variable/Photonic Models

Gaussian Circuit, Displacement/Squeezing/Phase-space Encoding, Beam-splitter
Network, Interferometer, Kerr Circuit, CV-QNN, Photonic Classifier/Regressor,
Gaussian Born Machine, and CV Autoencoder. These are specific to PennyLane
or photonic-device plugins and cannot be translated directly into the Qiskit
qubit adapter.

## 21. Framework Model Templates

Quantum Classifier, Regressor, Kernel Model, Hybrid Model, Sequential,
Parallel, Residual, Ensemble, Autoencoder, Generative Model, Anomaly Detector,
Optimizer, Reinforcement Agent, and IDS Quantum Bottleneck.

Executable core routes: Quantum Classifier/Regressor, Quantum Kernel Model,
Hybrid Model, Sequential, Parallel, Residual, Ensemble, and IDS Quantum
Bottleneck. Autoencoder, Generative, Anomaly Detector, Optimizer, and
Reinforcement Agent entries have an explicit `experimental/plugin_required`
status. An unknown or unimplemented `ModelSpec.template` is not silently
ignored; it raises `CapabilityError` during construction.
