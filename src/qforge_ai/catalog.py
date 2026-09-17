"""Built-in catalog covering the 21 QuantumForge component families.

The catalog is intentionally metadata-first: every documented research family is
discoverable even when it needs a vendor plugin, photonic device or domain package.
Runnable core builders are attached by :mod:`qforge_ai.circuits` and model adapters.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .registry import ComponentDescriptor, ComponentRegistry, normalize_key
from .specs import ComponentStatus, Engine

QISKIT_PL = frozenset({Engine.IR, Engine.QISKIT, Engine.PENNYLANE})
PL_ONLY = frozenset({Engine.IR, Engine.PENNYLANE})
QISKIT_ONLY = frozenset({Engine.IR, Engine.QISKIT})
IR_ONLY = frozenset({Engine.IR})


# category, default status, engines, capabilities, names
_SECTIONS: tuple[
    tuple[str, ComponentStatus, frozenset[Engine], frozenset[str], tuple[str, ...]], ...
] = (
    (
        "encoding",
        ComponentStatus.DESCRIPTOR,
        QISKIT_PL,
        frozenset({"data_encoding", "differentiable"}),
        (
            "Basis Encoding",
            "Binary Encoding",
            "One Hot Encoding",
            "Angle Encoding",
            "Rotation Encoding",
            "Phase Encoding",
            "Amplitude Encoding",
            "Complex Amplitude Encoding",
            "Probability Encoding",
            "Dense Angle Encoding",
            "Superposition Encoding",
            "Z Feature Map",
            "ZZ Feature Map",
            "Pauli Feature Map",
            "IQP Feature Map",
            "Hamiltonian Feature Map",
            "Fourier Feature Map",
            "Chebyshev Feature Map",
            "Polynomial Feature Map",
            "Kernel Feature Map",
            "Trainable Feature Map",
            "Data Reuploading Feature Map",
            "Higher Order Feature Map",
            "Image Patch Encoding",
            "Quanvolution Encoding",
            "Time Series Encoding",
            "Sequence Encoding",
            "Token Encoding",
            "Graph Node Encoding",
            "Graph Edge Encoding",
            "Adjacency Matrix Encoding",
            "Molecular Encoding",
            "Fermionic Encoding",
            "Portfolio Encoding",
            "QUBO Encoding",
            "State Preparation Encoding",
            "Oracle Encoding",
            "QRAM Encoding",
        ),
    ),
    (
        "ansatz",
        ComponentStatus.DESCRIPTOR,
        QISKIT_PL,
        frozenset({"variational", "differentiable"}),
        (
            "Real Amplitudes",
            "Efficient SU2",
            "Two Local",
            "N Local",
            "Hardware Efficient Ansatz",
            "Alternating Layer Ansatz",
            "Strongly Entangling Layers",
            "Basic Entangler Layers",
            "Random Layers",
            "Simplified Two Design",
            "Excitation Preserving Ansatz",
            "Symmetry Preserving Ansatz",
            "RX Ansatz",
            "RY Ansatz",
            "RZ Ansatz",
            "XYZ Rotation Ansatz",
            "Euler Rotation Ansatz",
            "U3 Ansatz",
            "Controlled Rotation Ansatz",
            "Ising Ansatz",
            "Pauli Rotation Ansatz",
        ),
    ),
    (
        "entanglement",
        ComponentStatus.NATIVE,
        QISKIT_PL,
        frozenset({"topology"}),
        (
            "Linear Entanglement",
            "Reverse Linear Entanglement",
            "Circular Entanglement",
            "Full Entanglement",
            "Pairwise Entanglement",
            "Alternating Entanglement",
            "Star Entanglement",
            "Grid Entanglement",
            "Ring Entanglement",
            "Random Entanglement",
            "Hardware Topology Entanglement",
            "User Defined Entanglement",
        ),
    ),
    (
        "physics_chemistry",
        ComponentStatus.EXPERIMENTAL,
        QISKIT_PL,
        frozenset({"domain", "hamiltonian"}),
        (
            "UCCSD",
            "UCC Singles",
            "UCC Doubles",
            "k UpCCGSD",
            "Gate Fabric",
            "Particle Conserving Ansatz",
            "Fermionic Excitation Ansatz",
            "Bogoliubov Ansatz",
            "Hartree Fock State",
            "Hamiltonian Variational Ansatz",
            "Trotterized Evolution",
            "Suzuki Trotter Evolution",
            "Time Evolution Ansatz",
            "ADAPT VQE Ansatz",
            "Unitary Coupled Cluster",
            "Quantum Chemistry Ansatz",
            "Lattice Model Ansatz",
            "Heisenberg Ansatz",
            "Hubbard Model Ansatz",
        ),
    ),
    (
        "tensor_network",
        ComponentStatus.EXPERIMENTAL,
        QISKIT_PL,
        frozenset({"tensor_network", "hierarchical"}),
        (
            "Matrix Product State Circuit",
            "Tree Tensor Network",
            "Tensor Tree Classifier",
            "MERA",
            "Hierarchical Quantum Circuit",
            "QCNN",
            "Quantum Convolution",
            "Quantum Pooling",
            "Quantum Residual Block",
            "Quantum Bottleneck",
            "Quantum Patch Circuit",
            "Tensor Ring Circuit",
            "Tensor Network Classifier",
        ),
    ),
    (
        "dynamic_circuit",
        ComponentStatus.EXPERIMENTAL,
        QISKIT_PL,
        frozenset({"dynamic", "mid_circuit_measurement"}),
        (
            "Dynamic Quantum Circuit",
            "Mid Circuit Measurement Circuit",
            "Conditional Gate Circuit",
            "Measurement Feedback Circuit",
            "Reset And Reuse Circuit",
            "Ancilla Assisted Circuit",
            "Adaptive Ansatz",
            "Adaptive Measurement Circuit",
            "Post Selection Circuit",
            "Repeat Until Success Circuit",
            "Teleportation Based Circuit",
            "Classically Controlled Circuit",
            "Feed Forward Circuit",
        ),
    ),
    (
        "measurement",
        ComponentStatus.DESCRIPTOR,
        QISKIT_PL,
        frozenset({"measurement"}),
        (
            "Computational Basis Measurement",
            "Probability Measurement",
            "Sample Measurement",
            "Count Measurement",
            "Expectation Measurement",
            "Variance Measurement",
            "Statevector Measurement",
            "Density Matrix Measurement",
            "Reduced Density Matrix Measurement",
            "Entropy Measurement",
            "Mutual Information Measurement",
            "Shadow Measurement",
            "Tomography Measurement",
            "Single Qubit Readout",
            "Multi Qubit Readout",
            "Parity Readout",
            "Probability Vector Readout",
            "Expectation Vector Readout",
            "Bitstring Class Mapping",
            "Observable Weighted Readout",
            "Trainable Readout",
            "Quantum Pooling Readout",
        ),
    ),
    (
        "observable",
        ComponentStatus.NATIVE,
        QISKIT_PL,
        frozenset({"measurement", "observable"}),
        (
            "Pauli X",
            "Pauli Y",
            "Pauli Z",
            "Identity",
            "Hadamard Observable",
            "Tensor Product Observable",
            "Pauli Hamiltonian",
            "Hermitian Observable",
            "Projector",
            "Sparse Hamiltonian",
            "Custom Observable",
            "Trainable Observable",
        ),
    ),
    (
        "qnn_model",
        ComponentStatus.COMPOSITE,
        QISKIT_PL,
        frozenset({"supervised", "variational"}),
        (
            "Estimator QNN",
            "Sampler QNN",
            "Variational Quantum Classifier",
            "Variational Quantum Regressor",
            "Quantum Only Classifier",
            "Quantum Only Regressor",
            "Multi Class QNN",
            "Multi Label QNN",
            "Multi Output QNN",
            "Quantum Perceptron",
            "Quantum Feed Forward Network",
            "Deep Variational Quantum Network",
            "Classical Quantum Classifier",
            "Classical Quantum Regressor",
            "Quantum Classical Classifier",
            "Quantum Classical Regressor",
            "Classical Quantum Classical Network",
            "Multi Quantum Layer Network",
            "Parallel Quantum Layer Network",
            "Residual Hybrid QNN",
            "Quantum Bottleneck Network",
            "Quantum Mixture Of Experts",
            "Quantum Classical Ensemble",
            "Qiskit PennyLane Ensemble",
        ),
    ),
    (
        "vision_model",
        ComponentStatus.COMPOSITE,
        QISKIT_PL,
        frozenset({"vision", "supervised"}),
        (
            "Quantum Convolutional Neural Network",
            "Quanvolutional Neural Network",
            "Hybrid CNN QNN",
            "Quantum Image Classifier",
            "Quantum Patch Classifier",
            "Quantum Pooling Network",
            "Quantum Residual CNN",
            "Multi Scale QCNN",
            "Quantum Feature Pyramid",
            "Quantum Vision Transformer",
            "Hybrid Vision Transformer QNN",
        ),
    ),
    (
        "sequence_model",
        ComponentStatus.COMPOSITE,
        QISKIT_PL,
        frozenset({"sequence", "time_series"}),
        (
            "Quantum Recurrent Neural Network",
            "Quantum LSTM",
            "Quantum GRU",
            "Quantum Reservoir Computing",
            "Quantum Echo State Network",
            "Quantum Temporal Convolution",
            "Quantum Sequence Classifier",
            "Quantum Sequence Regressor",
            "Quantum Time Series Forecaster",
            "Quantum Attention",
            "Quantum Transformer",
            "Hybrid Transformer QNN",
            "Quantum Memory Network",
            "Quantum State Space Model",
        ),
    ),
    (
        "graph_model",
        ComponentStatus.COMPOSITE,
        QISKIT_PL,
        frozenset({"graph", "supervised"}),
        (
            "Quantum Graph Neural Network",
            "Quantum Graph Convolution",
            "Quantum Graph Attention",
            "Quantum Message Passing Network",
            "Quantum Graph Classifier",
            "Quantum Node Classifier",
            "Quantum Link Predictor",
            "Quantum Graph Autoencoder",
            "Quantum Molecular Graph Network",
            "Hybrid GNN QNN",
        ),
    ),
    (
        "kernel",
        ComponentStatus.COMPOSITE,
        QISKIT_PL,
        frozenset({"kernel", "supervised"}),
        (
            "Fidelity Quantum Kernel",
            "Statevector Quantum Kernel",
            "Trainable Quantum Kernel",
            "Projected Quantum Kernel",
            "Quantum RBF Kernel",
            "Quantum Polynomial Kernel",
            "Quantum Alignment Kernel",
            "Quantum Graph Kernel",
            "Quantum Sequence Kernel",
            "Noise Aware Quantum Kernel",
            "QSVC",
            "QSVR",
            "Quantum Kernel Ridge",
            "Quantum Kernel Logistic Regression",
            "Quantum Kernel PCA",
            "Quantum Kernel Clustering",
            "Quantum Gaussian Process",
            "Quantum One Class SVM",
            "Quantum Kernel Anomaly Detector",
        ),
    ),
    (
        "unsupervised_model",
        ComponentStatus.EXPERIMENTAL,
        QISKIT_PL,
        frozenset({"unsupervised"}),
        (
            "Quantum K Means",
            "Quantum Hierarchical Clustering",
            "Quantum Spectral Clustering",
            "Quantum Kernel Clustering Model",
            "Quantum PCA",
            "Variational Quantum PCA",
            "Quantum Autoencoder",
            "Quantum Denoising Autoencoder",
            "Quantum Sparse Autoencoder",
            "Quantum Clustering Circuit",
            "Quantum Density Estimator",
            "Quantum Anomaly Detector",
            "Quantum One Class Classifier",
            "Quantum Isolation Model",
            "Quantum Self Organizing Map",
        ),
    ),
    (
        "generative_model",
        ComponentStatus.EXPERIMENTAL,
        QISKIT_PL,
        frozenset({"generative"}),
        (
            "Quantum GAN",
            "Hybrid Quantum GAN",
            "Quantum Conditional GAN",
            "Quantum Wasserstein GAN",
            "Quantum Born Machine",
            "Circuit Born Machine",
            "Quantum Variational Autoencoder",
            "Quantum Boltzmann Machine",
            "Quantum Restricted Boltzmann Machine",
            "Quantum Normalizing Flow",
            "Quantum Diffusion Model",
            "Quantum Energy Based Model",
            "Quantum Generative Model",
            "Quantum Sequence Generator",
            "Quantum Image Generator",
        ),
    ),
    (
        "reinforcement_model",
        ComponentStatus.EXPERIMENTAL,
        QISKIT_PL,
        frozenset({"reinforcement_learning"}),
        (
            "Variational Quantum Policy",
            "Quantum Policy Gradient",
            "Quantum Actor Critic",
            "Quantum Q Network",
            "Hybrid DQN QNN",
            "Quantum PPO",
            "Quantum A2C",
            "Quantum SAC",
            "Quantum Multi Agent RL",
            "Quantum Contextual Bandit",
            "Quantum Reinforcement Learning Agent",
            "QAOA Policy Optimizer",
        ),
    ),
    (
        "optimization",
        ComponentStatus.EXPERIMENTAL,
        QISKIT_PL,
        frozenset({"optimization", "variational"}),
        (
            "QAOA",
            "Recursive QAOA",
            "Warm Start QAOA",
            "Multi Angle QAOA",
            "Constraint Preserving QAOA",
            "VQE",
            "Subspace Search VQE",
            "Variational Quantum Deflation",
            "Grover Adaptive Search",
            "Quantum Minimum Finding",
            "Quantum Annealing Model",
            "QUBO Model",
            "Ising Optimization Model",
            "Portfolio Optimization Model",
            "Scheduling Optimization Model",
            "Routing Optimization Model",
            "Feature Selection QAOA",
        ),
    ),
    (
        "algorithm",
        ComponentStatus.EXPERIMENTAL,
        QISKIT_PL,
        frozenset({"algorithm"}),
        (
            "Quantum Fourier Transform",
            "Inverse QFT",
            "Quantum Phase Estimation",
            "Amplitude Estimation",
            "Iterative Amplitude Estimation",
            "Grover Search",
            "Amplitude Amplification",
            "HHL",
            "Swap Test",
            "Hadamard Test",
            "Quantum Distance Estimation",
            "Quantum Inner Product",
            "Quantum Fidelity Estimation",
            "Quantum Counting",
            "Quantum Walk",
            "Quantum Random Walk",
            "Quantum Singular Value Transformation",
            "Block Encoding",
            "Linear Combination Of Unitaries",
            "Quantum Signal Processing",
            "Quantum State Tomography",
            "Quantum Process Tomography",
        ),
    ),
    (
        "noise",
        ComponentStatus.DESCRIPTOR,
        QISKIT_PL,
        frozenset({"noise"}),
        (
            "No Noise",
            "Bit Flip Noise",
            "Phase Flip Noise",
            "Bit Phase Flip Noise",
            "Pauli Noise",
            "Depolarizing Noise",
            "Amplitude Damping",
            "Generalized Amplitude Damping",
            "Phase Damping",
            "Thermal Relaxation",
            "Readout Error",
            "Reset Error",
            "Coherent Error",
            "Kraus Channel",
            "Mixed Unitary Noise",
            "Crosstalk Noise",
            "Leakage Noise",
            "Shot Noise",
            "Custom Noise Channel",
        ),
    ),
    (
        "error_mitigation",
        ComponentStatus.EXPERIMENTAL,
        QISKIT_PL,
        frozenset({"error_mitigation"}),
        (
            "Readout Error Mitigation",
            "Zero Noise Extrapolation",
            "Probabilistic Error Cancellation",
            "Clifford Data Regression",
            "Virtual Distillation",
            "Symmetry Verification",
            "Dynamical Decoupling",
            "Pauli Twirling",
            "Randomized Compiling",
            "Measurement Mitigation",
            "Shot Frugal Estimation",
            "Error Aware Training",
            "Noise Adaptive Training",
        ),
    ),
    (
        "error_correction",
        ComponentStatus.EXPERIMENTAL,
        QISKIT_PL,
        frozenset({"error_correction"}),
        (
            "Three Qubit Bit Flip Code",
            "Three Qubit Phase Flip Code",
            "Shor Code",
            "Steane Code",
            "Repetition Code",
            "Stabilizer Code",
            "Surface Code",
            "Toric Code",
            "Color Code",
            "Bacon Shor Code",
            "Quantum LDPC Code",
            "Syndrome Measurement",
            "Logical Qubit Circuit",
            "Fault Tolerant Gate Simulation",
        ),
    ),
    (
        "continuous_variable",
        ComponentStatus.EXPERIMENTAL,
        PL_ONLY,
        frozenset({"continuous_variable", "photonic"}),
        (
            "Gaussian Circuit",
            "Displacement Encoding",
            "Squeezing Encoding",
            "Phase Space Encoding",
            "Beam Splitter Network",
            "Interferometer Circuit",
            "Kerr Circuit",
            "Continuous Variable QNN",
            "Photonic Classifier",
            "Photonic Regressor",
            "CV Quantum Neural Network",
            "Gaussian Born Machine",
            "Continuous Variable Autoencoder",
        ),
    ),
    (
        "model_template",
        ComponentStatus.COMPOSITE,
        QISKIT_PL,
        frozenset({"template"}),
        (
            "Quantum Classifier",
            "Quantum Regressor",
            "Quantum Kernel Model",
            "Hybrid Quantum Model",
            "Quantum Sequential",
            "Quantum Parallel",
            "Quantum Residual",
            "Quantum Ensemble",
            "Quantum Autoencoder Template",
            "Quantum Generative Model Template",
            "Quantum Anomaly Detector Template",
            "Quantum Optimizer",
            "Quantum Reinforcement Agent",
            "IDS Quantum Bottleneck",
        ),
    ),
)


_NATIVE_KEYS = {
    "basis_encoding",
    "binary_encoding",
    "angle_encoding",
    "rotation_encoding",
    "phase_encoding",
    "amplitude_encoding",
    "dense_angle_encoding",
    "superposition_encoding",
    "z_feature_map",
    "zz_feature_map",
    "pauli_feature_map",
    "iqp_feature_map",
    "trainable_feature_map",
    "data_reuploading_feature_map",
    "state_preparation_encoding",
    "real_amplitudes",
    "efficient_su2",
    "two_local",
    "n_local",
    "hardware_efficient_ansatz",
    "alternating_layer_ansatz",
    "strongly_entangling_layers",
    "basic_entangler_layers",
    "random_layers",
    "rx_ansatz",
    "ry_ansatz",
    "rz_ansatz",
    "xyz_rotation_ansatz",
    "euler_rotation_ansatz",
    "u3_ansatz",
    "ising_ansatz",
    "pauli_rotation_ansatz",
    "qcnn",
    "quantum_convolution",
    "quantum_pooling",
    "expectation_measurement",
    "probability_measurement",
    "sample_measurement",
    "count_measurement",
    "variance_measurement",
    "statevector_measurement",
    "density_matrix_measurement",
    "reduced_density_matrix_measurement",
    "computational_basis_measurement",
    "no_noise",
    "bit_flip_noise",
    "phase_flip_noise",
    "bit_phase_flip_noise",
    "depolarizing_noise",
    "amplitude_damping",
    "phase_damping",
    "thermal_relaxation",
    "readout_error",
    "shot_noise",
}

_ALIASES = {
    "angle_encoding": ("angle",),
    "basis_encoding": ("basis",),
    "amplitude_encoding": ("amplitude",),
    "phase_encoding": ("phase",),
    "dense_angle_encoding": ("dense_angle",),
    "data_reuploading_feature_map": ("data_reuploading", "data_reuploading_encoding"),
    "hardware_efficient_ansatz": ("hardware_efficient", "hea"),
    "strongly_entangling_layers": ("strongly_entangling",),
    "basic_entangler_layers": ("basic_entangler",),
    "expectation_measurement": ("expectation", "expval"),
    "probability_measurement": ("probability", "probs"),
    "sample_measurement": ("sample",),
    "statevector_measurement": ("statevector", "state"),
    "density_matrix_measurement": ("density_matrix",),
    "reduced_density_matrix_measurement": ("reduced_density_matrix", "rdm"),
    "estimator_qnn": ("estimatorqnn",),
    "sampler_qnn": ("samplerqnn",),
    "variational_quantum_classifier": ("vqc",),
    "variational_quantum_regressor": ("vqr",),
    "quantum_convolutional_neural_network": ("qcnn_model",),
    "no_noise": ("none",),
}


_COMPOSITE_KEYS = {
    "estimator_qnn",
    "sampler_qnn",
    "variational_quantum_classifier",
    "variational_quantum_regressor",
    "classical_quantum_classical_network",
    "parallel_quantum_layer_network",
    "residual_hybrid_qnn",
    "quantum_bottleneck_network",
    "fidelity_quantum_kernel",
    "statevector_quantum_kernel",
    "qsvc",
    "quantum_classifier",
    "quantum_regressor",
    "quantum_kernel_model",
    "hybrid_quantum_model",
    "quantum_sequential",
    "quantum_parallel",
    "quantum_residual",
    "quantum_ensemble",
    "ids_quantum_bottleneck",
}

_NATIVE_OBSERVABLES = {
    "pauli_x",
    "pauli_y",
    "pauli_z",
    "identity",
    "hadamard_observable",
    "hermitian_observable",
    "custom_observable",
}


# ---------------------------------------------------------------------------
# Per-entry capability corrections
# ---------------------------------------------------------------------------
#
# Capabilities were declared once per *section*, so all 38 encoding entries
# claimed ``differentiable``.  That is false for the entries the framework
# actually implements: a basis state is a step function of its bit and has no
# gradient on either engine, and superposition encoding takes no input at all,
# so there is nothing to differentiate with respect to.
#
# Three further tags are added here because they are what actually decides
# whether an encoding can be *composed* with a given model and engine.  They
# mirror the predicates in :mod:`qforge_ai.composition`, and the catalog tests
# assert that the two never disagree.
#
# ``symbolic_parameters``
#     Compiles with unbound parameters, i.e. usable inside a Qiskit
#     EstimatorQNN/SamplerQNN, which binds inputs after compilation.
# ``continuous_input``
#     Accepts arbitrary real features, i.e. usable behind the classical encoder
#     of a hybrid template.
# ``state_preparation``
#     Sets the initial state rather than loading data into an existing one.
_ENCODING_CAPABILITY_OVERRIDES: dict[str, frozenset[str]] = {
    # Discrete bits: no gradient, no symbolic form, no continuous input.
    "basis_encoding": frozenset({"data_encoding"}),
    "binary_encoding": frozenset({"data_encoding"}),
    "one_hot_encoding": frozenset({"data_encoding"}),
    "oracle_encoding": frozenset({"data_encoding"}),
    # Loads no data at all -- it only puts the register into |+>^n.
    "superposition_encoding": frozenset({"state_preparation"}),
    # Differentiable amplitudes, but ``initialize`` has no parameterized form,
    # so a Qiskit QNN cannot carry it.
    "amplitude_encoding": frozenset(
        {"data_encoding", "differentiable", "continuous_input", "state_preparation"}
    ),
    "complex_amplitude_encoding": frozenset(
        {"data_encoding", "differentiable", "continuous_input", "state_preparation"}
    ),
    "probability_encoding": frozenset(
        {"data_encoding", "differentiable", "continuous_input", "state_preparation"}
    ),
    "state_preparation_encoding": frozenset(
        {"data_encoding", "differentiable", "continuous_input", "state_preparation"}
    ),
    "qram_encoding": frozenset({"data_encoding", "state_preparation"}),
}

#: What every rotation/phase based encoding carries.
_DEFAULT_ENCODING_CAPABILITIES = frozenset(
    {"data_encoding", "differentiable", "continuous_input", "symbolic_parameters"}
)


def _capabilities_for(key: str, category: str, capabilities: frozenset[str]) -> frozenset[str]:
    """Return the capability set for *key*, correcting per-section defaults."""

    if category != "encoding":
        return capabilities
    return _ENCODING_CAPABILITY_OVERRIDES.get(key, _DEFAULT_ENCODING_CAPABILITIES)

DOCUMENT_SECTIONS = {
    "encoding": 1,
    "ansatz": 2,
    "entanglement": 2,
    "physics_chemistry": 3,
    "tensor_network": 4,
    "dynamic_circuit": 5,
    "measurement": 6,
    "observable": 6,
    "qnn_model": 7,
    "vision_model": 8,
    "sequence_model": 9,
    "graph_model": 10,
    "kernel": 11,
    "unsupervised_model": 12,
    "generative_model": 13,
    "reinforcement_model": 14,
    "optimization": 15,
    "algorithm": 16,
    "noise": 17,
    "error_mitigation": 18,
    "error_correction": 19,
    "continuous_variable": 20,
    "model_template": 21,
}

_CATEGORY_PURPOSE = {
    "encoding": "encodes classical data into a quantum state or into parametric phases",
    "ansatz": "builds a trainable variational quantum layer",
    "entanglement": "determines the connectivity topology of the two-qubit gates",
    "physics_chemistry": "represents physics and quantum chemistry Hamiltonian structures",
    "tensor_network": "defines a hierarchical, tensor-network inspired circuit architecture",
    "dynamic_circuit": "expresses mid-circuit measurement, reset and classical feed-forward flow",
    "measurement": "produces the selected output contract from the quantum state",
    "observable": "defines an observable for an expectation value or variance",
    "qnn_model": "connects the quantum layer to a supervised learning model",
    "vision_model": "defines an image and patch based quantum-vision architecture",
    "sequence_model": "defines a quantum model family for sequence and time series data",
    "graph_model": "defines a quantum learning approach over a graph structure",
    "kernel": "produces a classical kernel matrix from quantum state similarity",
    "unsupervised_model": "defines a quantum learning approach for unlabelled data",
    "generative_model": "defines a quantum-assisted generative distribution or model",
    "reinforcement_model": "defines a quantum-assisted policy or value model",
    "optimization": "defines a variational quantum optimization problem",
    "algorithm": "defines a fundamental quantum algorithm or subroutine",
    "noise": "adds physical or sampling noise to the simulation",
    "error_mitigation": "defines a runtime technique that reduces the effect of noise",
    "error_correction": "defines a logical qubit and error correction code",
    "continuous_variable": "defines the photonic continuous-variable component family",
    "model_template": "combines encoding, ansatz, measurement and classical layers into one architecture",
}

_CATEGORY_REFERENCES = {
    "encoding": ("https://doi.org/10.1038/s41586-019-0980-2",),
    "ansatz": ("https://doi.org/10.1038/s42254-021-00348-9",),
    "entanglement": ("CATALOG_EN.md",),
    "physics_chemistry": ("https://doi.org/10.1103/RevModPhys.92.015003",),
    "tensor_network": ("https://doi.org/10.1038/s41567-019-0648-8",),
    "dynamic_circuit": ("https://docs.quantum.ibm.com/guides/classical-feedforward-and-control-flow",),
    "measurement": ("https://docs.pennylane.ai/en/stable/introduction/measurements.html",),
    "observable": ("https://docs.pennylane.ai/en/stable/introduction/measurements.html",),
    "qnn_model": ("https://doi.org/10.1016/j.physrep.2021.08.003",),
    "vision_model": ("https://doi.org/10.1038/s41567-019-0648-8",),
    "sequence_model": ("https://doi.org/10.1016/j.physrep.2021.08.003",),
    "graph_model": ("https://doi.org/10.1088/2058-9565/ac6d98",),
    "kernel": ("https://doi.org/10.1038/s41586-019-0980-2",),
    "unsupervised_model": ("https://doi.org/10.1016/j.physrep.2021.08.003",),
    "generative_model": ("https://doi.org/10.1038/s41534-019-0157-8",),
    "reinforcement_model": ("https://doi.org/10.1038/s41534-021-00436-7",),
    "optimization": ("https://doi.org/10.48550/arXiv.1411.4028",),
    "algorithm": ("https://doi.org/10.1017/CBO9780511976667",),
    "noise": ("https://doi.org/10.1017/CBO9780511976667",),
    "error_mitigation": ("https://doi.org/10.1103/RevModPhys.95.045005",),
    "error_correction": ("https://doi.org/10.1017/CBO9780511976667",),
    "continuous_variable": ("https://doi.org/10.1103/RevModPhys.84.621",),
    "model_template": ("ARCHITECTURE_EN.md",),
}


def _description(name: str, category: str) -> str:
    return f"{name}, {_CATEGORY_PURPOSE[category]}."


def _status_for(key: str, category: str, default: ComponentStatus) -> ComponentStatus:
    if key in _NATIVE_KEYS or category == "entanglement" or key in _NATIVE_OBSERVABLES:
        return ComponentStatus.NATIVE
    if key in _COMPOSITE_KEYS:
        return ComponentStatus.COMPOSITE
    if default in {ComponentStatus.NATIVE, ComponentStatus.COMPOSITE}:
        return ComponentStatus.DESCRIPTOR
    return default


#: Kernel entries the core framework routes on both engines.  Everything else
#: in the kernel category stays a discovery descriptor.
_CORE_KERNEL_KEYS = frozenset(
    {"fidelity_quantum_kernel", "statevector_quantum_kernel", "qsvc"}
)


def _engines_for(
    key: str,
    category: str,
    engines: frozenset[Engine],
) -> frozenset[Engine]:
    if category == "continuous_variable":
        return IR_ONLY
    if category == "kernel":
        # PennyLane gained a real compute-uncompute fidelity kernel, so the
        # routed kernel entries are genuinely dual-engine.  The remaining
        # kernel families are still descriptors and stay Qiskit-shaped.
        return QISKIT_PL if key in _CORE_KERNEL_KEYS else QISKIT_ONLY
    return engines


def _implementation_for(status: ComponentStatus, category: str) -> str:
    if status == ComponentStatus.NATIVE:
        return "builtin_path" if category in {"entanglement", "measurement", "observable", "noise"} else "factory"
    if status == ComponentStatus.COMPOSITE:
        return "model_factory"
    if status == ComponentStatus.EXPERIMENTAL:
        return "plugin_required"
    return "descriptor"


def _dependencies_for(category: str, engines: frozenset[Engine]) -> tuple[str, ...]:
    dependencies = ["quantumforge-ai"]
    if Engine.QISKIT in engines:
        dependencies.append("qiskit>=2.0,<3")
    if Engine.PENNYLANE in engines:
        dependencies.append("pennylane>=0.40,<0.46")
    if category in {"qnn_model", "vision_model", "sequence_model", "graph_model", "model_template"}:
        dependencies.append("torch>=2.2,<3")
    if category == "physics_chemistry":
        dependencies.extend(("qiskit-nature or pennylane-qchem",))
    if category == "graph_model":
        dependencies.append("networkx")
    if category == "reinforcement_model":
        dependencies.append("gymnasium")
    if category == "continuous_variable":
        dependencies.append("strawberryfields/pennylane-sf plugin")
    return tuple(dict.fromkeys(dependencies))


def build_catalog() -> ComponentRegistry:
    registry = ComponentRegistry()
    for category, default_status, engines, capabilities, names in _SECTIONS:
        for name in names:
            key = normalize_key(name)
            status = _status_for(key, category, default_status)
            resolved_engines = _engines_for(key, category, engines)
            implementation = _implementation_for(status, category)
            registry.register(
                ComponentDescriptor(
                    key=key,
                    display_name=name,
                    category=category,
                    description=_description(name, category),
                    status=status,
                    engines=resolved_engines,
                    capabilities=_capabilities_for(key, category, capabilities),
                    aliases=_ALIASES.get(key, ()),
                    dependencies=_dependencies_for(category, resolved_engines),
                    references=_CATEGORY_REFERENCES[category],
                    implementation=implementation,
                    metadata={
                        "document_section": DOCUMENT_SECTIONS[category],
                        "implementation": implementation,
                        "maturity": str(status),
                        "runtime_contract": (
                            "runnable" if status in {ComponentStatus.NATIVE, ComponentStatus.COMPOSITE} else "discovery"
                        ),
                    },
                )
            )
    return registry


catalog = build_catalog()


def list_components(category: str | None = None) -> list[ComponentDescriptor]:
    return catalog.list(category=category)


def iter_section(category: str) -> Iterable[ComponentDescriptor]:
    yield from catalog.list(category=category)


def is_executable(key: str) -> bool:
    """Return whether *key* can actually be executed by the core framework.

    ``ComponentRegistry.is_runnable`` trusts the declared ``implementation``
    field.  This function verifies the claim instead: a ``model_factory``
    entry must resolve to a real :class:`~qforge_ai.models.ModelFactory`
    route, an ``entanglement`` entry must be accepted by ``AnsatzSpec``, and
    an ``observable`` entry must be one both adapters can evaluate.  The
    catalog previously advertised entries that satisfied none of these.
    """

    # Imported lazily: these modules import this catalog during package
    # initialization, so a module-level import would be circular.
    from .capabilities import CAPABILITIES
    from .circuits.factory import SUPPORTED_MEASUREMENTS
    from .models.factory import ModelFactory
    from .specs import Engine, normalize_entanglement, normalize_observable

    descriptor = catalog.get(key)
    if catalog.has_factory(descriptor.key):
        return True
    if descriptor.implementation == "model_factory":
        return ModelFactory.resolve_template(descriptor.key) is not None
    if descriptor.implementation != "builtin_path":
        return False
    if descriptor.category == "entanglement":
        return normalize_entanglement(descriptor.key) is not None
    if descriptor.category == "observable":
        return normalize_observable(descriptor.key) is not None
    if descriptor.category == "measurement":
        return descriptor.key in SUPPORTED_MEASUREMENTS
    if descriptor.category == "noise":
        return descriptor.key in CAPABILITIES[Engine.QISKIT].noise_channels
    return False


def executable_components() -> list[ComponentDescriptor]:
    """Return every catalog entry the core framework can actually execute."""

    # Importing the circuit factory registers the built-in builders.
    from .circuits.factory import CircuitFactory  # noqa: F401

    return [item for item in catalog.list() if is_executable(item.key)]


def capability_report() -> dict[str, Any]:
    """Return verified catalog counts.

    Published counts used to be maintained by hand in the documentation and
    overstated what the framework could run.  Deriving them here -- and
    asserting them in ``tests/test_catalog.py`` -- keeps the numbers from
    drifting away from the implementation again.
    """

    descriptors = catalog.list()
    executable = executable_components()
    executable_keys = {item.key for item in executable}
    by_status: dict[str, int] = {}
    by_category: dict[str, dict[str, int]] = {}
    for item in descriptors:
        by_status[str(item.status)] = by_status.get(str(item.status), 0) + 1
        bucket = by_category.setdefault(item.category, {"total": 0, "executable": 0})
        bucket["total"] += 1
        if item.key in executable_keys:
            bucket["executable"] += 1
    return {
        "total": len(descriptors),
        "executable": len(executable),
        "discovery_only": len(descriptors) - len(executable),
        "by_status": dict(sorted(by_status.items())),
        "by_category": dict(sorted(by_category.items())),
    }


__all__ = [
    "catalog",
    "list_components",
    "iter_section",
    "build_catalog",
    "is_executable",
    "executable_components",
    "capability_report",
]
