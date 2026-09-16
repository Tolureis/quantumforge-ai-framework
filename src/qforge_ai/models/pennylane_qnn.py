"""PennyLane differentiable layer and quantum-kernel factories.

The kernel itself lives in :mod:`qforge_ai.models.kernel`, which both engines
share; this module keeps the PennyLane-facing entry points and the
scikit-learn estimator that drives them.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import numpy as np

from ..backends._optional import require
from ..backends.pennylane_adapter import PennyLaneAdapter
from ..errors import ConfigurationError
from ..ir import CircuitIR
from ..specs import BackendSpec, Engine, NoiseSpec
from .kernel import (  # noqa: F401  - re-exported public names
    FidelityKernel,
    QuantumKernel,
    build_quantum_kernel,
    enforce_psd,
    feature_dimension,
    feature_map_operations,
)


def as_torch_layer(
    ir: CircuitIR,
    *,
    device: str = "default.qubit",
    shots: int | None = None,
    diff_method: str = "parameter-shift",
    noise: NoiseSpec | None = None,
    seed: int = 42,
    device_options: dict[str, Any] | None = None,
    gradient_options: dict[str, Any] | None = None,
) -> Any:
    qml = require("pennylane", package="pennylane", extra="pennylane", feature="TorchLayer")
    torch = require("torch", package="torch", extra="torch", feature="PennyLane TorchLayer")
    nn = require("torch.nn", package="torch", extra="torch", feature="PennyLane TorchLayer")
    qnode = PennyLaneAdapter(
        device=device,
        shots=shots,
        seed=seed,
        noise=noise,
        device_options=device_options,
        gradient_options=gradient_options,
    ).compile(
        ir, interface="torch", diff_method=diff_method
    )
    weight_count = ir.parameter_groups().get("weight", 0)
    layer = qml.qnn.TorchLayer(qnode, {"weights": (weight_count,)})

    class BatchSafeTorchLayer(nn.Module):
        """Normalize PennyLane multi-measurement axes to ``(*batch, outputs)``."""

        def __init__(self) -> None:
            super().__init__()
            self.layer = layer

        def forward(self, inputs: Any) -> Any:
            batch_shape = tuple(inputs.shape[:-1])
            if not batch_shape:
                return self.layer(inputs).reshape(-1)
            # PennyLane parameter-shift cannot differentiate trainable
            # broadcasted tapes.  Flattening and vmap/stacking single-sample
            # evaluations preserves batch semantics and both input/weight
            # gradients across all supported diff methods.
            flattened = inputs.reshape(-1, inputs.shape[-1])
            outputs = torch.stack([self.layer(sample).reshape(-1) for sample in flattened])
            return outputs.reshape(*batch_shape, -1)

    return BatchSafeTorchLayer()


# ---------------------------------------------------------------------------
# Quantum kernels
# ---------------------------------------------------------------------------
#
# ``FidelityKernel``, ``feature_map_operations`` and ``enforce_psd`` are
# imported from :mod:`qforge_ai.models.kernel` above and re-exported here:
# both engines now execute one shared kernel definition instead of two
# implementations that disagreed about duplicates, readout noise and input
# width.  The names are kept so 1.2.x imports and pickles keep resolving.


def build_fidelity_kernel(
    ir: CircuitIR,
    *,
    backend: BackendSpec | None = None,
    fidelity: str = "auto",
) -> tuple[QuantumKernel, str]:
    """Build the PennyLane quantum kernel *backend* and *fidelity* describe.

    Everything below the engine boundary is
    :func:`qforge_ai.models.kernel.build_quantum_kernel`, so the same
    definition means the same thing here as on the Qiskit route: the same
    resolved kernel kind, the same feature-map rules, the same input-width
    contract, the same capability preflight and the same Gram-matrix assembly.
    """

    require("pennylane", package="pennylane", extra="pennylane", feature="quantum kernel")
    resolved = backend or BackendSpec(engine=Engine.PENNYLANE, device="default.qubit")
    if resolved.engine != Engine.PENNYLANE:
        raise ConfigurationError(
            f"The PennyLane kernel path requires a PennyLane BackendSpec; got {resolved.engine}."
        )
    return build_quantum_kernel(ir, backend=resolved, fidelity=fidelity)


#: Cache for the lazily built estimator class.  ``scikit-learn`` is an optional
#: dependency and this module is imported during package initialization, so the
#: class cannot be declared at import time -- but it must still be a *single*
#: stable class object, otherwise ``pickle`` cannot resolve it by name.
_QSVC_CLASS: Any = None

if TYPE_CHECKING:  # pragma: no cover - resolved at runtime by ``__getattr__``
    #: Declared for static analysis only; see :func:`quantum_svc_class`.
    PennyLaneQSVC: Any


def quantum_svc_class() -> Any:
    """Return the module's scikit-learn compatible QSVC class.

    Built once and cached.  ``pickle`` stores an instance by
    ``module.__qualname__`` and then checks that the looked-up class *is* the
    original, so the cache is what makes a fitted estimator serializable --
    together with ``__getattr__`` below, which resolves the name for the
    unpickler on a fresh interpreter.
    """

    global _QSVC_CLASS
    if _QSVC_CLASS is not None:
        return _QSVC_CLASS
    svm = require(
        "sklearn.svm",
        package="scikit-learn",
        extra="pennylane",
        feature="PennyLane quantum kernel SVC",
    )
    exceptions = require(
        "sklearn.exceptions",
        package="scikit-learn",
        extra="pennylane",
        feature="PennyLane quantum kernel SVC",
    )
    validation = require(
        "sklearn.utils.validation",
        package="scikit-learn",
        extra="pennylane",
        feature="PennyLane quantum kernel SVC",
    )
    metaestimators = require(
        "sklearn.utils.metaestimators",
        package="scikit-learn",
        extra="pennylane",
        feature="PennyLane quantum kernel SVC",
    )

    class QuantumKernelNotFittedError(ConfigurationError, exceptions.NotFittedError):
        """Raised for an unfitted estimator, in both error vocabularies.

        scikit-learn's contract is that ``predict``/``decision_function``/
        ``predict_proba`` raise ``NotFittedError`` before ``fit``; QuantumForge's
        is that a misuse of a spec raises ``ConfigurationError``.  Inheriting
        from both keeps ``check_estimator`` and existing ``except
        ConfigurationError`` handlers working at the same time.
        """

    def _proba_available(estimator: Any) -> bool:
        """Mirror ``SVC.predict_proba``'s availability contract.

        ``predict_proba`` on ``SVC`` is an ``available_if`` descriptor: the
        attribute does not exist unless ``probability=True``.  Overriding it
        with a plain method made ``hasattr(estimator, 'predict_proba')`` true
        for every instance, and eight scikit-learn estimator checks failed with
        "This 'PennyLaneQSVC' has no attribute 'predict_proba'" when the
        underlying SVC then refused.
        """

        return bool(getattr(estimator, "probability", False))

    class PennyLaneQSVC(svm.SVC):  # type: ignore[misc, valid-type]
        """scikit-learn SVC whose Gram matrix comes from a quantum feature map.

        Hyper-parameters are delegated to ``SVC`` rather than re-declared.  A
        hard-coded parameter list is a copy of scikit-learn's signature that
        cannot be kept in sync: ``probability`` is deprecated in scikit-learn
        1.9 and removed in 1.11, so a class naming it explicitly would stop
        constructing on a supported version of a declared dependency.  Reading
        the list from ``SVC._get_param_names()`` means this estimator advertises
        exactly the hyper-parameters the installed scikit-learn has, and
        ``get_params``/``set_params``/``clone`` follow automatically.
        """

        def __init__(self, quantum_kernel: Any = None, **params: Any) -> None:
            unsupported = sorted({"kernel"} & set(params))
            if unsupported:
                raise ConfigurationError(
                    f"The quantum kernel SVC does not accept these keys: {unsupported}; "
                    "the kernel comes from the quantum circuit."
                )
            super().__init__(kernel="precomputed", **params)
            self.quantum_kernel = quantum_kernel

        @classmethod
        def _get_param_names(cls) -> list[str]:
            names = set(svm.SVC._get_param_names())
            names.discard("kernel")
            names.add("quantum_kernel")
            return sorted(names)

        # -- validation ---------------------------------------------------
        def _check_kernel(self) -> Any:
            if self.quantum_kernel is None:
                raise ConfigurationError(
                    "PennyLaneQSVC requires a quantum_kernel; "
                    "build it with build_pennylane_qsvc(...)."
                )
            return self.quantum_kernel

        def _validate_samples(self, X: Any, *, reset: bool) -> Any:  # noqa: N803
            """Run scikit-learn's own input validation on the raw features.

            The Gram matrix handed to ``super().fit`` is square, so everything
            scikit-learn infers from it -- ``n_features_in_`` above all -- used
            to describe the *sample count* instead of the feature count: a
            six-sample, two-feature problem reported ``n_features_in_ == 6``.
            Validating the real feature matrix here also gives the sparse,
            complex, NaN and empty-input rejections the estimator contract
            requires, in scikit-learn's own words.
            """

            return validation.validate_data(
                self,
                X,
                reset=reset,
                accept_sparse=False,
                dtype="numeric",
                ensure_all_finite=True,
                ensure_min_samples=1,
                ensure_min_features=1,
            )

        # ``fit`` keeps the training rows: a precomputed-kernel SVC needs the
        # *training* Gram matrix at predict time, not the raw features.
        def fit(self, X: Any, y: Any, sample_weight: Any = None) -> Any:  # noqa: N803
            kernel = self._check_kernel()
            samples = np.asarray(self._validate_samples(X, reset=True), dtype=float)
            self.qforge_train_ = samples
            fitted = super().fit(kernel(samples), y, sample_weight=sample_weight)
            # ``super().fit`` saw the square Gram matrix; restore the contract.
            self.n_features_in_ = samples.shape[1]
            return fitted

        def _gram(self, X: Any) -> np.ndarray:  # noqa: N803
            validation.check_is_fitted(self, "qforge_train_")
            kernel = self._check_kernel()
            samples = np.asarray(self._validate_samples(X, reset=False), dtype=float)
            return np.asarray(kernel(samples, self.qforge_train_), dtype=float)

        def __sklearn_check_is_fitted__(self) -> bool:
            return hasattr(self, "qforge_train_")

        @contextmanager
        def _as_precomputed(self) -> Any:
            """Let ``BaseLibSVM`` validate the Gram matrix, not the features.

            This estimator has two input widths: the public one -- the raw
            feature count, which ``n_features_in_`` must report -- and the
            internal one, the training-sample count of the precomputed Gram
            matrix ``libsvm`` actually receives.  ``_validate_for_predict``
            checks the second against ``n_features_in_``, so the attribute is
            swapped for exactly the duration of the delegated call and restored
            afterwards, including on error.
            """

            outer = getattr(self, "n_features_in_", None)
            self.n_features_in_ = int(self.qforge_train_.shape[0])
            try:
                yield
            finally:
                if outer is None:  # pragma: no cover - fit always sets it
                    del self.n_features_in_
                else:
                    self.n_features_in_ = outer

        def predict(self, X: Any) -> Any:  # noqa: N803
            gram = self._gram(X)
            with self._as_precomputed():
                return super().predict(gram)

        def decision_function(self, X: Any) -> Any:  # noqa: N803
            gram = self._gram(X)
            with self._as_precomputed():
                return super().decision_function(gram)

        @metaestimators.available_if(_proba_available)
        def predict_proba(self, X: Any) -> Any:  # noqa: N803
            gram = self._gram(X)
            with self._as_precomputed():
                return super().predict_proba(gram)

        @metaestimators.available_if(_proba_available)
        def predict_log_proba(self, X: Any) -> Any:  # noqa: N803
            gram = self._gram(X)
            with self._as_precomputed():
                return super().predict_log_proba(gram)

        def score(self, X: Any, y: Any, sample_weight: Any = None) -> Any:  # noqa: N803
            metrics = require(
                "sklearn.metrics",
                package="scikit-learn",
                extra="pennylane",
                feature="quantum kernel SVC scoring",
            )
            return metrics.accuracy_score(y, self.predict(X), sample_weight=sample_weight)

        # ``BaseLibSVM`` marks any ``kernel='precomputed'`` estimator as
        # *pairwise*, which tells cross-validation that ``X`` is already a
        # square Gram matrix and to slice it on both axes.  Here ``X`` is a
        # plain feature matrix -- the Gram matrix is built inside ``fit`` from
        # the quantum feature map -- so without this override every
        # ``GridSearchCV``/``cross_val_score`` run died with
        # "X should be a square kernel matrix".
        def __sklearn_tags__(self) -> Any:
            tags = super().__sklearn_tags__()
            input_tags = getattr(tags, "input_tags", None)
            if input_tags is not None:
                input_tags.pairwise = False
                input_tags.sparse = False
            return tags

        def _more_tags(self) -> dict[str, Any]:  # scikit-learn < 1.6
            return {"pairwise": False}

        @property
        def qforge_backend_spec(self) -> Any:
            kernel = self.quantum_kernel
            return None if kernel is None else kernel.qforge_backend_spec

        @property
        def qforge_kernel_kind(self) -> Any:
            kernel = self.quantum_kernel
            return None if kernel is None else kernel.qforge_kernel_kind

        @property
        def qforge_shots(self) -> Any:
            kernel = self.quantum_kernel
            return None if kernel is None else kernel.qforge_shots

        @property
        def qforge_quantum_kernel(self) -> Any:
            return self.quantum_kernel

    # The constructor takes ``**params`` so it never has to restate
    # scikit-learn's parameter list -- but scikit-learn also *introspects* the
    # signature (``check_do_not_raise_errors_in_init_or_set_params`` calls the
    # constructor once per declared parameter), and a ``**params`` catch-all
    # reads there as a parameter literally named "params".  Publishing the
    # resolved signature gives introspection the real names while the
    # implementation stays delegating.
    PennyLaneQSVC.__init__.__signature__ = _delegated_signature(svm.SVC)  # type: ignore[attr-defined]
    # ``NotFittedError`` has to reach callers of the *unfitted* estimator, and
    # ``check_is_fitted`` raises scikit-learn's own class; the hybrid above is
    # what the framework raises for the same condition from its own code.
    PennyLaneQSVC.QuantumKernelNotFittedError = QuantumKernelNotFittedError
    PennyLaneQSVC.__module__ = __name__
    PennyLaneQSVC.__qualname__ = "PennyLaneQSVC"
    _QSVC_CLASS = PennyLaneQSVC
    return PennyLaneQSVC


def _delegated_signature(svc: Any) -> Any:
    """Build ``(quantum_kernel=None, *, <every SVC parameter but kernel>)``."""

    import inspect

    declared = inspect.signature(svc.__init__).parameters
    parameters = [
        # ``self`` stays in the signature: ``inspect.signature`` on the *bound*
        # ``estimator.__init__`` drops the first parameter, and without it the
        # one parameter this estimator adds -- ``quantum_kernel`` -- would be
        # the parameter that disappeared from introspection.
        inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        inspect.Parameter("quantum_kernel", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None),
    ]
    for name in svc._get_param_names():
        if name == "kernel":
            continue
        default = declared[name].default if name in declared else None
        parameters.append(
            inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=default)
        )
    return inspect.Signature(parameters, return_annotation=None)


def __getattr__(name: str) -> Any:
    """Resolve ``PennyLaneQSVC`` on demand (also used by the unpickler)."""

    if name == "PennyLaneQSVC":
        return quantum_svc_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def build_pennylane_qsvc(
    ir: CircuitIR,
    *,
    backend: BackendSpec | None = None,
    fidelity: str = "auto",
    **kwargs: Any,
) -> Any:
    """Return a scikit-learn compatible SVC driven by the PennyLane kernel."""

    estimator_class = quantum_svc_class()
    unsupported = sorted({"kernel", "quantum_kernel"} & set(kwargs))
    if unsupported:
        raise ConfigurationError(
            f"The quantum kernel SVC does not accept these keys: {unsupported}; "
            "the kernel comes from the quantum circuit."
        )
    allowed = frozenset(estimator_class._get_param_names()) - {"quantum_kernel"}
    unknown = sorted(set(kwargs) - allowed)
    if unknown:
        raise ConfigurationError(
            f"PennyLaneQSVC does not accept these options keys: {unknown}. "
            f"Valid ones: {sorted(allowed | {'fidelity'})}"
        )
    kernel, _ = build_fidelity_kernel(ir, backend=backend, fidelity=fidelity)
    return estimator_class(quantum_kernel=kernel, **kwargs)


__all__ = [
    "as_torch_layer",
    "feature_dimension",
    "feature_map_operations",
    "enforce_psd",
    "FidelityKernel",
    "QuantumKernel",
    "PennyLaneQSVC",
    "quantum_svc_class",
    "build_fidelity_kernel",
    "build_pennylane_qsvc",
]
