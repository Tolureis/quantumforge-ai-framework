# Contribution Guide

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,qiskit,pennylane,torch]'
pytest
ruff check .
```

## Adding a New Catalog Component

1. Add a unique key and the correct category in `catalog.py`.

2. Mark the support level correctly as `native`, `composite`, `experimental`, or `descriptor`.

3. If the component is `native`, add an IR factory or an explicit `builtin_path`, along with at least one integration test for each supported engine. `catalog.is_runnable(key)` must return `true`.

4. Populate the `dependencies`, `references`, `implementation`, and documentation-section metadata.

5. Document qubit ordering, parameter ordering, shots behavior, and noise behavior.

6. Do not add new optional dependencies to the core import chain.

## Quality Gates

* All core tests must pass.
* Qiskit and PennyLane outputs must match within the defined tolerance for ideal circuits.
* New APIs must include type annotations and docstrings.
* Include an example configuration that preserves seed information and resource measurements.
* A research-level approach must not be marked as `native` as if it were fully implemented and operational.

