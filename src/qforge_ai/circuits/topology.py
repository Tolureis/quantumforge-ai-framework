"""Entanglement topology generation."""

from __future__ import annotations

import random
from collections.abc import Iterable

from ..errors import ConfigurationError


def entanglement_edges(
    num_qubits: int,
    topology: str,
    *,
    layer: int = 0,
    seed: int = 42,
    custom_edges: Iterable[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    if num_qubits < 2 or topology == "none":
        return []
    if topology == "linear":
        return [(i, i + 1) for i in range(num_qubits - 1)]
    if topology == "reverse_linear":
        return [(i + 1, i) for i in range(num_qubits - 2, -1, -1)]
    if topology in {"ring", "circular"}:
        if num_qubits == 2:
            return [(0, 1)]
        return [(i, (i + 1) % num_qubits) for i in range(num_qubits)]
    if topology == "full":
        return [(i, j) for i in range(num_qubits) for j in range(i + 1, num_qubits)]
    if topology == "pairwise":
        offset = layer % 2
        return [(i, i + 1) for i in range(offset, num_qubits - 1, 2)]
    if topology == "alternating":
        offset = layer % 2
        edges = [(i, i + 1) for i in range(offset, num_qubits - 1, 2)]
        if num_qubits > 2 and layer % 2:
            edges.append((num_qubits - 1, 0))
        return edges
    if topology == "star":
        return [(0, i) for i in range(1, num_qubits)]
    if topology == "grid":
        width = max(2, int(num_qubits**0.5))
        grid_edges: list[tuple[int, int]] = []
        for i in range(num_qubits):
            if i + 1 < num_qubits and (i + 1) % width:
                grid_edges.append((i, i + 1))
            if i + width < num_qubits:
                grid_edges.append((i, i + width))
        return grid_edges
    if topology == "random":
        rng = random.Random(seed + layer)
        candidates = [(i, j) for i in range(num_qubits) for j in range(i + 1, num_qubits)]
        rng.shuffle(candidates)
        return candidates[: max(num_qubits - 1, 1)]
    if topology in {"hardware", "user_defined"}:
        if custom_edges is None:
            raise ConfigurationError(f"custom_edges is required for {topology}.")
        edges = list(custom_edges)
        invalid = [edge for edge in edges if any(w < 0 or w >= num_qubits for w in edge)]
        if invalid:
            raise ConfigurationError(f"Invalid custom edge: {invalid}")
        return edges
    raise ConfigurationError(f"Unknown entanglement topology: {topology}")
