from qforge_ai import AnsatzSpec, CircuitSpec, EncodingSpec, ExperimentSpec
from qforge_ai.circuits import CircuitFactory
from qforge_ai.tracking import build_manifest, fingerprint_files, hash_files


def test_dataset_content_hash_is_independent_of_source_path(tmp_path) -> None:
    first = tmp_path / "first" / "dataset.bin"
    second = tmp_path / "second" / "renamed.bin"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"same quantum dataset\n")
    second.write_bytes(b"same quantum dataset\n")

    assert hash_files([first]) == hash_files([second])
    first_fingerprint = fingerprint_files([first])
    second_fingerprint = fingerprint_files([second])
    assert first_fingerprint.content_sha256 == second_fingerprint.content_sha256
    assert first_fingerprint.source_paths != second_fingerprint.source_paths
    assert first_fingerprint.file_count == 1


def test_multi_file_hash_is_content_order_independent(tmp_path) -> None:
    first = tmp_path / "one.bin"
    second = tmp_path / "two.bin"
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    assert hash_files([first, second]) == hash_files([second, first])


def test_manifest_separates_dataset_content_and_provenance(tmp_path) -> None:
    dataset = tmp_path / "dataset.bin"
    dataset.write_bytes(b"manifest dataset")
    experiment = ExperimentSpec(
        name="manifest-test",
        circuit=CircuitSpec(
            1,
            EncodingSpec("angle", 1),
            AnsatzSpec("real_amplitudes", 1),
        )
    )
    circuit = CircuitFactory().build(experiment.circuit)
    manifest = build_manifest(experiment, circuit, dataset_paths=[dataset])
    assert manifest.dataset is not None
    assert manifest.dataset_hash == manifest.dataset.content_sha256
    assert manifest.dataset.source_paths == (str(dataset),)
