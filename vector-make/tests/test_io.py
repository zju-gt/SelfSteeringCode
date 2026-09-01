from pathlib import Path

import torch

from self_steering.utils.io import (
    append_jsonl,
    atomic_save_json,
    atomic_save_tensors,
    read_jsonl,
    sha256_file,
    sha256_text,
)


def test_jsonl_round_trip_preserves_unicode(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "rows.jsonl"
    append_jsonl(path, {"text": "数量推理"})
    append_jsonl(path, {"text": "逻辑推演"})
    assert list(read_jsonl(path)) == [{"text": "数量推理"}, {"text": "逻辑推演"}]


def test_atomic_writers_create_complete_artifacts(tmp_path: Path) -> None:
    json_path = tmp_path / "metadata.json"
    tensor_path = tmp_path / "vectors.safetensors"
    atomic_save_json(json_path, {"ok": True})
    atomic_save_tensors(tensor_path, {"v": torch.tensor([1.0, 2.0])})
    assert json_path.read_text(encoding="utf-8").strip().startswith("{")
    assert tensor_path.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_sha256_text_is_stable() -> None:
    assert (
        sha256_text("abc")
        == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_sha256_file_hashes_binary_content(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"abc")
    assert sha256_file(path) == sha256_text("abc")
