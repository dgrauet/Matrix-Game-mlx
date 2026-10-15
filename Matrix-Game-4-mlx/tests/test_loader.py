import json
import pytest

from matrix_game_mlx.inference import loader


def _fake_pack(tmp_path, content=b"abc"):
    shared = tmp_path / "shared"; (shared / "google/umt5-xxl").mkdir(parents=True)
    files = {"t5_encoder.safetensors": content, "vae.safetensors": b"v",
             "google/umt5-xxl/tokenizer.json": b"{}"}
    for rel, data in files.items():
        (shared / rel).write_bytes(data)
    entries = [{"path": rel, "sha256": loader.sha256_file(shared / rel)} for rel in files]
    pack = tmp_path / "pack"; pack.mkdir()
    (pack / "split_model.json").write_text(json.dumps({"shared": {
        "repo_id": "x/y", "revision": "r", "components": {"t5_encoder": "t5_encoder.safetensors", "vae": "vae.safetensors"},
        "tokenizer": "google/umt5-xxl", "files": entries}}))
    return pack, shared


def test_local_shared_dir_ok(tmp_path, monkeypatch):
    pack, shared = _fake_pack(tmp_path)
    monkeypatch.setenv("MATRIX_GAME_4_MLX_SHARED_DIR", str(shared))
    out = loader.resolve_shared_files(pack)
    assert out["t5_encoder.safetensors"] == shared / "t5_encoder.safetensors"
    assert out["google/umt5-xxl"] == shared / "google/umt5-xxl"


def test_sha_mismatch_raises(tmp_path, monkeypatch):
    pack, shared = _fake_pack(tmp_path)
    (shared / "vae.safetensors").write_bytes(b"tampered")
    monkeypatch.setenv("MATRIX_GAME_4_MLX_SHARED_DIR", str(shared))
    with pytest.raises(ValueError, match="sha256"):
        loader.resolve_shared_files(pack)


def test_hub_download_called_with_pinned_revision(tmp_path, monkeypatch):
    pack, shared = _fake_pack(tmp_path)
    monkeypatch.delenv("MATRIX_GAME_4_MLX_SHARED_DIR", raising=False)
    calls = []
    def fake_download(repo_id, filename, revision):
        calls.append((repo_id, filename, revision))
        return str(shared / filename)
    monkeypatch.setattr(loader, "hf_hub_download", fake_download)
    loader.resolve_shared_files(pack)
    assert {c[2] for c in calls} == {"r"} and {c[0] for c in calls} == {"x/y"} and len(calls) == 3
