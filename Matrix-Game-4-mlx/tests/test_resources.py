import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from matrix_game_mlx.resources import checked_path, digest, files_for, install_file, main, matches


class ResourceTests(unittest.TestCase):
    def test_path_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "resources"
            root.mkdir()
            for bad in ("../escape", "/absolute", "C:/absolute", "a/../../b", "a\\b", "", "a/./b", "a//b", "a/"):
                with self.assertRaises(ValueError): checked_path(root, bad)
            (root / "link").symlink_to(Path(tmp), target_is_directory=True)
            with self.assertRaises(ValueError): checked_path(root, "link/escape")

    def test_verified_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp)/"source", Path(tmp)/"data/file"
            src.write_bytes(b"tensor-fixture")
            item = dict(path="data/file", size=src.stat().st_size, sha256=digest(src))
            install_file(src, dst, item)
            self.assertTrue(matches(dst, item))
            src.write_bytes(b"bad")
            with self.assertRaises(ValueError): install_file(src, dst, item)
            self.assertEqual(dst.read_bytes(), b"tensor-fixture")

    def test_manifest_selection(self):
        entry = lambda path: dict(path=path, size=0, sha256=hashlib.sha256(b"").hexdigest())
        manifest = dict(schema=1, bundles={"shared":[entry("common")], "castle":[entry("castle")], "ancient-egypt":[entry("egypt")]})
        self.assertEqual([f["path"] for f in files_for(manifest,"castle")], ["common","castle"])
        manifest["bundles"]["castle"].append(entry("common"))
        with self.assertRaises(ValueError): files_for(manifest,"castle")

    def test_repository_path_validation(self):
        for bad in ("../escape", "/absolute", "a\\b", "a//b", "", None):
            entry = dict(path="models/castle/model.safetensors", repo_path=bad,
                         size=0, sha256=hashlib.sha256(b"").hexdigest())
            with self.subTest(repo_path=bad), self.assertRaises(ValueError):
                files_for(dict(schema=1, bundles={"castle": [entry]}), "castle")

    def test_repository_path_download_keeps_local_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / "fixture"
            fixture.write_bytes(b"tensor-fixture")
            entry = dict(path="models/castle/model.safetensors",
                         repo_path="fps-castle/model.safetensors",
                         size=fixture.stat().st_size, sha256=digest(fixture))
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(dict(schema=1, bundles={"castle": [entry]})))
            download = Mock(return_value=str(fixture))
            with patch.dict("sys.modules", {"huggingface_hub": SimpleNamespace(hf_hub_download=download)}):
                main(["download", "--manifest", str(manifest), "--scene", "castle",
                      "--resources", str(root / "resources"), "--repo-id", "org/model",
                      "--repo-type", "model", "--revision", "a" * 40])
            download.assert_called_once_with("org/model", entry["repo_path"], repo_type="model", revision="a" * 40)
            self.assertTrue(matches(root / "resources" / entry["path"], entry))
            self.assertFalse((root / "resources" / entry["repo_path"]).exists())

    def test_external_repository_requires_pin(self):
        base = dict(path="base/tokenizer.json", size=0, sha256=hashlib.sha256(b"").hexdigest())
        for override in (dict(repo_id="org/model"), dict(repo_id="org/model", repo_type="model", revision="main"),
                         dict(repo_id="../model", repo_type="model", revision="b" * 40),
                         dict(repo_id=None, repo_type="model", revision="b" * 40), dict(revision="b" * 40)):
            with self.subTest(override=override), self.assertRaises(ValueError):
                files_for(dict(schema=1, bundles={"shared": [{**base, **override}]}), "castle")

    def test_existing_valid_file_not_redownloaded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "resources" / "data" / "file"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"already-here")
            entry = dict(path="data/file", size=target.stat().st_size, sha256=digest(target))
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(dict(schema=1, bundles={"shared": [entry]})))
            download = Mock(side_effect=AssertionError("must not download"))
            out = io.StringIO()
            with patch.dict("sys.modules", {"huggingface_hub": SimpleNamespace(hf_hub_download=download)}), \
                    contextlib.redirect_stdout(out):
                main(["download", "--manifest", str(manifest), "--scene", "castle",
                      "--resources", str(root / "resources"), "--repo-id", "org/model",
                      "--revision", "a" * 40])
            download.assert_not_called()
            self.assertIn("Verified data/file", out.getvalue())

    def test_external_repository_download(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / "fixture"
            fixture.write_bytes(b"upstream-fixture")
            entry = dict(path="base/tokenizer.json", repo_path="tokenizer.json",
                         repo_id="upstream/base", repo_type="model", revision="b" * 40,
                         size=fixture.stat().st_size, sha256=digest(fixture))
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(dict(schema=1, bundles={"shared": [entry]})))
            download = Mock(return_value=str(fixture))
            with patch.dict("sys.modules", {"huggingface_hub": SimpleNamespace(hf_hub_download=download)}):
                main(["download", "--manifest", str(manifest), "--scene", "castle",
                      "--resources", str(root / "resources"), "--repo-id", "org/model",
                      "--repo-type", "model", "--revision", "a" * 40])
            download.assert_called_once_with("upstream/base", "tokenizer.json", repo_type="model", revision="b" * 40)
            self.assertTrue(matches(root / "resources" / entry["path"], entry))
