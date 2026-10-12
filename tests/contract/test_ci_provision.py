"""tools/ci/provision.py: the pinned checkpoints of the oracle tiers. No network: a local HTTP server stands in for
the Hub."""

import hashlib
import http.server
import json
import os
import threading
from functools import partial

import pytest

from tools.ci import provision


def test_manifest_pins_every_checkpoint_the_tests_read():
    manifest = provision.load_manifest()
    assert provision.validate_manifest(manifest) == []
    for name in ("Qwen3.5-0.8B", "nvidia-Qwen3-8B-NVFP4", "Dogacel-Qwen3-8B-DSpark",
                 "mlx-community-Llama-3.2-1B-Instruct-4bit", "mlx-community-Llama-3.2-3B-Instruct-4bit",
                 "HuggingFaceTB-SmolLM2-1.7B-Instruct"):
        assert "config.json" in manifest[name]["files"]


def test_manifest_validation_rejects_unpinned_entries():
    bad = {"a/b": {"revision": "main", "files": {"../x": [1, "0" * 64], "y": [-1, "nope"]}}}
    problems = provision.validate_manifest(bad)
    assert len(problems) == 4


@pytest.fixture
def hub(tmp_path, monkeypatch):
    """Serves tmp_path/hub as <endpoint>/<repo>/resolve/<revision>/<file>."""
    root = tmp_path / "hub"
    root.mkdir()
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    handler.log_message = lambda *a: None
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("HF_ENDPOINT", f"http://127.0.0.1:{server.server_address[1]}")
    yield root
    server.shutdown()


def _publish(hub_root, files):
    rev = "a" * 40
    base = hub_root / "org" / "ckpt" / "resolve" / rev
    base.mkdir(parents=True)
    entry = {"repo": "org/ckpt", "revision": rev, "files": {}}
    for rel, data in files.items():
        (base / rel).parent.mkdir(parents=True, exist_ok=True)
        (base / rel).write_bytes(data)
        entry["files"][rel] = [len(data), hashlib.sha256(data).hexdigest()]
    return entry


def test_provision_downloads_verifies_and_prunes(hub, tmp_path):
    entry = _publish(hub, {"config.json": b"{}", "model.safetensors": b"w" * 5000})
    dest = tmp_path / "models"
    (dest / "ckpt").mkdir(parents=True)
    (dest / "ckpt" / "stale.safetensors").write_bytes(b"old")
    assert provision.provision_one("ckpt", entry, dest) == (2, 5002)
    assert not (dest / "ckpt" / "stale.safetensors").exists()
    assert (dest / "ckpt" / "model.safetensors").read_bytes() == b"w" * 5000
    assert provision.provision_one("ckpt", entry, dest) == (0, 0)          # verified and recorded: nothing to do
    (dest / "ckpt" / "model.safetensors").write_bytes(b"x" * 5000)        # same size, other bytes
    assert provision.provision_one("ckpt", entry, dest) == (1, 5000)
    assert (dest / "ckpt" / "model.safetensors").read_bytes() == b"w" * 5000


def test_provision_never_follows_a_symlink_out_of_the_cache(hub, tmp_path):
    entry = _publish(hub, {"config.json": b"{}"})
    outside = tmp_path / "home"
    (outside / "sub").mkdir(parents=True)
    (outside / "keep.txt").write_bytes(b"precious")
    (outside / "sub" / "keep.txt").write_bytes(b"precious")
    dest = tmp_path / "models"
    dest.mkdir()
    (dest / "ckpt").symlink_to(outside)                     # what a test could leave behind
    assert provision.provision_one("ckpt", entry, dest) == (1, 2)
    assert not (dest / "ckpt").is_symlink() and (dest / "ckpt" / "config.json").read_bytes() == b"{}"
    (dest / "ckpt" / "sub").symlink_to(outside / "sub")
    assert provision.provision_one("ckpt", entry, dest) == (0, 0)
    assert not (dest / "ckpt" / "sub").exists()
    assert (outside / "keep.txt").read_bytes() == (outside / "sub" / "keep.txt").read_bytes() == b"precious"


def test_provision_replaces_what_occupies_a_files_path(hub, tmp_path):
    entry = _publish(hub, {"config.json": b"{}", "sub/model.safetensors": b"w" * 10})
    dest = tmp_path / "models"
    (dest / "ckpt" / "config.json").mkdir(parents=True)       # a directory where a file belongs
    (dest / "ckpt" / "config.json" / "x").write_bytes(b"x")
    (dest / "ckpt" / "sub").write_bytes(b"not a directory")   # a file where a directory belongs
    assert provision.provision_one("ckpt", entry, dest) == (2, 12)
    assert (dest / "ckpt" / "config.json").read_bytes() == b"{}"
    assert (dest / "ckpt" / "sub" / "model.safetensors").read_bytes() == b"w" * 10
    (dest / "ckpt" / "config.json").unlink()
    (dest / "ckpt" / "config.json.part").mkdir()                # not a resumable download
    (dest / "ckpt" / "sub" / "model.safetensors").unlink()
    os.mkfifo(dest / "ckpt" / "sub" / "model.safetensors.part")   # opening it to resume would block
    assert provision.provision_one("ckpt", entry, dest) == (2, 12)
    assert not (dest / "ckpt" / "config.json.part").exists()
    (dest / "ckpt" / provision.STAMP).unlink()
    (dest / "ckpt" / provision.STAMP).mkdir()                   # the verification record's path, taken
    (dest / "ckpt" / (provision.STAMP + ".tmp")).mkdir()
    assert provision.provision_one("ckpt", entry, dest, rehash=True) == (0, 0)
    assert (dest / "ckpt" / provision.STAMP).is_file()


def test_provision_replaces_a_cache_root_that_is_not_a_directory(hub, tmp_path):
    entry = _publish(hub, {"config.json": b"{}"})
    manifest = tmp_path / "models.json"
    manifest.write_text(json.dumps({"ckpt": entry}))
    dest = tmp_path / "models"
    dest.symlink_to(tmp_path / "nowhere")                   # a dangling symlink a test left
    assert provision.main(["--manifest", str(manifest), "--dest", str(dest)]) == 0
    assert dest.is_dir() and not dest.is_symlink() and (dest / "ckpt" / "config.json").read_bytes() == b"{}"


def test_provision_refuses_bytes_that_do_not_match(hub, tmp_path):
    entry = _publish(hub, {"config.json": b"{}"})
    entry["files"]["config.json"][1] = "0" * 64
    with pytest.raises(RuntimeError, match="does not match"):
        provision.provision_one("ckpt", entry, tmp_path / "models")
    assert not (tmp_path / "models" / "ckpt" / "config.json").exists()
