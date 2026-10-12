#!/usr/bin/env python3
"""Fetch the pinned public checkpoints the GPU and oracle tiers read, verifying every byte.

    python tools/ci/provision.py --dest <dir> [--only NAME ...] [--rehash]
    python tools/ci/provision.py --pin NAME=OWNER/REPO[@REVISION] ...     # add or refresh manifest entries

The manifest (``tools/ci/models.json``) names each checkpoint by the directory the tests look it up under
(``$MONOLITH_MODELS/<name>``, tests/oracle_conftest.py), its Hub repository, the commit it is pinned to, and every
file with its size and SHA-256. A file is downloaded from that commit only, into ``<file>.part`` (resumed when a
previous attempt left one), and renamed into place once its digest matches. A file already in place is re-hashed
unless its size and mtime match the record of its last verification (``--rehash`` always re-hashes). Files in a
checkpoint directory that the manifest does not list are removed, so a test cannot read a stale shard.

No token is ever sent: CI reads public repositories only. ``HF_ENDPOINT`` selects a mirror.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

MANIFEST = Path(__file__).resolve().with_name("models.json")
STAMP = ".provision.json"
CHUNK = 8 << 20
# The files a checkpoint directory needs: weights, their index, configs and the tokenizer; never READMEs, ONNX or
# training logs.
KEEP = re.compile(r"^(.*\.safetensors|.*\.safetensors\.index\.json|config\.json|generation_config\.json|"
                  r"hf_quant_config\.json|.*preprocessor_config\.json|tokenizer.*\.(json|model)|vocab\.json|"
                  r"merges\.txt|special_tokens_map\.json|added_tokens\.json|chat_template\.jinja)$")


def endpoint() -> str:
    return os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def _open(url: str, headers: Optional[Dict[str, str]] = None, timeout: float = 60.0):
    request = urllib.request.Request(url, headers={"User-Agent": "lithos-metal-ci", **(headers or {})})
    return urllib.request.urlopen(request, timeout=timeout)


def load_manifest(path: Path = MANIFEST) -> Dict[str, dict]:
    return json.loads(path.read_text())


def validate_manifest(manifest: Dict[str, dict]) -> List[str]:
    """Return the problems with a manifest (empty = valid)."""
    problems = []
    for name, entry in manifest.items():
        if "/" in name or name.startswith("."):
            problems.append(f"{name}: a checkpoint name is one directory name")
        if not re.fullmatch(r"[0-9a-f]{40}", entry.get("revision", "")):
            problems.append(f"{name}: revision must be a full commit hash")
        files = entry.get("files") or {}
        if not files:
            problems.append(f"{name}: no files")
        for rel, (size, digest) in files.items():
            if rel.startswith("/") or ".." in Path(rel).parts:
                problems.append(f"{name}: {rel}: not a relative path inside the checkpoint")
            if not isinstance(size, int) or size < 0 or not re.fullmatch(r"[0-9a-f]{64}", digest):
                problems.append(f"{name}: {rel}: size and SHA-256 required")
    return problems


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(CHUNK)
            if not block:
                return h.hexdigest()
            h.update(block)


def _download(url: str, part: Path, size: int, attempts: int = 5) -> None:
    for attempt in range(1, attempts + 1):
        have = part.stat().st_size if part.exists() else 0
        if have > size:
            part.unlink()
            have = 0
        if have == size:
            return
        try:
            with _open(url, {"Range": f"bytes={have}-"} if have else None) as r:
                if have and r.status != 206:          # the server ignored the range: start over
                    have = 0
                with open(part, "ab" if have else "wb") as f:
                    shutil.copyfileobj(r, f, CHUNK)
            if part.stat().st_size == size:
                return
        except (urllib.error.URLError, OSError) as e:
            print(f"  attempt {attempt}: {e}", flush=True)
        time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"{url}: could not download {size} bytes")


def _read_stamp(directory: Path) -> Dict[str, list]:
    try:
        return json.loads((directory / STAMP).read_text())
    except (OSError, ValueError):
        return {}


def _write_stamp(directory: Path, stamp: Dict[str, list]) -> None:
    tmp = directory / (STAMP + ".tmp")
    tmp.write_text(json.dumps(stamp, indent=1, sort_keys=True))
    os.replace(tmp, directory / STAMP)


def _real_directory(path: Path) -> None:
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        path.unlink()
    path.mkdir(parents=True, exist_ok=True)


def _clear_for_file(directory: Path, target: Path) -> None:
    """Free ``target``'s path for a regular file: anything else there (a directory, a FIFO) and a file where one of its
    parent directories belongs are removed (a test could leave either)."""
    path = directory
    for part in target.relative_to(directory).parts[:-1]:
        path = path / part
        if path.exists() and not path.is_dir():
            path.unlink()
    if target.is_dir():
        shutil.rmtree(target)
    elif target.exists() and not target.is_file():
        target.unlink()


def provision_one(name: str, entry: dict, dest: Path, rehash: bool = False) -> Tuple[int, int]:
    """Make ``dest/name`` hold exactly the manifest's files. Returns (files downloaded, bytes downloaded)."""
    # The tests run as this user and could leave symlinks here: remove every link (never what it points at) before
    # anything is read or pruned, so nothing outside the cache is ever touched.
    directory = dest / name
    _real_directory(dest)
    _real_directory(directory)
    for root, dirs, names in os.walk(directory):          # does not descend into a symlinked directory
        for entry_name in dirs + names:
            if (Path(root) / entry_name).is_symlink():
                (Path(root) / entry_name).unlink()
    for control in (STAMP, STAMP + ".tmp"):
        _clear_for_file(directory, directory / control)
    stamp = _read_stamp(directory)
    files = entry["files"]
    for path in sorted(directory.rglob("*")):     # files the manifest does not list (a partial download resumes)
        rel = path.relative_to(directory).as_posix()
        if path.is_dir():
            continue
        if not path.is_file() or (rel not in files and rel != STAMP and rel.removesuffix(".part") not in files):
            path.unlink()
            stamp.pop(rel, None)
    fetched = fetched_bytes = 0
    for rel, (size, digest) in sorted(files.items()):
        target = directory / rel
        part = target.with_name(target.name + ".part")
        _clear_for_file(directory, target)
        _clear_for_file(directory, part)
        if target.exists():
            st = target.stat()
            if not rehash and stamp.get(rel) == [st.st_size, st.st_mtime_ns, digest]:
                continue
            if st.st_size == size and sha256_file(target) == digest:
                stamp[rel] = [st.st_size, st.st_mtime_ns, digest]
                continue
            target.unlink()
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"{endpoint()}/{entry['repo']}/resolve/{entry['revision']}/{urllib.parse.quote(rel)}"
        print(f"{name}: {rel} ({size / 1e9:.2f} GB)", flush=True)
        _download(url, part, size)
        got = sha256_file(part)
        if got != digest:
            part.unlink()
            raise RuntimeError(f"{name}: {rel}: SHA-256 {got} does not match the manifest's {digest}")
        os.replace(part, target)
        st = target.stat()
        stamp[rel] = [st.st_size, st.st_mtime_ns, digest]
        fetched, fetched_bytes = fetched + 1, fetched_bytes + size
        _write_stamp(directory, stamp)
    _write_stamp(directory, stamp)
    return fetched, fetched_bytes


def missing_bytes(manifest: Dict[str, dict], dest: Path, names: Iterable[str]) -> int:
    total = 0
    for name in names:
        for rel, (size, _) in manifest[name]["files"].items():
            p = dest / name / rel
            total += 0 if p.exists() and p.stat().st_size == size else size
    return total


def pin(spec: str) -> Tuple[str, dict]:
    """``NAME=OWNER/REPO[@REVISION]`` -> a manifest entry for that commit (small files are fetched to hash them)."""
    name, _, repo = spec.partition("=")
    repo, _, revision = repo.partition("@")
    with _open(f"{endpoint()}/api/models/{repo}/revision/{revision or 'main'}?blobs=true") as r:
        info = json.load(r)
    files = {}
    for sibling in info["siblings"]:
        rel = sibling["rfilename"]
        if "/" in rel or not KEEP.match(rel):
            continue
        lfs = sibling.get("lfs") or {}
        if lfs.get("sha256"):
            files[rel] = [int(lfs["size"]), lfs["sha256"]]
        else:
            with _open(f"{endpoint()}/{repo}/resolve/{info['sha']}/{urllib.parse.quote(rel)}") as r:
                data = r.read()
            files[rel] = [len(data), hashlib.sha256(data).hexdigest()]
    return name, {"repo": repo, "revision": info["sha"], "files": dict(sorted(files.items()))}


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--dest", type=Path, help="directory to provision; the tests read it as $MONOLITH_MODELS")
    ap.add_argument("--only", nargs="*", help="provision only these checkpoints")
    ap.add_argument("--rehash", action="store_true", help="re-hash every file, ignoring the verification record")
    ap.add_argument("--pin", nargs="*", metavar="NAME=OWNER/REPO[@REVISION]", help="add or refresh manifest entries")
    args = ap.parse_args(argv)
    manifest = load_manifest(args.manifest) if args.manifest.exists() else {}
    if args.pin:
        for spec in args.pin:
            name, entry = pin(spec)
            manifest[name] = entry
            print(f"pinned {name} = {entry['repo']}@{entry['revision']} ({len(entry['files'])} files)")
        args.manifest.write_text(json.dumps(dict(sorted(manifest.items())), indent=2) + "\n")
        return 0
    if args.dest is None:
        ap.error("--dest is required")
    problems = validate_manifest(manifest)
    if problems:
        print("provision: invalid manifest\n  " + "\n  ".join(problems))
        return 1
    names = args.only or sorted(manifest)
    unknown = [n for n in names if n not in manifest]
    if unknown:
        ap.error(f"not in the manifest: {', '.join(unknown)}")
    _real_directory(args.dest)          # a file or symlink a test left there is replaced, never followed
    need = missing_bytes(manifest, args.dest, names)
    free = shutil.disk_usage(args.dest).free
    if need + (2 << 30) > free:
        print(f"provision: {need / 1e9:.1f} GB to download but only {free / 1e9:.1f} GB free under {args.dest}")
        return 1
    t0 = time.time()
    total_files = total_bytes = 0
    for name in names:
        n, b = provision_one(name, manifest[name], args.dest, rehash=args.rehash)
        total_files, total_bytes = total_files + n, total_bytes + b
    print(f"provision: {len(names)} checkpoints verified under {args.dest}; downloaded {total_files} files, "
          f"{total_bytes / 1e9:.2f} GB in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
