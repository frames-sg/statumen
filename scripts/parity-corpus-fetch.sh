#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'USAGE'
usage: scripts/parity-corpus-fetch.sh [alias ...]

Downloads public parity-corpus slides from tests/fixtures/parity_corpus.public.toml.
Files are written to $STATUMEN_PARITY_CORPUS_CACHE, or to
~/.cache/slideviewer/parity-corpus when the variable is unset.

Pass one or more aliases such as svs-001 or dicom-jp2k-001 to fetch only those
slides. Set STATUMEN_PARITY_CORPUS_FORCE=1 to re-extract zip archives.
USAGE
  exit 0
fi

python3 - "$repo_root" "$@" <<'PY'
from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError as exc:
    raise SystemExit("python 3.11+ with tomllib is required") from exc

FORMAT_EXTENSIONS = {
    "aperio": "svs",
    "leica": "scn",
    "ventana": "bif",
    "philips_tiff": "tif",
    "tiff": "tif",
    "ndpi": "ndpi",
    "hamamatsu_vms": "zip",
    "dicom": "dcm",
    "mirax": "zip",
}


def main() -> int:
    repo_root = Path(sys.argv[1])
    aliases = {arg for arg in sys.argv[2:] if arg}
    manifest_path = repo_root / "tests" / "fixtures" / "parity_corpus.public.toml"
    with manifest_path.open("rb") as handle:
        manifest = tomllib.load(handle)

    cache_dir = Path(
        os.environ.get(
            "STATUMEN_PARITY_CORPUS_CACHE",
            Path.home() / ".cache" / "slideviewer" / "parity-corpus",
        )
    ).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    force_extract = truthy(os.environ.get("STATUMEN_PARITY_CORPUS_FORCE"))

    slides = manifest.get("slide", [])
    selected = [
        entry for entry in slides if not aliases or entry.get("alias") in aliases
    ]
    if aliases:
        found = {entry.get("alias") for entry in selected}
        missing = sorted(aliases - found)
        if missing:
            raise SystemExit(f"unknown corpus aliases: {', '.join(missing)}")

    for entry in selected:
        fetch_entry(entry, cache_dir, force_extract)
    return 0


def fetch_entry(entry: dict[str, object], cache_dir: Path, force_extract: bool) -> None:
    alias = str(entry["alias"])
    url = str(entry.get("url", ""))
    expected_sha = str(entry.get("sha256", ""))
    if not url:
        print(f"[skip] {alias}: no URL")
        return

    target = target_path(entry, cache_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and (not expected_sha or sha256_file(target) == expected_sha):
        print(f"[ok]   {alias}: {target}")
    else:
        download(url, target)
        if expected_sha:
            actual_sha = sha256_file(target)
            if actual_sha != expected_sha:
                target.unlink(missing_ok=True)
                raise SystemExit(
                    f"{alias}: sha256 mismatch for {target}: "
                    f"expected {expected_sha}, got {actual_sha}"
                )
        print(f"[get]  {alias}: {target}")

    if target.suffix.lower() == ".zip":
        extract_zip(alias, target, cache_dir / f"{alias}.d", force_extract)


def target_path(entry: dict[str, object], cache_dir: Path) -> Path:
    alias = str(entry["alias"])
    url = str(entry.get("url", ""))
    url_name = url.rsplit("/", 1)[-1]
    url_suffix = Path(url_name).suffix.lower()
    if url_suffix == ".zip":
        return cache_dir / f"{alias}.zip"

    ext = FORMAT_EXTENSIONS.get(str(entry.get("format", "")))
    if not ext and url_suffix:
        ext = url_suffix.lstrip(".")
    return cache_dir / (f"{alias}.{ext}" if ext else alias)


def download(url: str, target: Path) -> None:
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with urllib.request.urlopen(url) as response, tmp_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        tmp_path.replace(target)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def extract_zip(alias: str, archive: Path, extract_dir: Path, force_extract: bool) -> None:
    if extract_dir.is_dir() and not force_extract:
        print(f"[ok]   {alias}: {extract_dir}")
        return

    tmp_dir = extract_dir.with_name(f".{extract_dir.name}.tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    tmp_dir.mkdir(parents=True)
    try:
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(tmp_dir)
        tmp_dir.replace(extract_dir)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    print(f"[ext]  {alias}: {extract_dir}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def truthy(value: str | None) -> bool:
    return value is not None and value.lower() in {"1", "true", "yes", "on"}


raise SystemExit(main())
PY
