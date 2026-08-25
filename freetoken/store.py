"""ModelStore: list/inspect/select/verify/prepare local model artifacts.

Prepare pipeline stages (explicit, in order):
  locate    — find the artifact inside the workspace
  download  — only if the entry has a pinned https URL and it is missing,
              and only when explicitly allowed; writes to <name>.partial
  verify    — exact byte size + streaming SHA-256; never mutates artifacts
  convert   — HF safetensors → MLX, only when format requires it
              (in-process mlx_lm.convert; no shell is ever invoked)
  finalize  — resolve a path the engine can load

No telemetry, no logging of model content, no network unless an entry
carries a pinned URL and downloads are explicitly enabled.
"""

import hashlib
import os
import shutil
import urllib.request
from dataclasses import dataclass
from typing import List, Optional

from .manifest import Manifest, ManifestError, ModelEntry, load_manifest, safe_join

_HASH_CHUNK = 8 * 1024 * 1024


class StoreError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class VerifyResult:
    ok: bool
    exists: bool
    size_ok: bool
    sha256_ok: bool
    actual_size: Optional[int] = None
    actual_sha256: Optional[str] = None


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_HASH_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


class ModelStore:
    def __init__(self, root: str, manifest_path: Optional[str] = None):
        self.root = os.path.realpath(root)
        if manifest_path is None:
            manifest_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "models.manifest.json")
        self.manifest_path = manifest_path
        self.manifest: Manifest = load_manifest(manifest_path)

    # ---- queries -----------------------------------------------------
    def list_models(self) -> List[dict]:
        return [self.summary(m) for m in self.manifest.models]

    def summary(self, entry: ModelEntry) -> dict:
        return {
            "id": entry.id,
            "revision": entry.revision,
            "family": entry.family,
            "format": entry.format,
            "dtype": entry.dtype,
            "context_limit": entry.context_limit,
            "expected_resident_mb": entry.expected_resident_mb,
            "size_bytes": entry.size_bytes,
            "research_only": entry.research_only,
            "license": entry.license,
            "path": entry.path,
        }

    def get(self, model_id: str) -> ModelEntry:
        try:
            return self.manifest.get(model_id)
        except KeyError:
            raise StoreError(
                "unknown_model",
                f"unknown model id {model_id!r}; known: {self.manifest.ids()}",
            )

    def select(self, model_id: str) -> str:
        """Validate the id and return the artifact directory (absolute)."""
        entry = self.get(model_id)
        return safe_join(self.root, entry.path)

    # ---- verification --------------------------------------------------
    def weights_path(self, entry: ModelEntry) -> str:
        return safe_join(self.root, os.path.join(entry.path, entry.weights_file))

    def verify(self, model_id: str) -> VerifyResult:
        entry = self.get(model_id)
        path = self.weights_path(entry)
        if not os.path.isfile(path):
            return VerifyResult(ok=False, exists=False, size_ok=False, sha256_ok=False)
        actual_size = os.path.getsize(path)
        size_ok = actual_size == entry.size_bytes
        # Hash even when the size mismatches: the report is more useful, and a
        # mismatched file must never be accepted on any axis.
        actual_sha = sha256_file(path)
        sha_ok = actual_sha == entry.sha256
        return VerifyResult(
            ok=size_ok and sha_ok,
            exists=True,
            size_ok=size_ok,
            sha256_ok=sha_ok,
            actual_size=actual_size,
            actual_sha256=actual_sha,
        )

    # ---- download (only for pinned-URL entries, explicitly enabled) ----
    def download(self, entry: ModelEntry, allow_download: bool) -> None:
        dest_dir = safe_join(self.root, entry.path)
        dest = os.path.join(dest_dir, entry.weights_file)
        if os.path.exists(dest):
            return
        if not entry.url:
            raise StoreError(
                "not_available",
                f"model {entry.id!r} is not present locally and has no pinned URL",
            )
        if not allow_download:
            raise StoreError(
                "download_disabled",
                f"model {entry.id!r} is missing locally; re-run with downloads "
                "explicitly enabled to fetch the pinned URL",
            )
        os.makedirs(dest_dir, exist_ok=True)
        partial = dest + ".partial"
        if os.path.exists(partial):
            os.unlink(partial)  # never resume an unverified partial blob
        try:
            with urllib.request.urlopen(entry.url) as resp, open(partial, "wb") as f:
                shutil.copyfileobj(resp, f, _HASH_CHUNK)
        except Exception:
            # partial downloads are untrusted; remove and report
            if os.path.exists(partial):
                os.unlink(partial)
            raise StoreError("download_failed",
                             f"download of {entry.url!r} failed")
        finalize_download(partial, dest, entry.size_bytes, entry.sha256)


def finalize_download(partial: str, dest: str, expected_size: int,
                      expected_sha: str) -> None:
    """Atomically promote a downloaded .partial file: verify first, rename
    second. Refuses to overwrite an existing artifact."""
    if os.path.exists(dest):
        raise StoreError("exists", f"refusing to overwrite existing {dest!r}")
    if os.path.getsize(partial) != expected_size:
        raise StoreError("size_mismatch",
                         "downloaded file size does not match the manifest")
    if sha256_file(partial) != expected_sha:
        raise StoreError("checksum_mismatch",
                         "downloaded file checksum does not match the manifest")
    os.replace(partial, dest)


# ---- prepare pipeline ----------------------------------------------------
STAGES = ("locate", "download", "verify", "convert", "finalize")


def prepare(store: ModelStore, model_id: str, allow_download: bool = False,
            convert_output_root: Optional[str] = None) -> dict:
    entry = store.get(model_id)
    report = {"model": model_id, "stages": {}, "path": None}

    # locate
    artifact_dir = safe_join(store.root, entry.path)
    weights = os.path.join(artifact_dir, entry.weights_file)
    present = os.path.isfile(weights)
    report["stages"]["locate"] = "found" if present else "missing"

    # download (only for missing pinned-URL entries, explicitly allowed)
    if not present:
        store.download(entry, allow_download)
        report["stages"]["download"] = "done"
    else:
        report["stages"]["download"] = "skipped"

    # verify (never mutates artifacts)
    vr = store.verify(model_id)
    report["stages"]["verify"] = "ok" if vr.ok else "failed"
    if not vr.ok:
        if not vr.exists:
            raise StoreError("missing", f"weights file missing for {model_id!r}")
        if not vr.size_ok:
            raise StoreError(
                "size_mismatch",
                f"{model_id}: expected {entry.size_bytes} bytes, found "
                f"{vr.actual_size}; artifact left untouched",
            )
        raise StoreError(
            "checksum_mismatch",
            f"{model_id}: SHA-256 mismatch (expected {entry.sha256}, found "
            f"{vr.actual_sha256}); artifact left untouched",
        )

    # convert (only HF artifacts; in-process, never via shell)
    if entry.format == "hf-safetensors":
        out_dir = _convert_stage(store, entry, convert_output_root)
        report["stages"]["convert"] = "done"
        report["path"] = out_dir
    else:
        report["stages"]["convert"] = "skipped"
        report["path"] = artifact_dir

    # finalize
    report["stages"]["finalize"] = "ok"
    return report


def _convert_stage(store: ModelStore, entry: ModelEntry,
                   output_root: Optional[str]) -> str:
    from mlx_lm import convert as mlx_convert  # imported lazily; Python API only

    src = safe_join(store.root, entry.path)
    out_rel = os.path.join("converted", entry.id)
    out_dir = safe_join(output_root or store.root, out_rel)
    if os.path.exists(out_dir):
        raise StoreError("exists",
                         f"refusing to overwrite existing converted dir {out_dir!r}")
    quant = entry.quant or {}
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
    mlx_convert(
        hf_path=src,
        mlx_path=out_dir,
        quantize=bool(quant),
        q_bits=quant.get("bits"),
        q_group_size=quant.get("group_size"),
        q_mode=quant.get("mode", "affine"),
    )
    # verify the converted artifact structurally (no invented checksums: the
    # converted tree is recorded only after load-level checks in M0/M1 flows)
    produced = os.path.join(out_dir, "model.safetensors")
    if not (os.path.isfile(produced) and os.path.isfile(
            os.path.join(out_dir, "config.json"))):
        raise StoreError("convert_failed",
                         f"conversion did not produce expected files in {out_dir!r}")
    return out_dir
