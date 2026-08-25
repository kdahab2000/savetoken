"""Versioned model manifest: schema, validation, and safe path resolution.

The manifest is data (models.manifest.json, schema version below). Every
security-sensitive field is validated at load time; paths are always resolved
relative to the workspace root and may never escape it.
"""

import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import List, Optional

MANIFEST_SCHEMA_VERSION = 1

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
FORMATS = ("mlx", "hf-safetensors")
DTYPES = ("bfloat16", "float16", "float32", "4bit", "8bit")
CURVES = ("bf16-m1", "q4-m1")


class ManifestError(ValueError):
    pass


@dataclass
class ModelEntry:
    id: str
    revision: str
    family: str
    format: str                      # "mlx" | "hf-safetensors"
    dtype: str                       # bfloat16 | 4bit | ...
    path: str                        # relative to workspace root
    weights_file: str                # file inside `path` that carries the weights
    size_bytes: int                  # exact byte size of weights_file
    sha256: str                      # SHA-256 of weights_file (hex, lowercase)
    license: str
    research_only: bool
    notes: str
    context_limit: int
    expected_resident_mb: int        # M1-measured resident weight memory
    prefill_curve: str = "bf16-m1"   # key into capacity.PREFILL_CURVES
    quant: Optional[dict] = None     # e.g. {"bits": 4, "group_size": 32, "mode": "affine"}
    url: Optional[str] = None        # optional pinned download URL (https only)

    def validate(self) -> None:
        if not _ID_RE.match(self.id or ""):
            raise ManifestError(f"invalid model id {self.id!r}")
        if not _SAFE_LABEL_RE.match(self.revision or ""):
            raise ManifestError(f"invalid revision {self.revision!r}")
        if not _SAFE_LABEL_RE.match(self.family or ""):
            raise ManifestError(f"invalid family {self.family!r}")
        if self.format not in FORMATS:
            raise ManifestError(f"invalid format {self.format!r}")
        if self.dtype not in DTYPES:
            raise ManifestError(f"invalid dtype {self.dtype!r}")
        _validate_relpath(self.path, "path")
        _validate_relpath(self.weights_file, "weights_file")
        if not isinstance(self.size_bytes, int) or self.size_bytes <= 0:
            raise ManifestError("size_bytes must be a positive integer")
        if not _SHA256_RE.match(self.sha256 or ""):
            raise ManifestError("sha256 must be 64 lowercase hex chars")
        if not self.license:
            raise ManifestError("license is required")
        if not isinstance(self.context_limit, int) or self.context_limit <= 0:
            raise ManifestError("context_limit must be positive")
        if (not isinstance(self.expected_resident_mb, int)
                or self.expected_resident_mb <= 0):
            raise ManifestError("expected_resident_mb must be positive")
        if self.prefill_curve not in CURVES:
            raise ManifestError(f"unknown prefill_curve {self.prefill_curve!r}")
        if self.url is not None and not self.url.startswith("https://"):
            raise ManifestError("url must be https:// (pinned source)")
        if self.quant is not None and not isinstance(self.quant, dict):
            raise ManifestError("quant must be an object")


def _validate_relpath(rel: str, field_name: str) -> None:
    if not isinstance(rel, str) or not rel:
        raise ManifestError(f"{field_name} is required")
    if "\x00" in rel:
        raise ManifestError(f"{field_name} contains a NUL byte")
    if os.path.isabs(rel) or rel.startswith("~"):
        raise ManifestError(f"{field_name} must be relative: {rel!r}")
    parts = rel.replace("\\", "/").split("/")
    if any(p in ("", "..") for p in parts):
        raise ManifestError(f"{field_name} contains empty or '..' components: {rel!r}")


def safe_join(root: str, rel: str) -> str:
    """Join `rel` onto `root`, refusing any escape (traversal, absolute paths,
    symlink exits). Returns an absolute path inside root."""
    _validate_relpath(rel, "path")
    root = os.path.realpath(root)
    full = os.path.realpath(os.path.join(root, rel))
    if full != root and not full.startswith(root + os.sep):
        raise ManifestError(f"path escapes workspace root: {rel!r}")
    return full


@dataclass
class Manifest:
    schema_version: int
    models: List[ModelEntry] = field(default_factory=list)

    def get(self, model_id: str) -> ModelEntry:
        for m in self.models:
            if m.id == model_id:
                return m
        raise KeyError(f"unknown model id {model_id!r}")

    def ids(self) -> List[str]:
        return [m.id for m in self.models]


def load_manifest(path: str) -> Manifest:
    with open(path, "r") as f:
        raw = json.load(f)
    version = raw.get("schema_version")
    if version != MANIFEST_SCHEMA_VERSION:
        raise ManifestError(
            f"unsupported manifest schema_version {version!r} "
            f"(expected {MANIFEST_SCHEMA_VERSION})"
        )
    entries = []
    seen = set()
    for item in raw.get("models", []):
        try:
            entry = ModelEntry(**item)
        except TypeError as e:
            raise ManifestError(f"unknown manifest field: {e}")
        entry.validate()
        if entry.id in seen:
            raise ManifestError(f"duplicate model id {entry.id!r}")
        seen.add(entry.id)
        entries.append(entry)
    if not entries:
        raise ManifestError("manifest contains no models")
    return Manifest(schema_version=version, models=entries)


def dump_manifest(manifest: Manifest, path: str, force: bool = False) -> None:
    """Write the manifest atomically; refuses to overwrite without force."""
    if os.path.exists(path) and not force:
        raise ManifestError(
            f"manifest {path} already exists; refusing to overwrite without --force"
        )
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "models": [asdict(m) for m in manifest.models],
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)  # atomic on POSIX
