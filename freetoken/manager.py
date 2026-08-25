"""Model-manager CLI: list / show / verify / prepare / gen-manifest.

    python3 -m freetoken.manager list
    python3 -m freetoken.manager show qwen3.5-healthcare-bf16
    python3 -m freetoken.manager verify qwen3.5-healthcare-bf16
    python3 -m freetoken.manager prepare qwen3.5-healthcare-4bit
    python3 -m freetoken.manager gen-manifest --write [--force]

gen-manifest computes SHA-256 and byte sizes from the actual local artifacts;
checksums are never invented. No shell commands are ever invoked.
"""

import argparse
import dataclasses
import json
import os
import sys

from .manifest import (MANIFEST_SCHEMA_VERSION, Manifest, ModelEntry,
                       ManifestError, dump_manifest)
from .store import ModelStore, StoreError, prepare, sha256_file

DEFAULT_MANIFEST = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models.manifest.json")


def default_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Metadata template; sizes/checksums are filled in from the real artifacts.
KNOWN_MODELS = [
    dict(
        id="qwen3.5-healthcare-bf16",
        revision="m0-conv-20260824",
        family="qwen3.5-healthcare",
        format="mlx",
        dtype="bfloat16",
        path="m0_spike/mlx_models/qwen3.5-healthcare-bf16",
        weights_file="model.safetensors",
        license="Apache-2.0 (upstream Qwen3.5-2B-Base)",
        research_only=True,
        notes=("Research and development only; not for clinical diagnosis or "
               "treatment. Converted in M0 from "
               "training_material/qwen3.5_healthcare/model; mlx-lm drops the "
               "vision tower and MTP head. Fidelity reference model."),
        context_limit=262144,
        expected_resident_mb=3764,
        prefill_curve="bf16-m1",
        quant=None,
        url=None,
    ),
    dict(
        id="qwen3.5-healthcare-4bit",
        revision="m1-quant-20260824",
        family="qwen3.5-healthcare",
        format="mlx",
        dtype="4bit",
        path="m1_calibration/mlx_models/qwen3.5-healthcare-4bit",
        weights_file="model.safetensors",
        license="Apache-2.0 (upstream Qwen3.5-2B-Base)",
        research_only=True,
        notes=("Research and development only; not for clinical diagnosis or "
               "treatment. 4-bit affine quantization (5.003 bits/weight) of the "
               "same checkpoint, produced in M1. KV cache stays bf16; prefill "
               "is ~2x slower than bf16 on this hybrid architecture."),
        context_limit=262144,
        expected_resident_mb=1177,
        prefill_curve="q4-m1",
        quant={"bits": 4, "group_size": 32, "mode": "affine"},
        url=None,
    ),
]


def _filled_entry(root: str, template: dict) -> ModelEntry:
    t = dict(template)
    weights = os.path.join(root, t["path"], t["weights_file"])
    if not os.path.isfile(weights):
        raise StoreError("missing", f"artifact not found: {weights}")
    t["size_bytes"] = os.path.getsize(weights)
    t["sha256"] = sha256_file(weights)
    return ModelEntry(**t)


def cmd_gen_manifest(args) -> int:
    root = os.path.realpath(args.root)
    entries = []
    for template in KNOWN_MODELS:
        entry = _filled_entry(root, template)
        entries.append(entry)
        print(f"{entry.id}: size={entry.size_bytes} sha256={entry.sha256}")
    manifest = Manifest(schema_version=MANIFEST_SCHEMA_VERSION, models=entries)
    if args.write:
        dump_manifest(manifest, args.out, force=args.force)
        print(f"wrote {args.out}")
    else:
        print(json.dumps({
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "models": [dataclasses.asdict(m) for m in manifest.models],
        }, indent=2))
    return 0


def _store(args) -> ModelStore:
    return ModelStore(args.root, args.manifest)


def cmd_list(args) -> int:
    store = _store(args)
    print(json.dumps(store.list_models(), indent=2))
    return 0


def cmd_show(args) -> int:
    store = _store(args)
    entry = store.get(args.model_id)
    d = dataclasses.asdict(entry)
    print(json.dumps(d, indent=2))
    return 0


def cmd_verify(args) -> int:
    store = _store(args)
    entry = store.get(args.model_id)
    vr = store.verify(args.model_id)
    print(json.dumps({
        "model": entry.id,
        "ok": vr.ok,
        "exists": vr.exists,
        "size_ok": vr.size_ok,
        "expected_size": entry.size_bytes,
        "actual_size": vr.actual_size,
        "sha256_ok": vr.sha256_ok,
        "expected_sha256": entry.sha256,
        "actual_sha256": vr.actual_sha256,
    }, indent=2))
    return 0 if vr.ok else 1


def cmd_prepare(args) -> int:
    store = _store(args)
    report = prepare(store, args.model_id, allow_download=args.allow_download)
    print(json.dumps(report, indent=2))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="freetoken.manager")
    p.add_argument("--root", default=default_root(),
                   help="workspace root that model paths are relative to")
    p.add_argument("--manifest", default=DEFAULT_MANIFEST)
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list manifest models").set_defaults(fn=cmd_list)

    sp = sub.add_parser("show", help="show one manifest entry")
    sp.add_argument("model_id")
    sp.set_defaults(fn=cmd_show)

    sp = sub.add_parser("verify", help="verify exact size + SHA-256 of an artifact")
    sp.add_argument("model_id")
    sp.set_defaults(fn=cmd_verify)

    sp = sub.add_parser("prepare", help="run the prepare pipeline for a model")
    sp.add_argument("model_id")
    sp.add_argument("--allow-download", action="store_true",
                    help="permit download only if the entry has a pinned URL")
    sp.set_defaults(fn=cmd_prepare)

    sp = sub.add_parser("gen-manifest",
                        help="compute real sizes/checksums and emit the manifest")
    sp.add_argument("--write", action="store_true",
                    help=f"write to {DEFAULT_MANIFEST}")
    sp.add_argument("--out", default=DEFAULT_MANIFEST)
    sp.add_argument("--force", action="store_true",
                    help="overwrite an existing manifest")
    sp.set_defaults(fn=cmd_gen_manifest)

    args = p.parse_args(argv)
    try:
        return args.fn(args)
    except (StoreError, ManifestError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
