"""Phase 20G operational backup, integrity verification, and non-destructive restore drill tool.

Creates verified snapshots of staging runtime model weights, Compose/reverse-proxy configurations,
and non-secret staging templates without including research caches or raw datasets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

CRITICAL_RELATIVE_PATHS = [
    ".env.example",
    ".env.staging.example",
    "docker-compose.staging.yml",
    "Dockerfile.backend",
    "Dockerfile.frontend",
    "requirements.txt",
    "requirements-staging.txt",
    "api/config.py",
    "api/fusion.py",
    "deploy/caddy/Caddyfile",
    "deploy/nginx/default.conf",
    "deploy/nginx/security_headers.inc",
    "deploy/monitoring/prometheus.yml",
    "models/text_emotion/distilbert/best/config.json",
    "models/text_emotion/distilbert/best/model.safetensors",
    "models/text_emotion/distilbert/best/tokenizer.json",
    "models/text_emotion/distilbert/best/tokenizer_config.json",
    "models/speech_emotion/wav2vec2/best/config.json",
    "models/speech_emotion/wav2vec2/best/model.safetensors",
    "models/speech_emotion/wav2vec2/best/preprocessor_config.json",
    "models/facial_emotion/resnet18/exp5_ce_weighted_best.pt",
    "models/facial_emotion/resnet18/exp5_focal_g3_wt_best.pt",
    "models/facial_emotion/resnet18/manifest.json",
    "models/facial_emotion/facial_cnn_best.pt",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_backup(output_root: Path, label: str | None = None) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder_name = f"staging_snapshot_{stamp}" if not label else f"staging_snapshot_{label}_{stamp}"
    snapshot_dir = output_root / folder_name
    payload_dir = snapshot_dir / "payload"
    payload_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    total_bytes = 0
    for rel in CRITICAL_RELATIVE_PATHS:
        src = REPOSITORY_ROOT / rel
        if not src.is_file():
            raise FileNotFoundError(f"Required runtime/config asset missing: {src}")
        dst = payload_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        size = dst.stat().st_size
        total_bytes += size
        entries.append(
            {
                "relative_path": rel,
                "bytes": size,
                "sha256": _sha256(dst),
            }
        )

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "snapshot_dir": str(snapshot_dir),
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "files": entries,
    }
    (snapshot_dir / "backup_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return snapshot_dir


def verify_backup(snapshot_dir: Path) -> dict[str, Any]:
    manifest_path = snapshot_dir / "backup_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload_dir = snapshot_dir / "payload"

    verified = 0
    mismatches: list[str] = []
    for entry in manifest["files"]:
        rel = entry["relative_path"]
        target = payload_dir / rel
        if not target.is_file() or _sha256(target) != entry["sha256"]:
            mismatches.append(rel)
        else:
            verified += 1
    return {
        "snapshot_dir": str(snapshot_dir),
        "verified_files": verified,
        "mismatches": mismatches,
        "ok": len(mismatches) == 0,
    }


def restore_backup(snapshot_dir: Path, destination: Path) -> dict[str, Any]:
    check = verify_backup(snapshot_dir)
    if not check["ok"]:
        raise RuntimeError(f"Refusing restore from corrupted backup: {check['mismatches']}")
    manifest = json.loads((snapshot_dir / "backup_manifest.json").read_text(encoding="utf-8"))
    payload_dir = snapshot_dir / "payload"
    destination.mkdir(parents=True, exist_ok=True)

    restored = 0
    for entry in manifest["files"]:
        rel = entry["relative_path"]
        src = payload_dir / rel
        dst = destination / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        if _sha256(dst) != entry["sha256"]:
            raise RuntimeError(f"Post-restore SHA-256 verification failed on {dst}")
        restored += 1

    return {
        "restored_to": str(destination),
        "restored_files": restored,
        "ok": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Staging backup, verify, and restore drill helper.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_backup = sub.add_parser("backup", help="Create a verified staging snapshot.")
    p_backup.add_argument("--output-root", type=Path, default=REPOSITORY_ROOT / "backups")
    p_backup.add_argument("--label", type=str, default=None)

    p_verify = sub.add_parser("verify", help="Verify SHA-256 checksums inside an existing snapshot.")
    p_verify.add_argument("--snapshot-dir", type=Path, required=True)

    p_restore = sub.add_parser("restore", help="Restore snapshot into a target directory (non-destructive if pointed at drill dir).")
    p_restore.add_argument("--snapshot-dir", type=Path, required=True)
    p_restore.add_argument("--target-dir", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "backup":
        out = create_backup(args.output_root, args.label)
        print(json.dumps(verify_backup(out), indent=2))
        return 0
    if args.command == "verify":
        res = verify_backup(args.snapshot_dir)
        print(json.dumps(res, indent=2))
        return 0 if res["ok"] else 1
    if args.command == "restore":
        res = restore_backup(args.snapshot_dir, args.target_dir)
        print(json.dumps(res, indent=2))
        return 0 if res["ok"] else 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

