#!/usr/bin/env python3
"""Verify the archived V13 result files and checkpoints.

Run from any directory:
    python scripts/verify_v13_artifacts.py
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RESULT_DIR = REPO / "results" / "v13_token_mixer_lora_controls"
CHECKPOINT_DIR = REPO / "checkpoints" / "v13_controls"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest(manifest: Path, base: Path) -> tuple[int, list[str]]:
    checked = 0
    failures: list[str] = []
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            target = base / row["file"]
            if not target.is_file():
                failures.append(f"missing: {target.relative_to(REPO)}")
                continue
            actual_bytes = target.stat().st_size
            actual_digest = sha256(target)
            if actual_bytes != int(row["bytes"]):
                failures.append(
                    f"size mismatch: {target.relative_to(REPO)} "
                    f"({actual_bytes} != {row['bytes']})"
                )
            elif actual_digest != row["sha256"]:
                failures.append(f"SHA-256 mismatch: {target.relative_to(REPO)}")
            else:
                checked += 1
    return checked, failures


def main() -> None:
    result_count, result_failures = verify_manifest(
        RESULT_DIR / "result_file_manifest_sha256.csv", RESULT_DIR
    )
    checkpoint_count, checkpoint_failures = verify_manifest(
        RESULT_DIR / "checkpoint_manifest_sha256.csv", CHECKPOINT_DIR
    )
    failures = result_failures + checkpoint_failures
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)
    print(f"PASS: {result_count} V13 result files verified")
    print(f"PASS: {checkpoint_count} V13 checkpoints verified")


if __name__ == "__main__":
    main()
