#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Publish OtoDeger Business intelligence CSVs into the backend repository.

Canonical files remain in the Desktop/otodeger_intelligence folder.
Runtime copies are published into Desktop/car-valuation-backend.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

REQUIRED_FILES = (
    "business_stock_intelligence.csv",
    "business_company_intelligence.csv",
    "business_market_intelligence.csv",
    "business_company_activity_daily.csv",
)

DEFAULT_SOURCE_DIR = Path.home() / "Desktop" / "otodeger_intelligence"
DEFAULT_BACKEND_DIR = Path.home() / "Desktop" / "car-valuation-backend"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def validate_source_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required source file not found: {path}")
    if not path.is_file():
        raise RuntimeError(f"Expected a file but found something else: {path}")
    if path.stat().st_size <= 0:
        raise RuntimeError(f"Source file is empty: {path}")


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    os.close(fd)
    temp_path = Path(temp_name)

    try:
        shutil.copy2(source, temp_path)
        if temp_path.stat().st_size != source.stat().st_size:
            raise RuntimeError(f"Size verification failed for {source.name}")
        os.replace(temp_path, destination)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def publish(source_dir: Path, backend_dir: Path) -> int:
    source_dir = source_dir.resolve()
    backend_dir = backend_dir.resolve()

    print(f"Source:  {source_dir}")
    print(f"Backend: {backend_dir}")
    print()

    if not source_dir.exists():
        print(f"ERROR: Source directory does not exist: {source_dir}", file=sys.stderr)
        return 1
    if not backend_dir.exists():
        print(f"ERROR: Backend directory does not exist: {backend_dir}", file=sys.stderr)
        return 1

    try:
        for filename in REQUIRED_FILES:
            validate_source_file(source_dir / filename)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    copied = 0
    unchanged = 0

    for filename in REQUIRED_FILES:
        source = source_dir / filename
        destination = backend_dir / filename
        try:
            source_hash = sha256_file(source)
            if destination.exists() and destination.is_file():
                if sha256_file(destination) == source_hash:
                    print(f"UNCHANGED  {filename}")
                    unchanged += 1
                    continue

            atomic_copy(source, destination)

            if sha256_file(destination) != source_hash:
                raise RuntimeError(f"Hash verification failed after copying {filename}")

            print(f"UPDATED    {filename} ({source.stat().st_size:,} bytes)")
            copied += 1
        except Exception as exc:
            print(f"ERROR: Failed to publish {filename}: {exc}", file=sys.stderr)
            return 1

    print()
    print(
        f"Publish complete: {copied} updated, "
        f"{unchanged} unchanged, {len(REQUIRED_FILES)} checked."
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish OtoDeger Business intelligence files to the backend repo."
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--backend-dir", type=Path, default=DEFAULT_BACKEND_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return publish(args.source_dir, args.backend_dir)


if __name__ == "__main__":
    raise SystemExit(main())
