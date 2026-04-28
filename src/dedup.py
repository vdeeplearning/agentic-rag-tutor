"""Helpers for tracking which files have already been indexed.

The app uses a SHA256 hash of each uploaded file. If the same file is uploaded
again, the hash matches and we can skip embedding its chunks a second time.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import DATA_DIR


INDEXED_FILES_PATH = DATA_DIR / "indexed_files.json"


def compute_file_hash(file_path: Path) -> str:
    """Compute a SHA256 hash from the file contents."""
    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def load_indexed_files() -> dict[str, Any]:
    """Load the local index registry from disk."""
    if not INDEXED_FILES_PATH.exists():
        return {}

    with INDEXED_FILES_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_indexed_files(data: dict[str, Any]) -> None:
    """Save the local index registry to disk."""
    INDEXED_FILES_PATH.parent.mkdir(parents=True, exist_ok=True)

    with INDEXED_FILES_PATH.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def is_file_indexed(file_hash: str) -> bool:
    """Return True when this exact file hash has already been indexed."""
    indexed_files = load_indexed_files()
    return file_hash in indexed_files


def mark_file_indexed(file_hash: str, filename: str) -> None:
    """Record that a file was successfully indexed."""
    indexed_files = load_indexed_files()
    indexed_files[file_hash] = {
        "filename": filename,
        "indexed_at": datetime.now(timezone.utc).isoformat(),
    }
    save_indexed_files(indexed_files)
