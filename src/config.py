"""Shared project paths.

Keeping paths in one place makes the beginner version of the app easier to
follow and avoids hard-coding folders across multiple files.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma"
