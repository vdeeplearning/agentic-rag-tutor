"""Shared project paths.

Keeping paths in one place makes the beginner version of the app easier to
follow and avoids hard-coding folders across multiple files.
"""

from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma"


def get_openai_api_key(ui_key: str | None = None) -> str | None:
    """Return the OpenAI API key from the UI first, then from .env.

    The Streamlit UI key is only kept in session_state and should be passed into
    this function by callers. If no UI key is available, local development can
    still use OPENAI_API_KEY from a .env file.
    """
    if ui_key and ui_key.strip():
        return ui_key.strip()

    load_dotenv()

    import os

    env_key = os.getenv("OPENAI_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()

    return None
