
import os
from pathlib import Path


def get_secret(name: str, default: str | None = None) -> str | None:
    file_path = os.environ.get(name)
    if file_path:
        p = Path(file_path)
        if p.exists():
            return p.read_text().strip()
    return os.environ.get(name, default)