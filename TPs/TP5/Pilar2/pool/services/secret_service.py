import os
from pathlib import Path


def get_secret(name: str, default: str | None = None) -> str | None:
    """Return secret value preferring file-based injection."""
    file_path = os.environ.get(f"{name}_FILE")
    if file_path:
        p = Path(file_path)
        if p.exists():
            return p.read_text().strip()

    env_val = os.environ.get(name)
    if env_val:
        p = Path(env_val)
        if p.exists():
            return p.read_text().strip()
        return env_val

    return default
