
import os
from pathlib import Path


def get_secret(name: str, default: str | None = None) -> str | None:
    """Return secret value preferring file-based injection."""
    # Prefer NAME_FILE pointing to a file injected by Vault/K8s.
    file_path = os.environ.get(f"{name}_FILE")
    if file_path:
        p = Path(file_path)
        if p.exists():
            return p.read_text().strip()
    # Fallback: if NAME itself is set and points to a file, read it; otherwise return the raw value.
    env_val = os.environ.get(name)
    if env_val:
        p = Path(env_val)
        if p.exists():
            return p.read_text().strip()
        return env_val
    return default
