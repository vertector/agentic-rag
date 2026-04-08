import os
from pathlib import Path

def load_env():
    """
    Robust, dependency-free .env loader for the project.
    Locates the .env file in the project root and populates os.environ
    if keys are not already set.
    """
    try:
        # env_loader.py is in src/shared/
        # Root is three levels up: src/shared -> src -> root
        _actual_root = Path(__file__).resolve().parent.parent.parent
        _env_path = _actual_root / ".env"
        
        if _env_path.exists():
            with open(_env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, val = line.split("=", 1)
                        # Strip whitespace and potential surrounding quotes
                        key = key.strip()
                        val = val.strip().strip("'\"")
                        if key not in os.environ:
                            os.environ[key] = val
    except Exception:
        # Fail silently as this is a fallback for missing environment variables
        pass
