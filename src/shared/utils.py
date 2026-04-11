import os
import re
from pathlib import Path
from typing import Set

def get_project_root() -> Path:
    """Returns the absolute path to the project root."""
    # src/shared/utils.py -> parent: shared, parent.parent: src, parent.parent.parent: root
    return Path(__file__).resolve().parent.parent.parent

def validate_path(path: str | Path, allowed_roots: Set[Path] | None = None) -> Path:
    """
    Validates that a path is absolute and within the allowed roots.
    Prevents directory traversal attacks.
    
    For relative paths, it attempts to find the file in:
    1. Current Working Directory
    2. Project Root
    3. Project Root / data /
    """
    project_root = get_project_root()
    if allowed_roots is None:
        allowed_roots = {project_root}
    
    path_obj = Path(path)
    
    # Attempt resolution strategies for relative paths
    if not path_obj.is_absolute():
        candidates = [
            Path.cwd() / path_obj,
            project_root / path_obj,
            project_root / "data" / path_obj,
        ]
        
        found = False
        for cand in candidates:
            if cand.exists():
                path_obj = cand
                found = True
                break
        
        if not found:
            # Safe default
            path_obj = (project_root / path_obj).resolve()
    
    abs_path = path_obj.resolve()
    
    is_allowed = False
    for root in allowed_roots:
        try:
            abs_path.relative_to(root.resolve())
            is_allowed = True
            break
        except ValueError:
            continue
            
    if not is_allowed:
        raise ValueError(f"Path '{path}' resolves to '{abs_path}', which is not within any allowed directories.")
        
    return abs_path

def sanitize_stem(stem: str) -> str:
    """Sanitizes a string to be used as a safe filename stem."""
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', stem)

def resolve_placeholders(text: str) -> str:
    """Resolves known placeholders like {SAMPLE_PDF} to absolute paths."""
    project_root = get_project_root()
    data_dir = project_root / "data"
    
    # Map of stems/placeholders to absolute paths
    MAPPING = {
        "SAMPLE_PDF": str((data_dir / "sample.pdf").resolve()),
    }
    
    resolved = text
    for key, val in MAPPING.items():
        # Match {KEY} or just KEY (case insensitive, bounded by non-alphanumerics)
        pattern = re.compile(rf'{{?{re.escape(key)}}}?', re.IGNORECASE)
        resolved = pattern.sub(val, resolved)
    return resolved
