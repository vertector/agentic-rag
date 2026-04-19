import os
import re
import json
import tempfile
from pathlib import Path
from typing import List, Optional, Set, Union

class HeaderStack:
    """
    Manages a stack of document headers to provide hierarchical context (breadcrumbs).
    
    Logic:
    - Level 0 (e.g. document_title) clears the entire stack.
    - Level N (e.g. paragraph_title or ##) replaces headers at the same or lower level (>= N).
    """
    def __init__(self, initial_state: Optional[List[str]] = None):
        # List of (level, title) tuples
        self._stack: List[tuple[int, str]] = []
        if initial_state:
            # Reconstruct from serialized state: assumes sequential levels 0, 1, 2...
            for i, title in enumerate(initial_state):
                self._stack.append((i, title))

    @staticmethod
    def get_level(label: str, content: str = "") -> int:
        """
        Determines the hierarchical level of a header block.
        0 = Document Title
        1 = Section / H1
        2 = Subsection / H2
        ...
        """
        label = label.lower().replace(" ", "_")
        
        # 1. Hard-coded OCR Labels
        if label == "document_title":
            return 0
        if label in ("paragraph_title", "title", "section_title"):
            # Check markdown depth if content starts with #
            stripped = content.strip()
            if stripped.startswith("#"):
                match = re.match(r'^(#+)', stripped)
                if match:
                    return len(match.group(1))
            return 1 # Default level for paragraph_title
        
        if label == "figure_title":
            return 2 # Usually nested under a section
            
        return 1 # Fallback for unknown titles

    def push(self, level: int, title: str) -> None:
        """Adds a new header at the specified level, popping lower levels."""
        # Clean title: strip markdown headers if present
        clean_title = re.sub(r'^#+\s*', '', title.strip())
        if not clean_title: return

        if level == 0:
            self._stack = [(0, clean_title)]
        # Keep only levels strictly higher (smaller number) than the current level
        self._stack = [h for h in self._stack if h[0] < level]
        self._stack.append((level, clean_title))

    def format_breadcrumb(self, separator: str = " > ") -> str:
        """Returns the formatted breadcrumb string (e.g. 'Intro > Data > Tables')."""
        return separator.join(h[1] for h in self._stack)

    def get_state(self) -> List[str]:
        """Returns a serializable list of titles for cross-page persistence."""
        return [h[1] for h in self._stack]

    def __bool__(self) -> bool:
        return len(self._stack) > 0

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
        # Strictly match {KEY} (case insensitive)
        pattern = re.compile(rf'{{(?P<key>{re.escape(key)})}}', re.IGNORECASE)
        resolved = pattern.sub(val, resolved)
    return resolved

def atomic_write(file_path: Path, content: Union[str, bytes]) -> None:
    """
    Writes content to a temporary file and then atomically replaces the target.
    Prevents corrupted files on power loss or process crash.
    """
    dir_path = file_path.parent
    dir_path.mkdir(parents=True, exist_ok=True)
    
    # Use the same directory as the target to ensure they are on the same filesystem
    # (required for atomic os.replace)
    suffix = ".tmp"
    is_bytes = isinstance(content, bytes)
    mode = "wb" if is_bytes else "w"
    encoding = None if is_bytes else "utf-8"
    
    with tempfile.NamedTemporaryFile(mode=mode, dir=str(dir_path), encoding=encoding, suffix=suffix, delete=False) as tf:
        tmp_path = Path(tf.name)
        try:
            tf.write(content)
            tf.flush()
            os.fsync(tf.fileno()) # Force write to physical storage
            tf.close()
            os.replace(tmp_path, file_path)
        except Exception:
            if tmp_path.exists():
                os.unlink(tmp_path)
            raise

def atomic_json_dump(file_path: Path, data: Union[dict, list], indent: int = 2) -> None:
    """Atomically writes clear JSON to a file."""
    content = json.dumps(data, indent=indent, ensure_ascii=False)
    atomic_write(file_path, content)
