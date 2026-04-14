import hashlib
import os
import shutil
import logging
from pathlib import Path
from typing import Optional

try:
    from .utils import get_project_root
except ImportError:
    # Fallback if utils is not in the same package during direct script execution
    def get_project_root() -> Path:
        return Path(__file__).resolve().parent.parent.parent

logger = logging.getLogger(__name__)

class BlobStore:
    """
    A simple thread-safe Content-Addressed Storage (CAS) system.
    Stores files in a sharded directory structure based on SHA-256 hashes.
    """

    def __init__(self, storage_root: Optional[Path] = None):
        if storage_root is None:
            storage_root = get_project_root() / "data" / "blobs"
        self.root = storage_root
        self.root.mkdir(parents=True, exist_ok=True)
        logger.info(f"BlobStore initialised at {self.root}")

    def _compute_hash_bytes(self, data: bytes) -> str:
        """Compute SHA-256 for bytes."""
        return hashlib.sha256(data).hexdigest()

    def _compute_hash_file(self, file_path: Path) -> str:
        """Compute SHA-256 for a file without loading it entirely into memory."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                sha256.update(chunk)
        return sha256.hexdigest()

    def get_path(self, cid: str) -> Path:
        """Get the storage path for a given CID (sharded by first 2 chars)."""
        if not cid or len(cid) < 2:
            raise ValueError(f"Invalid CID: {cid}")
        prefix = cid[:2]
        return self.root / prefix / cid

    def exists(self, cid: str) -> bool:
        """Check if a blob with the given CID exists."""
        return self.get_path(cid).exists()

    def put_bytes(self, data: bytes) -> str:
        """Store bytes in the BlobStore and return the CID."""
        cid = self._compute_hash_bytes(data)
        path = self.get_path(cid)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            # Write to a temp file first and move it to ensure atomicity
            import tempfile
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
                tmp.write(data)
                tmp_path = Path(tmp.name)
            tmp_path.replace(path)
        return cid

    def put_file(self, file_path: Path) -> str:
        """Store a file in the BlobStore and return the CID. Does not delete source."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        cid = self._compute_hash_file(file_path)
        path = self.get_path(cid)
        
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            # Copy to temp and move for atomicity
            import tempfile
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
                tmp_close = tmp.name
            shutil.copy(file_path, tmp_close)
            Path(tmp_close).replace(path)
            
        return cid

    def get_bytes(self, cid: str) -> bytes:
        """Retrieve the bytes for a given CID."""
        path = self.get_path(cid)
        if not path.exists():
            raise FileNotFoundError(f"Blob with CID {cid} not found.")
        with open(path, "rb") as f:
            return f.read()

# Singleton instance for general use
_instance: Optional[BlobStore] = None

def get_blob_store() -> BlobStore:
    global _instance
    if _instance is None:
        _instance = BlobStore()
    return _instance
