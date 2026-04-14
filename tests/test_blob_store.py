import os
import shutil
import tempfile
import sys
from pathlib import Path

# Add project root and src to path for imports
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from shared.blob_store import BlobStore

def test_blob_store():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "blobs"
        store = BlobStore(storage_root=root)
        
        # Test put_bytes
        data = b"hello world"
        cid = store.put_bytes(data)
        print(f"CID for 'hello world': {cid}")
        
        assert store.exists(cid)
        assert store.get_bytes(cid) == data
        
        path = store.get_path(cid)
        assert path.exists()
        assert path.parent.name == cid[:2]
        
        # Test deduplication
        cid2 = store.put_bytes(data)
        assert cid == cid2
        
        # Test put_file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"file content")
            tmp_path = Path(tmp.name)
        
        try:
            cid_file = store.put_file(tmp_path)
            print(f"CID for 'file content': {cid_file}")
            assert store.exists(cid_file)
            assert store.get_bytes(cid_file) == b"file content"
        finally:
            if tmp_path.exists():
                os.unlink(tmp_path)
                
        print("BlobStore core tests passed!")

if __name__ == "__main__":
    test_blob_store()
