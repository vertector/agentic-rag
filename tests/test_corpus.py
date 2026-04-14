import os
import sys
from pathlib import Path
import pytest

# Ensure src is in python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from shared.corpus_manager import CorpusManager
from shared.schemas import Corpus, PipelineSettings
from shared.utils import get_project_root

@pytest.fixture
def test_corpus_manager(tmp_path):
    """Provides a CorpusManager focused on a temporary directory."""
    return CorpusManager(storage_root=tmp_path / "corpora")

def test_corpus_creation(test_corpus_manager):
    """Test that a Corpus can be created and persisted."""
    manager = test_corpus_manager
    corpus = manager.create_corpus(
        corpus_id="legal_kb_v1",
        description="Legal Documents"
    )
    
    assert corpus.corpus_id == "legal_kb_v1"
    assert corpus.description == "Legal Documents"
    assert corpus.corpus_merkle_root != ""
    
    # Reload from disk
    loaded = manager.get_corpus("legal_kb_v1")
    assert loaded is not None
    assert loaded.corpus_id == corpus.corpus_id
    assert loaded.corpus_merkle_root == corpus.corpus_merkle_root

def test_corpus_add_snapshot_updates_merkle(test_corpus_manager):
    """Test that adding snapshots correctly recalculates the corpus Merkle root."""
    manager = test_corpus_manager
    manager.create_corpus("medical_kb")
    
    corpus_initial = manager.get_corpus("medical_kb")
    initial_root = corpus_initial.corpus_merkle_root
    
    # Add a document
    manager.add_snapshot_to_corpus(
        corpus_id="medical_kb",
        filename="patient_records.pdf",
        doc_cid="blob_cid_123",
        settings_hash="hash_a",
        merkle_root="doc_merkle_abc"
    )
    
    corpus_after_first = manager.get_corpus("medical_kb")
    assert corpus_after_first.corpus_merkle_root != initial_root
    assert len(corpus_after_first.documents) == 1
    
    # Add a second document
    manager.add_snapshot_to_corpus(
        corpus_id="medical_kb",
        filename="lab_results.pdf",
        doc_cid="blob_cid_456",
        settings_hash="hash_a",
        merkle_root="doc_merkle_xyz"
    )
    
    corpus_after_second = manager.get_corpus("medical_kb")
    assert corpus_after_second.corpus_merkle_root != corpus_after_first.corpus_merkle_root
    assert len(corpus_after_second.documents) == 2

def test_corpus_deterministic_merkle(test_corpus_manager):
    """Test that the sequence of adding documents does not affect the final DAG root."""
    manager1 = CorpusManager(storage_root=test_corpus_manager.storage_root / "test1")
    manager1.create_corpus("corpus_a")
    manager1.add_snapshot_to_corpus("corpus_a", "doc1.pdf", "cid1", "sh1", "mr1")
    manager1.add_snapshot_to_corpus("corpus_a", "doc2.pdf", "cid2", "sh1", "mr2")
    
    manager2 = CorpusManager(storage_root=test_corpus_manager.storage_root / "test2")
    manager2.create_corpus("corpus_b")
    manager2.add_snapshot_to_corpus("corpus_b", "doc2.pdf", "cid2", "sh1", "mr2")
    manager2.add_snapshot_to_corpus("corpus_b", "doc1.pdf", "cid1", "sh1", "mr1")
    
    # Because of alphabetized hashing in CorpusManager, the root should match regardless of insertion order
    c1 = manager1.get_corpus("corpus_a")
    c2 = manager2.get_corpus("corpus_b")
    
    assert c1.corpus_merkle_root == c2.corpus_merkle_root

if __name__ == "__main__":
    pytest.main(["-v", __file__])
