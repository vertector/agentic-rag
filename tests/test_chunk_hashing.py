
import pytest
import uuid
from shared.schemas import Chunk, Grounding

def test_chunk_structural_hash_stability():
    """Verify that get_structural_hash is deterministic and ignores the summary field."""
    grounding = Grounding(chunk_type="table", bbox=[0, 0, 100, 100], page_index=1)
    
    chunk1 = Chunk(
        chunk_markdown="| col1 | col2 |\n|---|---|\n| val1 | val2 |",
        context="Header: Results",
        grounding=grounding,
        summary=None
    )
    
    chunk2 = Chunk(
        chunk_markdown="| col1 | col2 |\n|---|---|\n| val1 | val2 |",
        context="Header: Results",
        grounding=grounding,
        summary="A summary that should be ignored"
    )
    
    hash1 = chunk1.get_structural_hash()
    hash2 = chunk2.get_structural_hash()
    
    assert hash1 == hash2, "Structural hash should be identical regardless of summary content"
    assert len(hash1) == 64, "Should be a SHA-256 hex string"

def test_chunk_content_hash_sensitivity():
    """Verify that get_content_hash DOES include the summary."""
    grounding = Grounding(chunk_type="table", bbox=[0, 0, 100, 100], page_index=1)
    
    chunk1 = Chunk(
        chunk_markdown="data",
        grounding=grounding,
        summary="Summary A"
    )
    
    chunk2 = Chunk(
        chunk_markdown="data",
        grounding=grounding,
        summary="Summary B"
    )
    
    assert chunk1.get_content_hash() != chunk2.get_content_hash(), \
        "Content hash (Merkle leaf) must change if summary changes"
