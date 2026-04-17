
import asyncio
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from shared.schemas import Document, Metadata, Chunk, Grounding
from ingestion_pipeline.ingestion_pipeline import AsyncMerkleQdrantIngestor

@pytest.mark.anyio
async def test_summarize_chunk_caching():
    """Verify that _summarize_chunk hits Redis and avoids LLM if cached."""
    # Mocking Qdrant and Redis
    mock_qdrant = AsyncMock()
    mock_redis = AsyncMock()
    
    # Initialize ingestor with mocks
    ingestor = AsyncMerkleQdrantIngestor(qdrant_url="http://mock:6333")
    ingestor.qdrant = mock_qdrant
    ingestor.redis = mock_redis
    
    grounding = Grounding(chunk_type="table", bbox=[0, 0, 10, 10], page_index=1)
    chunk = Chunk(chunk_markdown="<table>data</table>", grounding=grounding)
    
    # --- Scenario 1: Cache Miss ---
    mock_redis.get.return_value = None
    
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Summary from LLM"
    
    with patch("litellm.acompletion", AsyncMock(return_value=mock_response)) as mock_llm:
        summary = await ingestor._summarize_chunk(chunk)
        
        assert summary == "Summary from LLM"
        assert mock_llm.called, "LLM should be called on cache miss"
        # Verify set was called to cache the result
        assert mock_redis.set.called
        cache_key = mock_redis.set.call_args[0][0]
        assert "cache:summary:" in cache_key

    # --- Scenario 2: Cache Hit ---
    mock_redis.get.return_value = "Summary from Cache"
    mock_redis.set.reset_mock()
    
    with patch("litellm.acompletion", AsyncMock()) as mock_llm:
        summary = await ingestor._summarize_chunk(chunk)
        
        assert summary == "Summary from Cache"
        assert not mock_llm.called, "LLM should NOT be called on cache hit"
        assert not mock_redis.set.called, "Redis set should NOT be called on cache hit"

@pytest.mark.anyio
async def test_summarize_chunk_skips_non_data():
    """Verify that non-data chunks (standard text) are NOT summarized."""
    ingestor = AsyncMerkleQdrantIngestor(qdrant_url="http://mock:6333")
    ingestor.redis = AsyncMock()
    
    grounding = Grounding(chunk_type="text", bbox=[0, 0, 10, 10], page_index=1)
    chunk = Chunk(chunk_markdown="Just some normal text here.", grounding=grounding)
    
    with patch("litellm.acompletion", AsyncMock()) as mock_llm:
        summary = await ingestor._summarize_chunk(chunk)
        assert summary is None
        assert not mock_llm.called
        assert not ingestor.redis.get.called
