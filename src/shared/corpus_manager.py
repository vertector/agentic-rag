import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timezone

from shared.schemas import Corpus, CorpusSnapshot, PipelineSettings
from shared.utils import get_project_root, atomic_json_dump

class CorpusManager:
    """
    Manages Corpus Knowledge Bases dynamically. Corpora act as the "Git Repositories"
    for our document snapshots.
    """
    def __init__(self, storage_root: Optional[Path] = None):
        if storage_root is None:
            storage_root = get_project_root() / "src" / ".cache" / "corpora"
        self.storage_root = storage_root
        self.storage_root.mkdir(parents=True, exist_ok=True)
    
    def _corpus_path(self, corpus_id: str) -> Path:
        return self.storage_root / f"{corpus_id}.json"

    def get_corpus(self, corpus_id: str) -> Optional[Corpus]:
        """Loads a Corpus from disk."""
        path = self._corpus_path(corpus_id)
        if not path.exists():
            return None
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return Corpus.model_validate(data)
        except Exception as e:
            print(f"Error loading Corpus {corpus_id}: {e}")
            return None

    def list_corpora(self) -> List[str]:
        """Lists all available Corpora IDs."""
        return [f.stem for f in self.storage_root.glob("*.json")]

    def save_corpus(self, corpus: Corpus) -> None:
        """Atomically saves a Corpus to disk, updating the Merkle root and timestamp."""
        corpus.corpus_merkle_root = corpus.compute_corpus_merkle_root()
        corpus.updated_at = datetime.now(timezone.utc)
        
        path = self._corpus_path(corpus.corpus_id)
        atomic_json_dump(path, corpus.model_dump(mode="json"))

    def create_corpus(self, corpus_id: str, description: str = "", settings: Optional[PipelineSettings] = None) -> Corpus:
        """Creates a new empty Corpus if one doesn't exist."""
        existing = self.get_corpus(corpus_id)
        if existing:
            return existing
            
        corpus = Corpus(
            corpus_id=corpus_id,
            description=description,
            settings=settings or PipelineSettings()
        )
        self.save_corpus(corpus)
        return corpus

    def add_snapshot_to_corpus(self, corpus_id: str, filename: str, doc_cid: str, settings_hash: str, merkle_root: str) -> Corpus:
        """
        Registers a document snapshot to a Corpus.
        Acts like a "commit" applying changes to the KB.
        """
        corpus = self.get_corpus(corpus_id)
        if not corpus:
            raise ValueError(f"Corpus {corpus_id} does not exist.")
            
        snapshot = CorpusSnapshot(
            doc_cid=doc_cid,
            settings_hash=settings_hash,
            merkle_root=merkle_root
        )
        
        corpus.documents[filename] = snapshot
        self.save_corpus(corpus)
        return corpus

    def remove_document(self, corpus_id: str, filename: str) -> Optional[Corpus]:
        """Removes a document pointer from a Corpus."""
        corpus = self.get_corpus(corpus_id)
        if not corpus or filename not in corpus.documents:
            return corpus
            
        del corpus.documents[filename]
        self.save_corpus(corpus)
        return corpus
