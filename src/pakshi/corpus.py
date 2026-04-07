from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .retrieval import FaissFlatL2Index, NumpyFlatL2Index, SearchIndex, load_metadata


@dataclass
class CorpusBundle:
    bundle_dir: Path
    metadata: List[Dict[str, Any]]
    index: SearchIndex
    backend: str


def load_corpus_bundle(bundle_dir: str | Path) -> CorpusBundle:
    bundle_path = Path(bundle_dir)
    metadata_path: Optional[Path] = None
    for candidate in [bundle_path / "metadata.json", bundle_path / "metadata.jsonl"]:
        if candidate.exists():
            metadata_path = candidate
            break
    if metadata_path is None:
        raise FileNotFoundError(f"No metadata.json or metadata.jsonl found in {bundle_path}")

    metadata = load_metadata(metadata_path)

    faiss_path = bundle_path / "index.faiss"
    embeddings_path = bundle_path / "embeddings.npy"

    if faiss_path.exists():
        try:
            index = FaissFlatL2Index.load(faiss_path)
            return CorpusBundle(bundle_dir=bundle_path, metadata=metadata, index=index, backend="faiss")
        except RuntimeError:
            if not embeddings_path.exists():
                raise

    if embeddings_path.exists():
        embeddings = np.load(embeddings_path)
        index = NumpyFlatL2Index.from_embeddings(embeddings.astype(np.float32))
        return CorpusBundle(bundle_dir=bundle_path, metadata=metadata, index=index, backend="numpy")

    raise FileNotFoundError(f"No usable index.faiss or embeddings.npy found in {bundle_path}")


def write_metadata_jsonl(rows: List[Dict[str, Any]], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")
