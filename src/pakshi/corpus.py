from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .retrieval import FaissFlatL2Index, NumpyFlatL2Index, SearchIndex, load_metadata


@dataclass
class BundleMetadata:
    model_family: str
    model_style: str
    embedding_sample_rate: int
    model_path: str


@dataclass
class CorpusBundle:
    bundle_dir: Path
    metadata: List[Dict[str, Any]]
    index: SearchIndex
    backend: str
    bundle_metadata: Optional[BundleMetadata] = None


def _load_bundle_metadata(bundle_path: Path) -> Optional[BundleMetadata]:
    metadata_path = bundle_path / "bundle_metadata.json"
    if not metadata_path.exists():
        return None
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    return BundleMetadata(
        model_family=str(payload["model_family"]),
        model_style=str(payload["model_style"]),
        embedding_sample_rate=int(payload["embedding_sample_rate"]),
        model_path=str(payload["model_path"]),
    )


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
    bundle_metadata = _load_bundle_metadata(bundle_path)

    faiss_path = bundle_path / "index.faiss"
    embeddings_path = bundle_path / "embeddings.npy"

    if faiss_path.exists():
        try:
            index = FaissFlatL2Index.load(faiss_path)
            return CorpusBundle(
                bundle_dir=bundle_path,
                metadata=metadata,
                index=index,
                backend="faiss",
                bundle_metadata=bundle_metadata,
            )
        except RuntimeError:
            if not embeddings_path.exists():
                raise

    if embeddings_path.exists():
        embeddings = np.load(embeddings_path)
        index = NumpyFlatL2Index.from_embeddings(embeddings.astype(np.float32))
        return CorpusBundle(
            bundle_dir=bundle_path,
            metadata=metadata,
            index=index,
            backend="numpy",
            bundle_metadata=bundle_metadata,
        )

    raise FileNotFoundError(f"No usable index.faiss or embeddings.npy found in {bundle_path}")


def write_metadata_jsonl(rows: List[Dict[str, Any]], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def write_bundle_metadata(bundle_metadata: BundleMetadata, path: str | Path) -> None:
    Path(path).write_text(json.dumps(asdict(bundle_metadata), indent=2), encoding="utf-8")
