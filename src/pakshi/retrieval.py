from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np

from .segmentation import Segment

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None

try:
    import onnxruntime as ort  # type: ignore
except Exception:  # pragma: no cover
    ort = None


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return arr / norms


class EmbeddingModel(Protocol):
    def embed_batch(self, waveforms: np.ndarray) -> np.ndarray:
        ...


class SearchIndex(Protocol):
    def search(self, queries: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        ...


class OnnxEmbeddingModel:
    def __init__(self, model_path: str | Path, providers: Optional[Sequence[str]] = None):
        if ort is None:  # pragma: no cover
            raise RuntimeError("onnxruntime is not installed. Add onnxruntime to run the pakshi worker.")
        self.model_path = str(model_path)
        self.providers = list(providers) if providers else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(self.model_path, providers=self.providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def embed_batch(self, waveforms: np.ndarray) -> np.ndarray:
        arr = np.asarray(waveforms, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Expected waveform batch of shape [B, T], got {arr.shape}")
        out = self.session.run([self.output_name], {self.input_name: arr})[0]
        out = np.asarray(out, dtype=np.float32)
        if out.ndim == 2:
            return out
        if out.ndim == 3:
            # Handle exporter/runtime variants:
            # - [1, B, D]: singleton leading axis, segment axis in the middle
            # - [B, 1, D]: singleton middle axis
            # - [B, T', D]: short time axis that should be pooled per query
            if out.shape[0] == 1:
                return out[0]
            if out.shape[1] == 1:
                return out[:, 0, :]
            return out.mean(axis=1)
        raise ValueError(f"Expected 2D or 3D embedding output, got shape {out.shape}")


class FaissFlatL2Index:
    def __init__(self, index: Any):
        self.index = index

    @classmethod
    def from_embeddings(cls, embeddings: np.ndarray) -> "FaissFlatL2Index":
        if faiss is None:  # pragma: no cover
            raise RuntimeError("faiss is not installed. Add faiss-cpu to use the FAISS backend.")
        arr = np.asarray(embeddings, dtype=np.float32)
        index = faiss.IndexFlatL2(arr.shape[1])
        index.add(arr)
        return cls(index)

    @classmethod
    def load(cls, path: str | Path) -> "FaissFlatL2Index":
        if faiss is None:  # pragma: no cover
            raise RuntimeError("faiss is not installed. Add faiss-cpu to use the FAISS backend.")
        return cls(faiss.read_index(str(path)))

    def save(self, path: str | Path) -> None:
        if faiss is None:  # pragma: no cover
            raise RuntimeError("faiss is not installed. Add faiss-cpu to use the FAISS backend.")
        faiss.write_index(self.index, str(path))

    def search(self, queries: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        arr = np.asarray(queries, dtype=np.float32)
        return self.index.search(arr, k)


class NumpyFlatL2Index:
    def __init__(self, embeddings: np.ndarray):
        arr = np.asarray(embeddings, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got {arr.shape}")
        self.embeddings = arr

    @classmethod
    def from_embeddings(cls, embeddings: np.ndarray) -> "NumpyFlatL2Index":
        return cls(embeddings)

    def search(self, queries: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        arr = np.asarray(queries, dtype=np.float32)
        dists = np.sum((arr[:, None, :] - self.embeddings[None, :, :]) ** 2, axis=2)
        topk = np.argsort(dists, axis=1)[:, :k]
        topd = np.take_along_axis(dists, topk, axis=1)
        return topd.astype(np.float32), topk.astype(np.int64)


@dataclass
class SegmentMatch:
    phrase_id: int
    segment_index: int
    distance: float
    corpus_index: int
    metadata: Dict[str, Any]
    start_seconds: float
    end_seconds: float


@dataclass
class RetrievalSequence:
    phrase_id: int
    matches: List[SegmentMatch]

    def to_event(self) -> Dict[str, Any]:
        return {
            "type": "retrieval_sequence_ready",
            "phrase_id": self.phrase_id,
            "num_segments": len(self.matches),
            "matches": [
                {
                    "segment_index": m.segment_index,
                    "distance": m.distance,
                    "corpus_index": m.corpus_index,
                    "start_seconds": m.start_seconds,
                    "end_seconds": m.end_seconds,
                    "metadata": m.metadata,
                }
                for m in self.matches
            ],
        }


class RetrievalEngine:
    def __init__(self, embedder: EmbeddingModel, index: SearchIndex, metadata: Sequence[Dict[str, Any]]):
        self.embedder = embedder
        self.index = index
        self.metadata = list(metadata)

    def query_segments(self, phrase_id: int, segments: Sequence[Segment]) -> RetrievalSequence:
        if not segments:
            return RetrievalSequence(phrase_id=phrase_id, matches=[])
        batch = np.stack([seg.waveform for seg in segments], axis=0).astype(np.float32)
        emb = normalize_rows(self.embedder.embed_batch(batch))
        distances, indices = self.index.search(emb, k=1)

        matches: List[SegmentMatch] = []
        for row, seg in enumerate(segments):
            corpus_index = int(indices[row, 0])
            metadata = dict(self.metadata[corpus_index]) if 0 <= corpus_index < len(self.metadata) else {}
            matches.append(
                SegmentMatch(
                    phrase_id=phrase_id,
                    segment_index=seg.index,
                    distance=float(distances[row, 0]),
                    corpus_index=corpus_index,
                    metadata=metadata,
                    start_seconds=seg.start_seconds,
                    end_seconds=seg.end_seconds,
                )
            )
        return RetrievalSequence(phrase_id=phrase_id, matches=matches)


def load_metadata(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    if p.suffix == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with p.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with p.open("r", encoding="utf-8") as handle:
        return list(json.load(handle))
