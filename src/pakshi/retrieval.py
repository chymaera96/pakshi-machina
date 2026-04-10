from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np

from .config import RuntimeConfig
from .segmentation import Segment

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None

try:
    import onnxruntime as ort  # type: ignore
except Exception:  # pragma: no cover
    ort = None


CREPE_TARGET_SR = 16000
CREPE_TARGET_SAMPLES = 16000


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return arr / norms


def _resample_waveform(waveform: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr == target_sr:
        return waveform.astype(np.float32, copy=False)
    if waveform.size == 0:
        return waveform.astype(np.float32, copy=False)
    target_len = max(1, int(round(waveform.size * float(target_sr) / float(source_sr))))
    source_x = np.linspace(0.0, 1.0, num=waveform.size, endpoint=False, dtype=np.float32)
    target_x = np.linspace(0.0, 1.0, num=target_len, endpoint=False, dtype=np.float32)
    return np.interp(target_x, source_x, waveform).astype(np.float32)


def preprocess_crepe_waveform(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    target_sr: int = CREPE_TARGET_SR,
    target_samples: int = CREPE_TARGET_SAMPLES,
) -> np.ndarray:
    wav = np.asarray(waveform, dtype=np.float32)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    wav = wav.reshape(-1)
    if sample_rate != target_sr:
        wav = _resample_waveform(wav, sample_rate, target_sr)
    if wav.size == 0:
        return np.zeros(target_samples, dtype=np.float32)
    if wav.size < target_samples:
        reps = target_samples // wav.size + 1
        wav = np.tile(wav, reps)[:target_samples]
    else:
        wav = wav[:target_samples]
    return wav.astype(np.float32, copy=False)


class EmbeddingModel(Protocol):
    def embed_batch(self, waveforms: np.ndarray, sample_rate: Optional[int] = None) -> np.ndarray:
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
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.output_name = self._select_output_name(self.output_names)
        self.model_style = "crepe_latent_1s"
        if self.input_name != "audio":
            raise RuntimeError(f"Expected CREPE input tensor named 'audio', got '{self.input_name}'.")

    def _select_output_name(self, output_names: Sequence[str]) -> str:
        lowered = {name.lower(): name for name in output_names}
        for candidate in ["latent", "embedding", "penultimate", "features"]:
            if candidate in lowered:
                return lowered[candidate]
        return output_names[0]

    def embed_batch(self, waveforms: np.ndarray, sample_rate: Optional[int] = None) -> np.ndarray:
        arr = np.asarray(waveforms, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Expected waveform batch of shape [B, T], got {arr.shape}")
        source_sr = int(sample_rate or RuntimeConfig.sample_rate)
        batch = np.stack(
            [preprocess_crepe_waveform(waveform, source_sr) for waveform in arr],
            axis=0,
        ).astype(np.float32)
        outputs = [self._embed_single(waveform) for waveform in batch]
        return np.stack(outputs, axis=0).astype(np.float32)

    def _embed_single(self, waveform: np.ndarray) -> np.ndarray:
        out = self.session.run([self.output_name], {self.input_name: waveform.reshape(1, -1)})[0]
        return self._pool_output(out)

    def _pool_output(self, output: np.ndarray) -> np.ndarray:
        out = np.asarray(output, dtype=np.float32)
        if out.ndim == 1:
            return out
        if out.ndim == 2:
            if out.shape[0] == 1:
                return out[0]
            return out.mean(axis=0)
        if out.ndim == 3:
            if out.shape[0] == 1:
                return out[0].mean(axis=0)
            return out.mean(axis=(0, 1))
        raise ValueError(f"Expected 1D, 2D, or 3D CREPE latent output, got shape {out.shape}")


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
    onset_seconds: float
    scheduled_offset_seconds: float


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
                    "onset_seconds": m.onset_seconds,
                    "scheduled_offset_seconds": m.scheduled_offset_seconds,
                    "metadata": m.metadata,
                }
                for m in self.matches
            ],
        }


class RetrievalEngine:
    def __init__(
        self,
        embedder: EmbeddingModel,
        index: SearchIndex,
        metadata: Sequence[Dict[str, Any]],
        *,
        sample_rate: int = RuntimeConfig.sample_rate,
    ):
        self.embedder = embedder
        self.index = index
        self.metadata = list(metadata)
        self.sample_rate = sample_rate

    def query_segments(self, phrase_id: int, segments: Sequence[Segment]) -> RetrievalSequence:
        if not segments:
            return RetrievalSequence(phrase_id=phrase_id, matches=[])
        batch = np.stack([seg.waveform for seg in segments], axis=0).astype(np.float32)
        emb = normalize_rows(self.embedder.embed_batch(batch, sample_rate=self.sample_rate))
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
                    onset_seconds=seg.onset_seconds,
                    scheduled_offset_seconds=seg.scheduled_offset_seconds,
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
