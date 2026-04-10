from __future__ import annotations

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


EFFNET_BIO_SAMPLE_RATE = 16_000
EFFNET_BIO_CLIP_SAMPLES = 4_000
EFFNET_BIO_N_FFT = 800
EFFNET_BIO_HOP_LENGTH = 160
EFFNET_BIO_WIN_LENGTH = 800
EFFNET_BIO_N_MELS = 128


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return arr / norms


def load_metadata(path: str | Path) -> List[Dict[str, Any]]:
    metadata_path = Path(path)
    if metadata_path.suffix == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with metadata_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(__import__("json").loads(line))
        return rows
    with metadata_path.open("r", encoding="utf-8") as handle:
        return list(__import__("json").load(handle))


def _resample_waveform(waveform: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr == target_sr:
        return waveform.astype(np.float32, copy=False)
    if waveform.size == 0:
        return waveform.astype(np.float32, copy=False)
    target_len = max(1, int(round(waveform.size * float(target_sr) / float(source_sr))))
    source_x = np.linspace(0.0, 1.0, num=waveform.size, endpoint=False, dtype=np.float32)
    target_x = np.linspace(0.0, 1.0, num=target_len, endpoint=False, dtype=np.float32)
    return np.interp(target_x, source_x, waveform).astype(np.float32)


def preprocess_effnet_bio_waveform(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    target_sr: int = EFFNET_BIO_SAMPLE_RATE,
    target_samples: int = EFFNET_BIO_CLIP_SAMPLES,
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
        padded = np.zeros(target_samples, dtype=np.float32)
        padded[: wav.size] = wav
        return padded
    return wav[:target_samples].astype(np.float32, copy=False)


def _hann_window(length: int) -> np.ndarray:
    return (0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(length) / length))).astype(np.float32)


def _mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    fmin, fmax = 0.0, sr / 2.0
    mel_min = 2595.0 * np.log10(1.0 + fmin / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + fmax / 700.0)
    mels = np.linspace(mel_min, mel_max, n_mels + 2)
    freqs = 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
    bins = np.floor((n_fft + 1) * freqs / sr).astype(int)

    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        lo, mid, hi = bins[i], bins[i + 1], bins[i + 2]
        for j in range(lo, mid):
            fb[i, j] = (j - lo) / max(mid - lo, 1)
        for j in range(mid, hi):
            fb[i, j] = (hi - j) / max(hi - mid, 1)
    return fb


_EFFNET_WINDOW = _hann_window(EFFNET_BIO_WIN_LENGTH)
_EFFNET_MEL_FB = _mel_filterbank(EFFNET_BIO_SAMPLE_RATE, EFFNET_BIO_N_FFT, EFFNET_BIO_N_MELS)


def compute_effnet_bio_mel_spectrogram(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    batch_size = arr.shape[0]
    results: List[np.ndarray] = []
    for i in range(batch_size):
        wav = arr[i]
        pad = EFFNET_BIO_N_FFT // 2
        wav = np.pad(wav, (pad, pad), mode="reflect")
        n_frames = 1 + (len(wav) - EFFNET_BIO_N_FFT) // EFFNET_BIO_HOP_LENGTH
        frames = np.stack(
            [
                wav[j * EFFNET_BIO_HOP_LENGTH : j * EFFNET_BIO_HOP_LENGTH + EFFNET_BIO_N_FFT]
                * _EFFNET_WINDOW
                for j in range(n_frames)
            ],
            axis=0,
        )
        stft = np.fft.rfft(frames, n=EFFNET_BIO_N_FFT, axis=1)
        power = np.abs(stft).astype(np.float32) ** 2
        mel = _EFFNET_MEL_FB @ power.T
        mel = np.log(mel + 1e-6)
        mel_min = mel.min()
        mel_max = mel.max()
        mel = (mel - mel_min) / (mel_max - mel_min + 1e-8)
        mel_3ch = np.stack([mel, mel, mel], axis=0)
        results.append(mel_3ch.astype(np.float32))
    return np.stack(results, axis=0).astype(np.float32)


class EmbeddingModel(Protocol):
    def embed_batch(self, waveforms: np.ndarray, sample_rate: Optional[int] = None) -> np.ndarray:
        ...


class SearchIndex(Protocol):
    def search(self, queries: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        ...


class EffNetBioEmbeddingModel:
    def __init__(
        self,
        model_path: str | Path,
        input_sample_rate: int = RuntimeConfig.sample_rate,
        providers: Optional[Sequence[str]] = None,
        *,
        batch_size: int = RuntimeConfig.embedding_batch_size,
    ):
        if ort is None:  # pragma: no cover
            raise RuntimeError("onnxruntime is not installed. Add onnxruntime to run the pakshi worker.")
        self.model_path = str(model_path)
        self.input_sample_rate = int(input_sample_rate)
        self.providers = list(providers) if providers else ["CPUExecutionProvider"]
        self.batch_size = max(1, int(batch_size))
        self.session = ort.InferenceSession(self.model_path, providers=self.providers)
        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name
        self.output_name = self.session.get_outputs()[0].name
        self.model_style = "effnet_bio_emb1024"
        input_shape = list(input_meta.shape)
        self._fixed_batch_size = None
        if input_shape and isinstance(input_shape[0], int):
            self._fixed_batch_size = int(input_shape[0])
        if self.input_name != "mel_spec":
            raise RuntimeError(f"Expected model input tensor named 'mel_spec', got '{self.input_name}'.")

    def embed_batch(self, waveforms: np.ndarray, sample_rate: Optional[int] = None) -> np.ndarray:
        arr = np.asarray(waveforms, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        if arr.ndim != 2:
            raise ValueError(f"Expected waveform batch of shape [B, T], got {arr.shape}")
        source_sr = int(sample_rate or self.input_sample_rate)
        prepared = np.stack(
            [preprocess_effnet_bio_waveform(waveform, source_sr) for waveform in arr],
            axis=0,
        )
        mel = compute_effnet_bio_mel_spectrogram(prepared)
        fixed_batch_size = getattr(self, "_fixed_batch_size", None)
        if fixed_batch_size in (None, mel.shape[0]):
            embeddings = self.session.run([self.output_name], {self.input_name: mel})[0]
            embeddings = np.asarray(embeddings, dtype=np.float32)
        elif fixed_batch_size == 1:
            outputs = []
            for item in mel:
                out = self.session.run([self.output_name], {self.input_name: item[np.newaxis, ...]})[0]
                outputs.append(np.asarray(out[0], dtype=np.float32))
            embeddings = np.stack(outputs, axis=0)
        else:
            raise RuntimeError(
                f"EffNetBio ONNX expects fixed batch size {fixed_batch_size}, "
                f"but received batch size {mel.shape[0]}."
            )
        if embeddings.ndim != 2:
            raise ValueError(f"Expected EffNetBio output shape [B, D], got {embeddings.shape}")
        return embeddings


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
    segment_start_seconds: float
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
                    "segment_start_seconds": m.segment_start_seconds,
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
                    segment_start_seconds=seg.segment_start_seconds,
                    scheduled_offset_seconds=seg.scheduled_offset_seconds,
                )
            )
        return RetrievalSequence(phrase_id=phrase_id, matches=matches)
