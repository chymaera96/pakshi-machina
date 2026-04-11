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


MODEL_FAMILY_EFFNET_BIO = "effnet_bio"
MODEL_FAMILY_CREPE_LATENT = "crepe_latent"
MODEL_FAMILIES = (MODEL_FAMILY_EFFNET_BIO, MODEL_FAMILY_CREPE_LATENT)
DEFAULT_BUNDLE_DIR_BY_FAMILY = {
    MODEL_FAMILY_EFFNET_BIO: "pakshi_bundle_effnet_bio",
    MODEL_FAMILY_CREPE_LATENT: "pakshi_bundle_crepe_latent",
}
DEFAULT_SEGMENT_SECONDS_BY_FAMILY = {
    MODEL_FAMILY_EFFNET_BIO: 0.25,
    MODEL_FAMILY_CREPE_LATENT: 0.5,
}

EFFNET_BIO_SAMPLE_RATE = 16_000
EFFNET_BIO_CLIP_SAMPLES = 4_000
EFFNET_BIO_N_FFT = 800
EFFNET_BIO_HOP_LENGTH = 160
EFFNET_BIO_WIN_LENGTH = 800
EFFNET_BIO_N_MELS = 128

CREPE_TARGET_SR = 16_000
CREPE_TARGET_SAMPLES = 16_000


@dataclass(frozen=True)
class BackendDescriptor:
    model_family: str
    model_style: str
    model_path: str
    input_sample_rate: int
    default_bundle_dir_name: str
    preprocessing: str


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


def infer_model_family(model_path: str | Path) -> str:
    name = Path(model_path).name.lower()
    if "effnet" in name:
        return MODEL_FAMILY_EFFNET_BIO
    if "crepe" in name:
        return MODEL_FAMILY_CREPE_LATENT
    raise ValueError(
        f"Could not infer model family from '{model_path}'. "
        f"Pass an explicit family from {MODEL_FAMILIES}."
    )


def default_bundle_dir_name_for_family(model_family: str) -> str:
    if model_family not in DEFAULT_BUNDLE_DIR_BY_FAMILY:
        raise ValueError(f"Unsupported model family '{model_family}'. Expected one of {MODEL_FAMILIES}.")
    return DEFAULT_BUNDLE_DIR_BY_FAMILY[model_family]


def default_segment_seconds_for_family(model_family: str) -> float:
    if model_family not in DEFAULT_SEGMENT_SECONDS_BY_FAMILY:
        raise ValueError(f"Unsupported model family '{model_family}'. Expected one of {MODEL_FAMILIES}.")
    return float(DEFAULT_SEGMENT_SECONDS_BY_FAMILY[model_family])


def describe_backend(
    model_path: str | Path,
    *,
    model_family: Optional[str] = None,
    input_sample_rate: Optional[int] = None,
) -> BackendDescriptor:
    family = model_family or infer_model_family(model_path)
    if family == MODEL_FAMILY_EFFNET_BIO:
        return BackendDescriptor(
            model_family=family,
            model_style="effnet_bio_emb1024",
            model_path=str(model_path),
            input_sample_rate=int(input_sample_rate or EFFNET_BIO_SAMPLE_RATE),
            default_bundle_dir_name=default_bundle_dir_name_for_family(family),
            preprocessing="mono -> 16kHz -> 250ms -> mel_spec[3,128,26]",
        )
    if family == MODEL_FAMILY_CREPE_LATENT:
        return BackendDescriptor(
            model_family=family,
            model_style="crepe_latent_1s",
            model_path=str(model_path),
            input_sample_rate=int(input_sample_rate or CREPE_TARGET_SR),
            default_bundle_dir_name=default_bundle_dir_name_for_family(family),
            preprocessing="mono -> 16kHz -> repeat-pad/crop 1s waveform",
        )
    raise ValueError(f"Unsupported model family '{family}'. Expected one of {MODEL_FAMILIES}.")


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
                wav[j * EFFNET_BIO_HOP_LENGTH : j * EFFNET_BIO_HOP_LENGTH + EFFNET_BIO_N_FFT] * _EFFNET_WINDOW
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
    model_family: str
    model_style: str
    model_path: str
    input_sample_rate: int

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
        self.model_family = MODEL_FAMILY_EFFNET_BIO
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
        prepared = np.stack([preprocess_effnet_bio_waveform(waveform, source_sr) for waveform in arr], axis=0)
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


class CrepeLatentEmbeddingModel:
    def __init__(self, model_path: str | Path, providers: Optional[Sequence[str]] = None):
        if ort is None:  # pragma: no cover
            raise RuntimeError("onnxruntime is not installed. Add onnxruntime to run the pakshi worker.")
        self.model_path = str(model_path)
        self.model_family = MODEL_FAMILY_CREPE_LATENT
        self.input_sample_rate = CREPE_TARGET_SR
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
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        if arr.ndim != 2:
            raise ValueError(f"Expected waveform batch of shape [B, T], got {arr.shape}")
        source_sr = int(sample_rate or self.input_sample_rate)
        batch = np.stack([preprocess_crepe_waveform(waveform, source_sr) for waveform in arr], axis=0).astype(np.float32)
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


def create_embedding_model(
    model_path: str | Path,
    *,
    model_family: Optional[str] = None,
    input_sample_rate: int = RuntimeConfig.sample_rate,
    providers: Optional[Sequence[str]] = None,
    batch_size: int = RuntimeConfig.embedding_batch_size,
) -> EmbeddingModel:
    family = model_family or infer_model_family(model_path)
    if family == MODEL_FAMILY_EFFNET_BIO:
        return EffNetBioEmbeddingModel(
            model_path,
            input_sample_rate=input_sample_rate,
            providers=providers,
            batch_size=batch_size,
        )
    if family == MODEL_FAMILY_CREPE_LATENT:
        return CrepeLatentEmbeddingModel(model_path, providers=providers)
    raise ValueError(f"Unsupported model family '{family}'. Expected one of {MODEL_FAMILIES}.")


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
    query_embeddings: Optional[np.ndarray] = None

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
            return RetrievalSequence(phrase_id=phrase_id, matches=[], query_embeddings=np.zeros((0, 0), dtype=np.float32))
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
        return RetrievalSequence(phrase_id=phrase_id, matches=matches, query_embeddings=emb)
