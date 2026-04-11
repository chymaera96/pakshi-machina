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
class VisualizationData:
    points: List[Dict[str, Any]]
    mean: np.ndarray
    components: np.ndarray
    scale: float


@dataclass
class CorpusBundle:
    bundle_dir: Path
    metadata: List[Dict[str, Any]]
    index: SearchIndex
    backend: str
    bundle_metadata: Optional[BundleMetadata] = None
    visualization: Optional[VisualizationData] = None


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


def _load_visualization_data(bundle_path: Path) -> Optional[VisualizationData]:
    visualization_path = bundle_path / "visualization.json"
    if not visualization_path.exists():
        return None
    payload = json.loads(visualization_path.read_text(encoding="utf-8"))
    projection = payload.get("projection", {})
    return VisualizationData(
        points=list(payload.get("points", [])),
        mean=np.asarray(projection.get("mean", []), dtype=np.float32),
        components=np.asarray(projection.get("components", []), dtype=np.float32),
        scale=float(projection.get("scale", 1.0)),
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
    visualization = _load_visualization_data(bundle_path)

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
                visualization=visualization,
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
            visualization=visualization,
        )

    raise FileNotFoundError(f"No usable index.faiss or embeddings.npy found in {bundle_path}")


def write_metadata_jsonl(rows: List[Dict[str, Any]], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def write_bundle_metadata(bundle_metadata: BundleMetadata, path: str | Path) -> None:
    Path(path).write_text(json.dumps(asdict(bundle_metadata), indent=2), encoding="utf-8")


def write_visualization_data(visualization: VisualizationData, path: str | Path) -> None:
    payload = {
        "points": visualization.points,
        "projection": {
            "mean": visualization.mean.astype(np.float32).tolist(),
            "components": visualization.components.astype(np.float32).tolist(),
            "scale": float(visualization.scale),
        },
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def compute_visualization_projection(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    arr = np.asarray(embeddings, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D embeddings array, got {arr.shape}")
    mean = arr.mean(axis=0).astype(np.float32)
    centered = arr - mean
    if arr.shape[0] < 2:
        components = np.zeros((3, arr.shape[1]), dtype=np.float32)
        components[0, 0] = 1.0
    else:
        _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
        dims = min(3, vh.shape[0])
        components = np.zeros((3, arr.shape[1]), dtype=np.float32)
        components[:dims] = vh[:dims].astype(np.float32)
    for row in range(components.shape[0]):
        if not np.any(components[row]):
            continue
        pivot = int(np.argmax(np.abs(components[row])))
        if components[row, pivot] < 0:
            components[row] *= -1.0
    coords = centered @ components.T
    max_abs = float(np.max(np.abs(coords))) if coords.size else 0.0
    scale = float(max(max_abs, 1e-6))
    coords = (coords / scale).astype(np.float32)
    return mean, components, scale, coords


def project_embeddings_3d(embeddings: np.ndarray, mean: np.ndarray, components: np.ndarray, scale: float) -> np.ndarray:
    arr = np.asarray(embeddings, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    centered = arr - np.asarray(mean, dtype=np.float32)
    coords = centered @ np.asarray(components, dtype=np.float32).T
    divisor = float(scale) if float(scale) > 0 else 1.0
    return (coords / divisor).astype(np.float32)
