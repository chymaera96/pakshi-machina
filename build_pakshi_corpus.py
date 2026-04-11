#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import soundfile as sf

from src.pakshi.config import RuntimeConfig
from src.pakshi.corpus import (
    BundleMetadata,
    VisualizationData,
    compute_visualization_projection,
    write_bundle_metadata,
    write_metadata_jsonl,
    write_visualization_data,
)
from src.pakshi.retrieval import (
    MODEL_FAMILIES,
    create_embedding_model,
    default_bundle_dir_name_for_family,
    describe_backend,
    infer_model_family,
    FaissFlatL2Index,
    NumpyFlatL2Index,
    normalize_rows,
)


def _load_manifest(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with path.open("r", encoding="utf-8") as handle:
        return list(json.load(handle))


def _print_progress(index: int, total: int, path: Path, started_at: float) -> None:
    elapsed = max(1e-6, time.time() - started_at)
    rate = index / elapsed
    remaining = max(total - index, 0)
    eta = remaining / rate if rate > 0 else float("inf")
    width = 28
    filled = int(width * index / max(total, 1))
    bar = "#" * filled + "-" * (width - filled)
    eta_text = f"{eta:6.1f}s" if np.isfinite(eta) else "   infs"
    message = f"\r[{bar}] {index:4d}/{total:4d} | {rate:4.2f} files/s | eta {eta_text} | {path.name[:40]}"
    sys.stderr.write(message)
    sys.stderr.flush()
    if index == total:
        sys.stderr.write("\n")


def _load_audio_batch(rows: Sequence[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Path, np.ndarray, int]]:
    batch: List[Tuple[Dict[str, Any], Path, np.ndarray, int]] = []
    for row in rows:
        path = Path(row["path"])
        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        batch.append((row, path, np.asarray(audio, dtype=np.float32), int(sr)))
    return batch


def _infer_syllable_type(row: Dict[str, Any]) -> str:
    for key in ("syl_type", "syllable_type", "type", "category"):
        value = row.get(key)
        if value:
            return str(value)
    label = str(row.get("name") or Path(str(row.get("path", ""))).stem)
    match = re.match(r"([A-Za-z]+)", label)
    return match.group(1) if match else "Unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Pakshi corpus bundle from a manifest and ONNX embedding model.")
    parser.add_argument("--model", type=Path, required=True, help="Path to retrieval ONNX model")
    parser.add_argument("--model-family", choices=MODEL_FAMILIES, default=None, help="Optional retrieval model family override")
    parser.add_argument("--manifest", type=Path, required=True, help="JSON or JSONL manifest with sound file metadata")
    parser.add_argument("--out_dir", type=Path, default=None, help="Output corpus bundle directory")
    parser.add_argument("--sample_rate", type=int, default=RuntimeConfig.sample_rate)
    parser.add_argument("--batch-size", type=int, default=RuntimeConfig.embedding_batch_size)
    args = parser.parse_args()

    model_family = args.model_family or infer_model_family(args.model)
    out_dir = args.out_dir or Path(default_bundle_dir_name_for_family(model_family))
    manifest = _load_manifest(args.manifest)
    batch_size = max(1, int(args.batch_size))
    embedder = create_embedding_model(
        args.model,
        model_family=model_family,
        input_sample_rate=args.sample_rate,
        batch_size=batch_size,
    )
    backend = describe_backend(args.model, model_family=model_family, input_sample_rate=embedder.input_sample_rate)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(manifest)
    started_at = time.time()
    rows: List[Dict[str, Any]] = []
    embeddings: List[np.ndarray] = []
    for batch_start in range(0, total, batch_size):
        loaded = _load_audio_batch(manifest[batch_start : batch_start + batch_size])
        by_sr: Dict[int, List[Tuple[Dict[str, Any], Path, np.ndarray, int]]] = {}
        for item in loaded:
            by_sr.setdefault(item[3], []).append(item)

        batch_embeddings: Dict[str, np.ndarray] = {}
        for sr, sr_items in by_sr.items():
            max_len = max(audio.size for _row, _path, audio, _sr in sr_items)
            padded = np.zeros((len(sr_items), max_len), dtype=np.float32)
            for row_index, (_row, _path, audio, _sr) in enumerate(sr_items):
                padded[row_index, : audio.size] = audio
            sr_embs = normalize_rows(embedder.embed_batch(padded, sample_rate=sr))
            for item, emb in zip(sr_items, sr_embs):
                batch_embeddings[str(item[1])] = emb

        for offset, (row, path, audio, sr) in enumerate(loaded, start=1):
            embeddings.append(batch_embeddings[str(path)])
            out_row = dict(row)
            out_row["path"] = str(path)
            out_row.setdefault("duration_seconds", float(audio.size / sr))
            rows.append(out_row)
            _print_progress(batch_start + offset, total, path, started_at)

    emb_arr = np.stack(embeddings, axis=0).astype(np.float32)
    np.save(out_dir / "embeddings.npy", emb_arr)
    write_metadata_jsonl(rows, out_dir / "metadata.jsonl")
    mean, components, scale, coords = compute_visualization_projection(emb_arr)
    points: List[Dict[str, Any]] = []
    for idx, (row, coord) in enumerate(zip(rows, coords)):
        label = str(row.get("name") or row.get("label") or Path(str(row.get("path", idx))).stem)
        points.append(
            {
                "index": idx,
                "label": label,
                "syl_type": _infer_syllable_type(row),
                "x": float(coord[0]),
                "y": float(coord[1]),
                "z": float(coord[2]),
            }
        )
    write_visualization_data(
        VisualizationData(
            points=points,
            mean=mean,
            components=components,
            scale=scale,
        ),
        out_dir / "visualization.json",
    )
    write_bundle_metadata(
        BundleMetadata(
            model_family=backend.model_family,
            model_style=backend.model_style,
            embedding_sample_rate=backend.input_sample_rate,
            model_path=str(Path(args.model).resolve()),
        ),
        out_dir / "bundle_metadata.json",
    )

    try:
        FaissFlatL2Index.from_embeddings(emb_arr).save(out_dir / "index.faiss")
    except Exception:
        NumpyFlatL2Index.from_embeddings(emb_arr)

    print(f"Wrote {backend.model_family} corpus bundle to {out_dir}")


if __name__ == "__main__":
    main()
