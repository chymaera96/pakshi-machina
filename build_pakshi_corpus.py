#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import soundfile as sf

from src.pakshi.config import RuntimeConfig
from src.pakshi.corpus import write_metadata_jsonl
from src.pakshi.retrieval import FaissFlatL2Index, NumpyFlatL2Index, OnnxEmbeddingModel, normalize_rows


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Pakshi corpus bundle from a manifest and CREPE-latent ONNX model.")
    parser.add_argument("--model", type=Path, required=True, help="Path to CREPE-latent waveform->embedding ONNX model")
    parser.add_argument("--manifest", type=Path, required=True, help="JSON or JSONL manifest with sound file metadata")
    parser.add_argument("--out_dir", type=Path, required=True, help="Output corpus bundle directory")
    parser.add_argument("--sample_rate", type=int, default=RuntimeConfig.sample_rate)
    args = parser.parse_args()

    manifest = _load_manifest(args.manifest)
    embedder = OnnxEmbeddingModel(args.model)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    total = len(manifest)
    started_at = time.time()
    rows: List[Dict[str, Any]] = []
    embeddings: List[np.ndarray] = []
    for index, row in enumerate(manifest, start=1):
        path = Path(row["path"])
        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        emb = embedder.embed_batch(np.expand_dims(audio.astype(np.float32), axis=0), sample_rate=sr)
        emb = normalize_rows(emb)[0]
        embeddings.append(emb)
        out_row = dict(row)
        out_row["path"] = str(path)
        out_row.setdefault("duration_seconds", float(audio.size / sr))
        rows.append(out_row)
        _print_progress(index, total, path, started_at)

    emb_arr = np.stack(embeddings, axis=0).astype(np.float32)
    np.save(args.out_dir / "embeddings.npy", emb_arr)
    write_metadata_jsonl(rows, args.out_dir / "metadata.jsonl")

    try:
        FaissFlatL2Index.from_embeddings(emb_arr).save(args.out_dir / "index.faiss")
    except Exception:
        NumpyFlatL2Index.from_embeddings(emb_arr)

    print(f"Wrote corpus bundle to {args.out_dir}")


if __name__ == "__main__":
    main()
