#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Iterable, List
from urllib.request import urlretrieve


ML4BL_RECORD_URL = "https://zenodo.org/records/5545872"
ML4BL_ZIP_URL = "https://zenodo.org/records/5545872/files/ML4BL_ZF.zip?download=1"
ML4BL_ZIP_NAME = "ML4BL_ZF.zip"
ML4BL_DIR_NAME = "ML4BL_ZF"


def _print_progress(block_count: int, block_size: int, total_size: int) -> None:
    if total_size <= 0:
        return
    downloaded = min(block_count * block_size, total_size)
    percent = 100.0 * downloaded / total_size
    sys.stdout.write(f"\rDownloading ML4BL: {percent:5.1f}%")
    sys.stdout.flush()
    if downloaded >= total_size:
        sys.stdout.write("\n")


def download_zip(zip_path: Path, *, force: bool = False) -> Path:
    if zip_path.exists() and not force:
        print(f"Using existing archive: {zip_path}")
        return zip_path

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {ML4BL_ZIP_URL}")
    urlretrieve(ML4BL_ZIP_URL, zip_path, _print_progress)
    print(f"Saved archive to {zip_path}")
    return zip_path


def extract_zip(zip_path: Path, data_dir: Path, *, force: bool = False) -> Path:
    target_dir = data_dir / ML4BL_DIR_NAME
    if target_dir.exists() and not force:
        print(f"Using existing extracted dataset: {target_dir}")
        return target_dir

    if target_dir.exists() and force:
        shutil.rmtree(target_dir)

    data_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(data_dir)
    print(f"Extracted dataset to {target_dir}")
    return target_dir


def write_wav_manifest(wav_dir: Path, manifest_path: Path) -> Path:
    wav_paths = sorted(wav_dir.glob("*.wav"))
    rows = [
        {
            "id": wav_path.stem,
            "name": wav_path.stem,
            "path": str(wav_path.resolve()),
        }
        for wav_path in wav_paths
    ]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")
    print(f"Wrote manifest with {len(rows)} items to {manifest_path}")
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and unpack the ML4BL zebra finch dataset into ./data/ML4BL_ZF."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory where ML4BL_ZF.zip is stored and extracted",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download the ZIP archive even if it already exists",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Delete and re-extract the ML4BL_ZF directory if it already exists",
    )
    # parser.add_argument(
    #     "--write-manifest",
    #     action="store_true",
    #     help="Also write a JSONL manifest for data/ML4BL_ZF/wavs",
    # )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default="data/ml4bl_wavs.jsonl",
        help="Optional output path for the generated manifest JSONL",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    zip_path = data_dir / ML4BL_ZIP_NAME

    dataset_dir = data_dir / ML4BL_DIR_NAME
    if not data_dir.exists():
        dataset_dir = extract_zip(
            download_zip(zip_path, force=args.force_download),
            data_dir,
            force=args.force_extract,
        )

    wav_dir = dataset_dir / "wavs"
    if not wav_dir.exists():
        raise FileNotFoundError(
            f"Expected wav directory at {wav_dir}, but it was not found after extraction."
        )

    print(f"ML4BL record: {ML4BL_RECORD_URL}")
    print(f"WAV directory ready: {wav_dir}")

    manifest_path = args.manifest_path
    if manifest_path is None:
        manifest_path = data_dir / "ml4bl_wavs.jsonl"
    write_wav_manifest(wav_dir, manifest_path.resolve())


if __name__ == "__main__":
    main()
