# pakshi-machina

Realtime segmented phrase retrieval for a voice-and-sound performance system.

## Overview
- Capture a sung phrase using a CREPE-confidence VAD proxy.
- Split the completed phrase into fixed `2.0 s` non-overlapping segments.
- Embed each segment with an ONNX model exported from `SFXSearch/export_onnx.py`.
- L2-normalize each embedding and retrieve the top-1 match from a static FAISS corpus.
- Play the resulting sequence of retrieved sounds in segment order.

## Repo Layout
- `src/pakshi/`: Python runtime, phrase segmentation, retrieval, corpus loading, playback helpers
- `pakshi_worker.py`: JSON-lines worker entrypoint
- `build_pakshi_corpus.py`: build a static corpus bundle from audio metadata
- `setup_ml4bl.py`: download and unpack the ML4BL zebra finch dataset into `data/`
- `tests/`: runtime tests
- `pakshi_ui/`: Electron control surface scaffold
- `models/`: optional local place to drop `model.onnx` for the UI bootstrap path

## Python Setup
Use Python `3.12` for this project. Avoid older system interpreters such as macOS Python `3.9`, because newer `onnxruntime` and related wheels may not resolve correctly there.

Recommended interpreter on this machine:

```bash
/Users/abhattacharjee/miniforge3/bin/python3.12 --version
```

Create the virtualenv with that interpreter:

```bash
/Users/abhattacharjee/miniforge3/bin/python3.12 -m venv .venv
./.venv/bin/python --version
./.venv/bin/python -m pip --version
```

When you are ready to install dependencies:

```bash
./.venv/bin/python -m pip install -r requirements.txt
```

## Installation Notes
- Installation was intentionally not completed yet; package download/setup has been deferred.
- If `python3 -m venv .venv` gives you Python `3.9`, delete the env and recreate it with the explicit Miniforge `python3.12` path above.
- `onnxruntime` is pinned to a version that is more likely to resolve on the available package index for Python `3.12`.
- The Electron UI does not bundle a model file. Set `PAKSHI_MODEL_PATH=/absolute/path/to/model.onnx` before launching the UI, or place a model at `models/model.onnx`.
- Runtime audio packages such as `sounddevice` may also require local audio system permissions from macOS on first use.
- If `faiss-cpu` fails to install on a target machine, the Python runtime can still load a corpus from `embeddings.npy` and fall back to the in-repo NumPy search backend, though FAISS remains the preferred path.

Suggested step-by-step install flow:

```bash
rm -rf .venv
/Users/abhattacharjee/miniforge3/bin/python3.12 -m venv .venv
./.venv/bin/python --version
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
```

## ML4BL Dataset Setup
The project uses the ML4BL zebra finch dataset hosted on Zenodo. Zenodo’s record lists a downloadable archive named `ML4BL_ZF.zip`, containing an `ML4BL_ZF/wavs` directory with the `.wav` files.

Source:
- https://zenodo.org/records/5545872

To download and unpack the dataset into `data/ML4BL_ZF/wavs`:

```bash
./.venv/bin/python setup_ml4bl.py
```

This stores:
- `data/ML4BL_ZF.zip`
- `data/ML4BL_ZF/`
- `data/ML4BL_ZF/wavs/`

To also generate a JSONL manifest from the wav directory:

```bash
./.venv/bin/python setup_ml4bl.py --write-manifest
```

That writes `data/ml4bl_wavs.jsonl`, which can be passed directly to the corpus builder.

## Build A Corpus Bundle
The bundle should contain:
- `metadata.json` or `metadata.jsonl`
- `index.faiss` and/or `embeddings.npy`

Expected manifest fields:
- required: `path`
- recommended: `name`
- optional: `duration_seconds`, `category`, `tags`, any other display metadata

Create one from a manifest:

```bash
./.venv/bin/python build_pakshi_corpus.py \
  --model /path/to/model.onnx \
  --manifest data/ml4bl_wavs.jsonl \
  --out_dir pakshi_bundle
```

## Run The Worker

```bash
./.venv/bin/python pakshi_worker.py --model /path/to/model.onnx
```

Optional startup flags:

```bash
./.venv/bin/python pakshi_worker.py \
  --model /path/to/model.onnx \
  --bundle /path/to/pakshi_bundle \
  --arm
```

The worker speaks JSON lines over stdin/stdout and supports:
- `load_corpus`
- `set_params`
- `arm`
- `disarm`
- `stop_all`
- `get_state`
- `process_frame`
- `process_phrase`

## Electron App Installation
Install Node.js first. On macOS, `brew install node` is a simple path if Node is not already present.

From the repo root:

```bash
cd pakshi_ui
npm install
```

That installs Electron locally into `pakshi_ui/node_modules`.

## Run The UI
Recommended environment variables before launch:

```bash
export PAKSHI_PYTHON_PATH=/Users/abhattacharjee/pakshi-machina/.venv/bin/python
export PAKSHI_MODEL_PATH=/absolute/path/to/model.onnx
export PAKSHI_BUNDLE_PATH=/absolute/path/to/pakshi_bundle
export PAKSHI_ARM_ON_BOOT=1
```

Then start Electron:

```bash
cd pakshi_ui
npm start
```

Notes:
- If `PAKSHI_MODEL_PATH` is not set, the UI looks for `models/model.onnx`.
- If `PAKSHI_BUNDLE_PATH` is set, the worker preloads that corpus at launch.
- If `PAKSHI_PYTHON_PATH` is not set, the UI uses `../.venv/bin/python`.
- The UI is currently a control surface and event monitor; it does not yet perform live microphone capture itself.

## Validation
Once dependencies are installed, a minimal verification pass is:

```bash
./.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

## Plan Status
Implemented:
- VAD-defined phrase segmentation logic with fixed `2.0 s` non-overlapping segment slicing in [src/pakshi/segmentation.py](/Users/abhattacharjee/pakshi-machina/src/pakshi/segmentation.py)
- Per-segment ONNX embedding, L2 normalization, and top-1 retrieval in [src/pakshi/retrieval.py](/Users/abhattacharjee/pakshi-machina/src/pakshi/retrieval.py)
- Static corpus loading with FAISS or NumPy fallback in [src/pakshi/corpus.py](/Users/abhattacharjee/pakshi-machina/src/pakshi/corpus.py)
- Ordered playback-sequence scheduling and worker events in [src/pakshi/worker.py](/Users/abhattacharjee/pakshi-machina/src/pakshi/worker.py)
- Electron control surface scaffold in [pakshi_ui/main.js](/Users/abhattacharjee/pakshi-machina/pakshi_ui/main.js) and [pakshi_ui/renderer.js](/Users/abhattacharjee/pakshi-machina/pakshi_ui/renderer.js)
- ML4BL dataset bootstrap into `data/ML4BL_ZF/wavs` via [setup_ml4bl.py](/Users/abhattacharjee/pakshi-machina/setup_ml4bl.py)

Still pending for full parity with the original plan:
- Live microphone capture in the worker is not wired yet; audio is currently expected via worker commands rather than an internal input stream
- CREPE-based confidence estimation exists in [src/pakshi/audio.py](/Users/abhattacharjee/pakshi-machina/src/pakshi/audio.py) but is not yet connected to a realtime capture loop
- The UI is a control/event monitor, not yet the final performance interface with live latent-space graphics
