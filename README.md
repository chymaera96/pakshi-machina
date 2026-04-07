# pakshi-machina

Realtime segmented phrase retrieval for a voice-and-sound performance system.

## Overview (v0.2)
- Capture a sung phrase using a noise-gate envelope VAD tuned through an operator-assisted setup stage.
- Split the completed phrase into fixed `0.5 s` non-overlapping segments.
- Embed each segment with an ONNX model.
- L2-normalize each embedding and retrieve the top-1 match from a static FAISS corpus.
- Play the resulting sequence of retrieved sounds in segment order.

## Repo Layout
- `src/pakshi/`: Python runtime, phrase segmentation, retrieval, corpus loading, playback helpers
- `pakshi_worker.py`: JSON-lines worker entrypoint
- `build_pakshi_corpus.py`: build a static corpus bundle from audio metadata
- `setup_ml4bl.py`: download and unpack the ML4BL zebra finch dataset into `data/`
- `tests/`: runtime tests
- `pakshi_ui/`: Electron control surface scaffold
- `*.onnx`: the app auto-discovers the single ONNX model in the repo root
- `pakshi_bundle/`: default retrieval bundle used by the app

## Setup
Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate pakshi-machina
```

If you update `environment.yml` later, refresh the env with:

```bash
conda env update -f environment.yml --prune
```

Notes:
- Run the Python commands below inside the active `pakshi-machina` environment.
- The app auto-discovers the single `*.onnx` file in the repo root and uses `pakshi_bundle/` by default.
- You only need `PAKSHI_MODEL_PATH` or `PAKSHI_BUNDLE_PATH` if you want to override those defaults.
- If `faiss-cpu` is unavailable on a machine, the runtime can fall back to `embeddings.npy` with the in-repo NumPy backend.

## ML4BL Dataset Setup
The project uses the ML4BL zebra finch dataset from Zenodo:
- https://zenodo.org/records/5545872

To download and unpack the dataset into `data/ML4BL_ZF/wavs`:


```bash
python setup_ml4bl.py --write-manifest
```

## Build A Corpus Bundle
```bash
python build_pakshi_corpus.py \
  --model model_tc11.onnx \
  --manifest data/ml4bl_wavs.jsonl \
  --out_dir pakshi_bundle
```

The bundle contains `metadata.jsonl` plus `index.faiss` and/or `embeddings.npy`.

## Run The Worker (for debugging)

```bash
python pakshi_worker.py \
  --model model_tc11.onnx \
  --bundle pakshi_bundle
```

The worker is the long-running backend. During setup it captures room noise and realistic singing level, then derives gate thresholds automatically. When armed, it opens the microphone, monitors live amplitude in dBFS, segments phrases with a noise gate, runs retrieval, and schedules playback.

You do not need to start `pakshi_worker.py` manually when using the Electron app. The app launches it automatically on startup.

## Run The UI
Install the Electron app:

```bash
cd pakshi_ui
npm install
npm start
```

UI notes:
- Use Setup Mode to capture room noise and then capture realistic singing. Setup completes automatically after the singing capture.
- After setup, use the record control to open the mic for live listening.
- During playback, the mic is paused so the system does not trigger on its own bird vocalisations.
- Tap the lit record control at any time to stop playback and reset the live state.


## Validation
```bash
python -m unittest discover -s tests -p 'test_*.py'
```
