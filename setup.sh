#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(pwd)"
EFFNET_MODEL_PATH="${REPO_ROOT}/effnet_bio_zf_emb1024.onnx"
CREPE_MODEL_PATH="${REPO_ROOT}/crepe_latent.onnx"
MANIFEST_PATH="${REPO_ROOT}/data/ml4bl_wavs.jsonl"
EFFNET_BUNDLE_PATH="${REPO_ROOT}/pakshi_bundle_effnet_bio"
CREPE_BUNDLE_PATH="${REPO_ROOT}/pakshi_bundle_crepe_latent"

python "${REPO_ROOT}/build_pakshi_corpus.py" \
  --model "${EFFNET_MODEL_PATH}" \
  --model-family effnet_bio \
  --manifest "${MANIFEST_PATH}" \
  --batch-size 32 \
  --out_dir "${EFFNET_BUNDLE_PATH}"

python "${REPO_ROOT}/build_pakshi_corpus.py" \
  --model "${CREPE_MODEL_PATH}" \
  --model-family crepe_latent \
  --manifest "${MANIFEST_PATH}" \
  --batch-size 32 \
  --out_dir "${CREPE_BUNDLE_PATH}"

echo
echo "Bundles built successfully!"
echo "Next step:"
echo "  cd pakshi_ui && npm start"
