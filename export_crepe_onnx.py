#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path


CREPE_SAMPLE_RATE = 16000
CREPE_WINDOW_SAMPLES = 1024
CREPE_STEP_MS = 10
CREPE_HOP_SAMPLES = int(CREPE_SAMPLE_RATE * CREPE_STEP_MS / 1000)
CREPE_PAD = CREPE_WINDOW_SAMPLES // 2
VALID_CAPACITIES = ["tiny", "small", "medium", "large", "full"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a CREPE-derived latent ONNX for Pakshi.")
    parser.add_argument("--out", type=Path, required=True, help="Output ONNX path")
    parser.add_argument("--capacity", choices=VALID_CAPACITIES, default="tiny", help="CREPE model capacity to export")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    try:
        import tensorflow as tf  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("tensorflow is required to export a CREPE-derived ONNX model.") from exc

    try:
        import tf2onnx  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("tf2onnx is required to export a CREPE-derived ONNX model.") from exc

    try:
        import crepe  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("The Python package 'crepe' is required for export.") from exc

    model = crepe.core.build_and_load_model(args.capacity)

    layer_map = {layer.name: layer for layer in model.layers}
    latent_layer = layer_map.get("flatten")
    if latent_layer is None:
        classifier = layer_map.get("classifier")
        if classifier is None:
            raise RuntimeError("Could not locate CREPE classifier or penultimate latent layer.")
        inbound = getattr(classifier, "input", None)
        if inbound is None:
            raise RuntimeError("Could not access the tensor feeding CREPE's classifier layer.")
        frame_latent_model = tf.keras.Model(inputs=model.input, outputs=inbound, name="crepe_frame_latent")
    else:
        frame_latent_model = tf.keras.Model(inputs=model.input, outputs=latent_layer.output, name="crepe_frame_latent")

    @tf.function(input_signature=[tf.TensorSpec([1, 16000], tf.float32, name="audio")])
    def serving(audio):
        mono = tf.reshape(audio, [1, CREPE_SAMPLE_RATE])
        padded = tf.pad(mono, [[0, 0], [CREPE_PAD, CREPE_PAD]])
        frames = tf.signal.frame(padded, frame_length=CREPE_WINDOW_SAMPLES, frame_step=CREPE_HOP_SAMPLES, axis=1)
        frames = tf.reshape(frames, [-1, CREPE_WINDOW_SAMPLES])
        frame_mean = tf.reduce_mean(frames, axis=1, keepdims=True)
        centered = frames - frame_mean
        frame_std = tf.math.reduce_std(centered, axis=1, keepdims=True)
        normalized = centered / tf.maximum(frame_std, 1e-8)
        frame_latents = frame_latent_model(normalized, training=False)
        if len(frame_latents.shape) == 3:
            frame_latents = tf.reduce_mean(frame_latents, axis=1)
        latent = tf.reduce_mean(frame_latents, axis=0, keepdims=True)
        return {"latent": latent}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tf2onnx.convert.from_function(
        serving,
        input_signature=serving.input_signature,
        opset=args.opset,
        output_path=str(args.out),
    )
    print(f"Wrote CREPE latent ONNX to {args.out} using capacity '{args.capacity}'")


if __name__ == "__main__":
    main()
