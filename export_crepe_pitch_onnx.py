#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path


VALID_CAPACITIES = ["tiny", "small", "medium", "large", "full"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a CREPE pitch-classifier ONNX for Pakshi segmentation.")
    parser.add_argument("--out", type=Path, required=True, help="Output ONNX path")
    parser.add_argument("--capacity", choices=VALID_CAPACITIES, default="tiny", help="CREPE model capacity to export")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    try:
        import tensorflow as tf  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("tensorflow is required to export a CREPE pitch ONNX model.") from exc

    try:
        import tf2onnx  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("tf2onnx is required to export a CREPE pitch ONNX model.") from exc

    try:
        import crepe  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("The Python package 'crepe' is required for export.") from exc

    model = crepe.core.build_and_load_model(args.capacity)
    layer_map = {layer.name: layer for layer in model.layers}
    classifier = layer_map.get("classifier")
    if classifier is None:
        raise RuntimeError("Could not locate CREPE classifier layer for pitch export.")

    pitch_model = tf.keras.Model(inputs=model.input, outputs=classifier.output, name="crepe_pitch_classifier")

    @tf.function(input_signature=[tf.TensorSpec([None, 1024], tf.float32, name="frames")])
    def serving(frames):
        activation = pitch_model(frames, training=False)
        return {"activation": activation}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tf2onnx.convert.from_function(
        serving,
        input_signature=serving.input_signature,
        opset=args.opset,
        output_path=str(args.out),
    )
    print(f"Wrote CREPE pitch ONNX to {args.out} using capacity '{args.capacity}'")


if __name__ == "__main__":
    main()
