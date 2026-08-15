"""Command-line prediction runner."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .engine import StudentEngine
from .video import read_video


def inference_main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run MINT prediction-only inference on one video.")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", default="artifacts/inference", help="Output directory")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint file or Accelerate checkpoint directory")
    parser.add_argument("--config", default="configs/training/lingbotmap_base.yaml")
    parser.add_argument("--devices", default="auto", help="auto, cpu, a CUDA device, or a comma-separated list")
    parser.add_argument("--max-frames", type=int, default=160)
    parser.add_argument("--target-fps", type=float, default=15.0)
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--camera-mode", choices=("chunked", "max_chunked", "streaming", "full"), default="chunked")
    parser.add_argument("--hand-mode", choices=("hard", "blend", "smooth"), default="smooth")
    parser.add_argument("--render-mode", choices=("mesh", "skeleton", "mesh_skel"), default="mesh_skel")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args(argv)

    source = Path(args.input).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    video = read_video(source, max_frames=args.max_frames, target_fps=args.target_fps)
    print(f"Decoded {len(video.frames_rgb)} frames at {video.fps:.2f} FPS from {source.name}")

    started = time.perf_counter()
    engine = StudentEngine(
        args.config,
        ckpt=args.checkpoint,
        devices=args.devices,
        window=args.window,
    )
    prediction = engine.predict(
        video.frames_rgb,
        cam_mode=args.camera_mode,
        hand_mode=args.hand_mode,
    )
    elapsed = time.perf_counter() - started

    arrays = {key: np.asarray(value) for key, value in prediction.items() if not key.startswith("_")}
    np.savez_compressed(output / "prediction.npz", **arrays)
    if not args.no_render:
        from mint.visualization.render import render_prediction

        render_prediction(
            video.frames_rgb,
            prediction,
            output / "prediction.mp4",
            video.fps,
            mode=args.render_mode,
        )
    summary = {
        "schema_version": 1,
        "input_name": source.name,
        "frames": len(video.frames_rgb),
        "fps": video.fps,
        "duration_seconds": round(elapsed, 4),
        "camera_mode": args.camera_mode,
        "hand_mode": args.hand_mode,
        "rendered": not args.no_render,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Artifacts written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(inference_main())
