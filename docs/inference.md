# Inference and viewer

## Web viewer

```bash
python -m mint viewer
```

The viewer is the original `eval/model_effect` interactive interface. It starts
at `data/samples/lerobot_v3` and supports the bundled dataset's
ground-truth/prediction 2D and 3D comparison workflow. Model loading and
inference are separate explicit actions. The default model configuration is
`configs/training/stage2_resume_worldengine_camera_only.yaml`.

Pass a path only when overriding a default, for example:

```bash
python -m mint viewer --ckpt /path/to/model.safetensors
```

If the default checkpoint is missing, run `bash scripts/download_assets.sh`.
The viewer provides:

- a LeRobot episode browser and raw GT mode;
- GT/prediction overlay and side-by-side playback;
- fixed-world and current-camera interactive 3D views;
- frame values, configured losses, cancellation, and video/frame export;
- optional MuJoCo and Wuji Hand panels when their dependencies/assets exist.

The viewer is a resident process. Optional `--compile-mode` and `--fp8-mode`
settings apply to repeated workloads; leave them unset for the default eager
execution path.

Benchmark code is intentionally separate from the Viewer UI. Run
`eval/model_effect/benchmark/run.py` directly after configuring the required
datasets and optional environment.

## Headless command-line inference

`mint infer` is an alternative for automation and artifact export; it does not
need the viewer, and the viewer does not require it to run first.

```bash
python -m mint infer \
  --input /path/to/video.mp4 \
  --checkpoint checkpoints/model.safetensors \
  --output artifacts/example
```

The command decodes a bounded number of frames, runs windowed camera-and-hand
prediction, saves `prediction.npz`, and renders `prediction.mp4`. No annotation
file or ground-truth column is opened. Use `--no-render` when MANO assets are
unavailable or only numeric output is needed.

The one-video CLI keeps acceleration off by default because compiling a fresh
process can cost more than one short inference. Use `--compile-mode auto`,
`--fp8-mode auto`, and `--warmup-passes 2` only for repeated or sufficiently
large headless workloads.

## Network safety

The default port is 8011. If remote access is needed, prefer an SSH
port forward:

```bash
ssh -L 8011:127.0.0.1:8011 user@remote-host
```

Exposing `--host 0.0.0.0` requires authentication and TLS at a reverse proxy.
The built-in development server is not a public production server.
