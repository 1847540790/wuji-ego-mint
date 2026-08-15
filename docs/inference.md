# Inference and viewer

## Command-line inference

```bash
mint infer \
  --input data/samples/epic-kitchens-01.mp4 \
  --checkpoint checkpoints/model.safetensors \
  --output artifacts/example
```

The command decodes a bounded number of frames, runs windowed camera-and-hand
prediction, saves `prediction.npz`, and renders `prediction.mp4`. No annotation
file or ground-truth column is opened.

Use `--no-render` when MANO assets are unavailable or only numeric output is
needed. Use `--max-frames` and `--target-fps` to control latency and memory.

## Web viewer

```bash
mint viewer \
  --samples data/samples \
  --checkpoint checkpoints/model.safetensors \
  --open
```

The viewer starts immediately and loads the checkpoint lazily on the first
job. It provides:

- a curated sample reel;
- source and prediction-overlay playback;
- an interactive dependency-free trajectory canvas;
- bounded progress and cancellation;
- NPZ artifact download.

There is no ground-truth comparison mode. The source tab is the original video,
not an annotation or evaluation target.

## Network safety

The default address is localhost. If remote access is needed, prefer an SSH
port forward:

```bash
ssh -L 7860:127.0.0.1:7860 user@remote-host
```

Exposing `--host 0.0.0.0` requires authentication and TLS at a reverse proxy.
The built-in development server is not a public production server.

