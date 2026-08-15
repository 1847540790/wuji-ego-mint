# Inference and viewer

## Command-line inference

```bash
python -m mint infer \
  --input data/samples/epic-kitchens-01.mp4 \
  --checkpoint checkpoints/model.safetensors \
  --output artifacts/example
```

The command decodes a bounded number of frames, runs windowed camera-and-hand
prediction, saves `prediction.npz`, and renders `prediction.mp4`. No annotation
file or ground-truth column is opened.

Use `--no-render` when MANO assets are unavailable or only numeric output is
needed. Use `--max-frames` and `--target-fps` to control latency and memory.

The one-video CLI keeps acceleration off by default because compiling a fresh
process can cost much more than one short inference. For a warm, reusable
process or a batch of videos, enable the tested fixed-window path explicitly:

```bash
python -m mint infer \
  --input data/samples/epic-kitchens-01.mp4 \
  --checkpoint checkpoints/model.safetensors \
  --compile-mode auto \
  --fp8-mode auto \
  --warmup-passes 2
```

`auto` selects `reduce-overhead` compile on CUDA and enables dynamic FP8 only
when every selected GPU has CUDA capability 8.9 or newer. Unsupported devices
fall back to BF16 without failing. With compile enabled, the scheduler limits
the total window batch to the number of model replicas, keeping local batch 1
on every GPU. A four-GPU process therefore uses four independent windows per
step, one on each GPU. Hotspots use the captured 32-frame shape; a shorter
final window automatically stays eager instead of spending tens of seconds
compiling a one-off tail shape.

On one NVIDIA H20, the 32-frame Ego4D test improved from 65.7 ms/frame for
BF16 eager batch 4 to 62.0 ms/frame for FP8 plus compile, while peak allocated
memory decreased from 7.82 GiB to 4.60 GiB against the BF16 batch-1 baseline.
The first FP8 compile warmup took about 169 seconds, so this mode is intended
for a resident viewer or multi-video workload rather than a single cold run.

## Web viewer

```bash
python -m mint viewer \
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

The viewer is a resident process, so acceleration defaults to
`--compile-mode auto --fp8-mode auto --warmup-passes 2`. Its first job includes
model compilation; later jobs use the captured graphs. Use
`--compile-mode off --fp8-mode off --warmup-passes 0` to restore eager BF16.

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
