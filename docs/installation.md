# Installation

## Supported platform

- Linux x86_64
- Python 3.10
- NVIDIA GPU with a driver compatible with the selected PyTorch CUDA build
- Conda or Mamba
- FFmpeg with H.264 encoding support

The tested model environment uses PyTorch 2.8.0 and torchvision 0.23.0 from the
CUDA 12.8 wheel index. CPU import and configuration tests are supported, but
practical training and model inference require a CUDA GPU.

## Clean full environment

```bash
bash scripts/create_env.sh full
conda activate mint
mint doctor --profile full --strict
```

This performs a clean solve from `environments/mint.yml`, then installs each
pinned requirements layer. It never clones another local environment. If the
environment already exists, the script exits rather than mutating it silently.

## Clean inference environment

```bash
bash scripts/create_env.sh inference
conda activate mint-inference
mint doctor --profile inference --strict
```

The inference profile includes PyTorch, OpenCV, Flask, numerical dependencies,
and FFmpeg. It excludes Ray, PyArrow, W&B, data conversion, model compilers,
and upstream processing backends.

## Model checkpoint

Place a released MINT checkpoint at `checkpoints/model.safetensors`, or pass an
explicit path to `--checkpoint`. Do not commit checkpoints to Git. The model
configuration must match the checkpoint architecture.

The optional pretrained LingBot-Map backbone belongs at:

```text
assets/models/lingbot-map.pt
```

Obtain it from the official LingBot-Map release and review its model terms.
The helper below downloads that public backbone and the redistributable robot
description bundle:

```bash
bash scripts/download_assets.sh
```

## MANO assets

Mesh and skeleton rendering requires separately licensed MANO files:

```text
assets/mano/mano_right/MANO_RIGHT.pkl
assets/mano/mano_left/MANO_LEFT.pkl
```

Create an account on the [official MANO website](https://mano.is.tue.mpg.de/),
download `mano_v*_*.zip`, and accept the
[MANO license](https://mano.is.tue.mpg.de/license.html). MANO cannot be
downloaded or redistributed by this project.

HaWoR and the data pipeline use the upstream `_DATA` layout:

```text
third_party/HaWoR/_DATA/data/mano/MANO_RIGHT.pkl
third_party/HaWoR/_DATA/data_left/mano_left/MANO_LEFT.pkl
```

The standalone viewer also accepts the public-project layout shown above. It
automatically checks the HaWoR locations when `assets/mano/` is absent.

## Data backends

Run the interactive installer only after reviewing the restrictions listed in
`THIRD_PARTY_NOTICES.md`:

```bash
bash scripts/install_data_backends.sh
mint doctor --profile data --strict
```

Some native CUDA extensions used by the full processing chain must be compiled
for the installed PyTorch and GPU architecture. Do not copy extensions from a
different Conda environment: extension ABI mismatches can produce silent
errors or crashes.

## Offline installation

For an offline host, populate a wheelhouse on a connected machine with the
same Linux, Python, PyTorch, and CUDA targets. Install with `--no-index` and a
trusted local `--find-links` directory. Keep the wheelhouse outside Git because
CUDA wheels are large and may carry their own redistribution terms.
