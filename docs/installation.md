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
python -m mint doctor --profile full --strict
```

This performs a clean solve from `environments/mint.yml`, then installs each
pinned requirements layer. It never clones another local environment. If the
environment already exists, the script exits rather than mutating it silently.
The initial solve uses the channels already configured on the host. If that
solve fails, the script automatically retries with an isolated temporary
configuration: official conda-forge first, followed by the USTC conda-forge
mirror. This fallback never edits `.condarc` or other global Conda settings.
Set `MINT_CONDA_FALLBACK_CHANNELS` to a space-separated list of trusted channel
URLs when another fallback order is required.
For a network that cannot reach the official PyTorch wheel CDN, set
`MINT_TORCH_INDEX_URL` for a compatible package index, or
`MINT_TORCH_FIND_LINKS` for a trusted flat wheel mirror, before running the
script. `MINT_PYPI_INDEX_URL` can override a broken system-level PyPI mirror.
The public defaults remain PyPI and the official CUDA 12.8 index.

## Clean inference environment

```bash
bash scripts/create_env.sh inference
conda activate mint-inference
python -m mint doctor --profile inference --strict
```

The environment script streams subprocess output directly. Conda runs in
verbose mode, and pip download/install progress bars are forced on, so long
CUDA wheel downloads no longer appear to hang silently.

If a download is interrupted, run the same command again. The script reuses
the existing environment and resumes package installation instead of requiring
the partially created environment to be deleted. Only activate the environment
and run `mint doctor` after the script prints that the environment is ready.

The inference profile includes PyTorch, OpenCV, Flask, numerical dependencies,
and FFmpeg. It excludes Ray, PyArrow, W&B, data conversion, model compilers,
and upstream processing backends.

## Model checkpoint

The asset helper downloads the public MINT checkpoint to
`checkpoints/model.safetensors`. It tries the
[Hugging Face release](https://huggingface.co/ZZJAsher/mint_v1) first and falls
back to the [ModelScope release](https://www.modelscope.cn/models/AsherZhu/mint_v1).
Set `HF_TOKEN` if Hugging Face requires authentication. The ModelScope CLI
included in both environments provides the fallback. The helper selects only
`model.safetensors`; it does not fetch optimizer or random-state files. You may
also place that file manually or pass another compatible checkpoint with
`--checkpoint`.

Do not commit checkpoints to Git. The model configuration must match the
checkpoint architecture.

The optional pretrained LingBot-Map backbone belongs at:

```text
assets/models/lingbot-map.pt
```

Obtain it from the official LingBot-Map release and review its model terms.
The helper below downloads the student checkpoint, that public backbone, and
the redistributable robot description bundle. Every file is checksum-verified
before it replaces the destination:

```bash
bash scripts/download_assets.sh
```

Downloads show a live progress bar and use `.part` files for safe resume. For
Hugging Face assets, the helper first tries the official endpoint. If the
connection does not start transferring at least 1 KiB/s within 30 seconds, it
automatically continues the same partial file through `https://hf-mirror.com`.
Run the same command again after any interruption; completed bytes are kept.
Set `MINT_HF_MIRROR` to use a different mirror, or tune the timeout with
`MINT_DOWNLOAD_SPEED_TIME`.

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
python -m mint doctor --profile data --strict
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
