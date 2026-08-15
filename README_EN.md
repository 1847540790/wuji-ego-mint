# wuji-ego-mint

[Chinese README](README.md)

MINT is an open toolkit for egocentric video processing, camera-and-hand model
training, and prediction-only visual inspection. It packages the Ray data
pipeline and training stack behind a small, reproducible command surface.

The public repository is self-contained at the orchestration layer and
intentionally excludes datasets, model checkpoints, MANO assets, credentials,
cloud-specific launchers, and private infrastructure settings.

## What is included

| Area | Entry point | Purpose |
| --- | --- | --- |
| Data | `mint pipeline` | Process videos into training-ready LeRobot v3 datasets with Ray. |
| Train | `mint train` | Train the camera-and-hand student model with Accelerate/DDP. |
| Infer | `mint infer` | Run prediction-only inference and export overlays plus numeric results. |
| View | `mint viewer` | Browse approved samples and inspect predictions in a focused web interface. |
| Audit | `mint doctor` | Verify the environment, optional backends, assets, and GPU runtime. |

The viewer never reads ground truth and cannot browse outside its configured
sample directory. This is a deliberate privacy and security boundary.

## Installation

MINT uses Python 3.10 and PyTorch 2.8 with CUDA 12.8. Both environments are
resolved from clean specifications; no existing Conda environment is cloned.

### Full environment

```bash
git clone git@github.com:1847540790/wuji-ego-mint.git
cd wuji-ego-mint
bash scripts/create_env.sh full
conda activate mint
mint doctor --profile full
```

The full profile installs the training and Ray data-pipeline dependencies.
Optional research backends and their weights must be installed separately
after reviewing their licenses:

```bash
bash scripts/install_data_backends.sh
mint doctor --profile data
```

### Minimal inference environment

```bash
bash scripts/create_env.sh inference
conda activate mint-inference
mint doctor --profile inference
```

The inference profile omits Ray, dataset conversion, training loggers, and
data-pipeline research backends. See [Installation](docs/installation.md) for
CUDA, MANO, checkpoints, and offline installation details.

### Download redistributable assets

```bash
bash scripts/download_assets.sh
```

The script downloads the pretrained backbone from the official LingBot-Map
Hugging Face repository and the redistributable robot-hand URDF/mesh bundle
from this repository's GitHub Release. A trained wuji-ego-mint student
checkpoint must be published separately and passed with `--checkpoint`; large
model weights are never committed to Git.

## Quick start

### Process approved videos

```bash
mint pipeline \
  --input data/samples \
  --output output/processed \
  --num-gpus 1
```

Run `mint doctor --profile data` before a long job and begin with one short,
non-sensitive clip.

### Train

Set the dataset root in `configs/training/lingbotmap_base.yaml`, then run:

```bash
mint train --config configs/training/lingbotmap_base.yaml --inspect
mint train --config configs/training/lingbotmap_base.yaml
```

`--inspect` constructs the model without loading a dataset or checkpoint and
is the recommended first configuration check.

### Run inference

```bash
mint infer \
  --video data/samples/example.mp4 \
  --checkpoint checkpoints/model.safetensors \
  --output artifacts/example
```

### Start the viewer

```bash
mint viewer \
  --samples data/samples \
  --checkpoint checkpoints/model.safetensors \
  --config configs/training/lingbotmap_base.yaml
```

Open `http://127.0.0.1:7860`. The server binds to localhost by default.
Exposing it on another interface requires appropriate access controls.

## Sample data policy

Ego4D, EPIC-KITCHENS, and EgoDex assets are not redistributed automatically.
Dataset availability does not automatically grant redistribution rights.
Place only explicitly approved clips in `data/samples/`, then run:

```bash
python scripts/prepare_samples.py \
  --input /path/to/approved-clips \
  --output data/samples \
  --review-manifest data/samples/review.json
python scripts/privacy_audit.py --strict
```

The preparation tool removes container metadata, normalizes filenames, limits
duration and resolution, and can apply user-supplied privacy masks. Automated
processing is not a substitute for frame-by-frame human review.

## Repository layout

```text
mint/
|-- configs/          Public, portable configuration templates
|-- data/samples/     Approved examples only; empty by default
|-- data_cleaning/    Trajectory cleaning and smoothing
|-- docs/             Architecture and operational guides
|-- environments/     Full and inference-only dependency specifications
|-- mint/             CLI, inference engine, renderer, and web viewer
|-- model_train/      Training engine, model, losses, and LeRobot loader
|-- modules/          Data-pipeline model adapters
|-- ray_pipeline/     Ray scheduling, actors, manifests, and export
|-- scripts/          Reproducible setup, asset, privacy, and sample tools
|-- tests/            Fast public contract and smoke tests
`-- third_party/      Manifest only; upstream source and weights are ignored
```

## Documentation

- [Architecture](docs/architecture.md)
- [Installation](docs/installation.md)
- [Data pipeline](docs/data-pipeline.md)
- [Training](docs/training.md)
- [Inference and viewer](docs/inference.md)
- [Privacy and release checklist](docs/privacy.md)
- [Security policy](SECURITY.md)

## License

wuji-ego-mint's original code is released under the MIT License. Upstream models,
datasets, MANO assets, vendored LingBot-Map files, and optional research
backends retain their own licenses. Review [Third-party notices](THIRD_PARTY_NOTICES.md)
before distribution.
