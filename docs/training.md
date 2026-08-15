# Training

## Dataset

The public loader expects a LeRobot v3 dataset with the camera, image, hand,
and hand-presence fields produced by the MINT pipeline. Set `data.root` in a
copy of `configs/training/lingbotmap_base.yaml`.

Dataset files are never part of the source distribution. Keep local dataset
configuration in an ignored file when paths reveal infrastructure details.

## Inspect first

```bash
python -m mint train --config configs/training/lingbotmap_base.yaml --inspect
```

Inspection builds the model on CPU, prints the module and freeze structure, and
skips pretrained weights and dataset loading. It catches registry and shape
configuration errors without allocating training GPUs.

## Train

```bash
python -m mint train --config configs/training/lingbotmap_base.yaml
```

Accelerate selects the visible GPU topology. For reproducibility, the public
configuration fixes model, data loader, and sampling seeds. Strictly
deterministic CUDA algorithms remain disabled because some required operations
do not provide deterministic implementations.

W&B is disabled by default. To enable it, authenticate through the W&B CLI or
an environment variable. Never store an API key in a YAML or Markdown file.

## Resume and initialize

- `--resume <step-directory>` restores model, optimizer, scheduler, random
  state, and global step from an Accelerate checkpoint.
- `--init-from <checkpoint>` loads model parameters only and starts a new
  optimizer and schedule.

These options are intentionally exclusive. Keep the resolved configuration
snapshot beside each run to make later inference reproducible.
