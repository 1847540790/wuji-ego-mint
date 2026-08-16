# Training

## Dataset

The loader expects LeRobot v3 datasets. The repository keeps exactly two
historical training configurations: Stage 1 uses four real-data roots and
Stage 2 uses the WorldEngine root.

The retained historical recipes intentionally point to the shared CPFS data
and checkpoint locations used by those runs. Update those paths only when
reproducing the stages in a different environment.

See [LeRobot v3 training data contract](lerobot-training-data.md) for the exact
directory layout, Parquet columns, coordinate conventions, masks, video-frame
alignment, and clip construction used by these recipes.

## Inspect first

```bash
python -m mint train --config configs/training/stage1_lingbotmap_distill_axis_angle_refine.yaml --inspect
python -m mint train --config configs/training/stage2_resume_worldengine_camera_only.yaml --inspect
```

Inspection builds the model on CPU, prints the module and freeze structure, and
skips pretrained weights and dataset loading. It catches registry and shape
configuration errors without allocating training GPUs.

## Train

```bash
python -m mint train --config configs/training/stage1_lingbotmap_distill_axis_angle_refine.yaml
python -m mint train --config configs/training/stage2_resume_worldengine_camera_only.yaml
```

Stage 1 reproduces the H20 `step_00019000` run. Stage 2 initializes from that
model, freezes the aggregator, FoV, and hand modules, and trains only the camera
head on WorldEngine data. Accelerate selects the visible GPU topology. Strictly
deterministic CUDA algorithms remain disabled because some required operations
do not provide deterministic implementations.

Both historical configurations keep the original online W&B setting and refer
to the existing external credential file; the credential itself is not copied
into this repository.

## Resume and initialize

- `--resume <step-directory>` restores model, optimizer, scheduler, random
  state, and global step from an Accelerate checkpoint.
- `--init-from <checkpoint>` loads model parameters only and starts a new
  optimizer and schedule.

These options are intentionally exclusive. Keep the resolved configuration
snapshot beside each run to make later inference reproducible.
