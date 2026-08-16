# wuji-ego-mint

[中文说明](README_ZH.md)

MINT is an open toolkit for egocentric camera-and-hand inference, model
training, model-effect visualization, and benchmarking. The public repository
also exposes reusable Ray orchestration, data-flow, cleaning, and LeRobot export
components, but it is not a complete redistributable copy of the production
data-generation pipeline.

The public Git repository includes a sub-20 MB Hot3D LeRobot v3 sample
and the benchmark implementation. It excludes full datasets, model checkpoints,
MANO assets, credentials, and private infrastructure settings. Benchmark data,
optional dependencies, and local or cloud runtime setup are user-managed. The public MINT
model is hosted on Hugging Face and ModelScope and is downloaded only into
a local Git-ignored directory.

## Quick Start: Web Viewer

The Web Viewer is the primary MINT entry point. It lets you select a checkpoint,
load the MINT model, run inference on a LeRobot episode, and inspect GT/prediction
overlays, camera trajectories, hand motion, frame metrics, and benchmark results
in one interface.

**Hardware requirement: MINT model inference requires an NVIDIA GPU with at
least 24 GB of VRAM.**

Install the inference environment, download the public MINT model, and launch
the Viewer:

```bash
git clone git@github.com:1847540790/wuji-ego-mint.git
cd wuji-ego-mint
bash scripts/create_env.sh inference
conda activate mint-inference
bash scripts/download_assets.sh
python -m mint doctor --profile inference
python -m mint viewer
```

The Viewer automatically opens `http://127.0.0.1:8011` in the default browser. Then:

1. In **Model and Sample**, keep `checkpoints/model.safetensors` or select another compatible checkpoint.
2. Click **Load Model** and wait until the model status is ready.
3. Select a LeRobot episode and choose the camera, hand-window, and geometry settings.
4. Click **Start Inference**.
5. Inspect the synchronized GT/Pred 2D view, fixed-world and camera-frame 3D panels, per-frame values, losses, exports, and optional benchmark tools.

![MINT Web Viewer after loading the model and running inference](data/samples/mint-web-viewer.png)

## What is included

| Area | Entry point | Purpose |
| --- | --- | --- |
| Pipeline reference | `ray_pipeline/` | Reuse the open Ray orchestration, interfaces, cleaning, and LeRobot export code after integrating the required upstream backends locally. |
| Train | `python -m mint train` | Train the camera-and-hand MINT model with Accelerate/DDP. |
| Infer and view | `python -m mint viewer` | Use the original model_effect web UI for LeRobot GT, predictions, 2D/3D trajectories, and frame metrics. |
| Benchmark | `python eval/model_effect/benchmark/run.py` | Run the open benchmark CLI with user-provided data and environment. |
| Audit | `python -m mint doctor` | Verify the environment, optional backends, assets, and GPU runtime. |

The Viewer opens automatically after startup. Model and checkpoint selection,
sample and LeRobot episode browsing, model loading, inference, GT/prediction
comparison, 2D/3D visualization, frame values and losses, exports, and
Benchmark operations are all available directly in the Viewer panel. No
additional command-line visualization step is required.

## Installation Profiles and Viewer Options

All visualization operations are available in the Viewer panel. See
[Installation](docs/installation.md) for installation, CUDA, MANO, and offline
deployment details.

## Training and optional pipeline reconstruction

Training or local pipeline-development work requires the full environment:

```bash
bash scripts/create_env.sh full
conda activate mint
python -m mint doctor --profile full
```

The public release uses the Viewer as the unified entry point for MINT
inference, visualization, and artifact export. This project publishes only code
that third-party licenses permit us to distribute; license-restricted
adaptations and internal integrations are excluded, so this repository does not
contain the complete production data pipeline.

GeoCalib, MoGe, and Mega-SAM source snapshots are distributed under
`third_party/`, but some production adaptations cannot be published under their
upstream license terms. In particular, the locally adapted HaWoR source is
Git-ignored because CC BY-NC-ND prohibits redistribution of modifications.
Weights, MANO files, and other separately licensed assets are also excluded.

If you need to reconstruct the data-generation pipeline, first read
`THIRD_PARTY_NOTICES.md`, obtain and install every required upstream library and
asset under its own terms, and implement the necessary compatibility adapters
locally. The open code in `ray_pipeline/` documents the orchestration,
interfaces, data flow, cleaning, manifests, and LeRobot export contracts. An AI
coding assistant may help reconcile upstream API differences, but the resulting
integration remains the user's responsibility and must comply with all licenses.
Only after completing that local integration should the data-profile doctor and
`python -m mint pipeline` be treated as usable entry points.

### Train

Only the two configurations associated with the selected checkpoints are kept. `step_00019000` is Stage 1; `step_00004500` is the Stage 2 WorldEngine camera-only adaptation initialized from Stage 1:

```bash
python -m mint train --config configs/training/mint_step1.yaml --inspect
python -m mint train --config configs/training/mint_step1.yaml

python -m mint train --config configs/training/mint_step2.yaml --inspect
python -m mint train --config configs/training/mint_step2.yaml
```

The Stage 2 `train.init_from` points to the Stage 1
`step_00019000/model.safetensors`. `--inspect` constructs the model without
loading a dataset and is the recommended first configuration check.

`mint train` consumes a compatible, separately prepared LeRobot dataset and
writes training checkpoints. After training, select the new checkpoint directly
in the Viewer panel for interactive inspection.

## LeRobot sample

`data/samples/lerobot_v3/` is the only bundled sample. It takes the two middle
entries after sorting the Hot3D sequence exports and crops the centered 15
seconds from each, producing a valid two-episode, 900-frame LeRobot v3 dataset
with synchronized video, camera/hand labels, task text, and episode metadata.
Participant IDs and original sequence names are not stored in the sample.

Rebuild it from local full exports with:

```bash
python scripts/build_sample_lerobot.py \
  --source-root /path/to/hot3d_to_lerobot \
  --output data/samples/lerobot_v3
```

Dataset access does not itself grant redistribution rights. The publisher must
still confirm licensing, participant consent, and frame-by-frame privacy review.

## Benchmark

The complete implementation and tests live in `eval/model_effect/benchmark/`
and are integrated into the Viewer's Benchmark panel. Users provide datasets
and install the optional runtime themselves; the CLI remains available:

**Benchmark integrity statement.** We commit that every metric reported by this
project is an authentic result produced under the stated evaluation protocol;
we do not alter the original numeric results. To reproduce a baseline or another
method's exact values, use that method's official repository and environment.
The metric definitions, alignment rules, aggregation logic, and reporting code
used by MINT are available in `eval/model_effect/benchmark/` for inspection.

```bash
python eval/model_effect/benchmark/run.py \
  --ckpt /path/to/checkpoint \
  --config configs/training/mint_step2.yaml \
  --data-root /path/to/benchmark-data
```

Set `CAMERA_TRAJECTORY_ROOT` for camera-trajectory exports when needed. Aliyun
defaults are placeholders; users must configure the workspace, resource, image,
CPFS, credentials, and environment. This project does not provision or maintain
benchmark environments.

## Repository layout

```text
mint/
|-- configs/          Two-stage training recipes and inference settings
|-- data/samples/     Approved Hot3D LeRobot v3 sample
|-- eval/model_effect Original visualization, inference adapters, and benchmarks
|-- docs/             Architecture and operational guides
|-- environments/     Full and inference-only dependency specifications
|-- mint/             CLI, inference engine, renderer, and web viewer
|-- model_train/      Training engine, model, losses, and LeRobot loader
|-- ray_pipeline/     Ray scheduling, actors, model backends, trajectory cleanup, manifests, and export
|-- scripts/          Reproducible setup, asset, privacy, and sample tools
`-- third_party/      Redistributable source snapshot; adapted HaWoR is local-only, assets excluded
```

## Documentation

- [Architecture](docs/architecture.md)
- [Installation](docs/installation.md)
- [Data pipeline](docs/data-pipeline.md)
- [Training](docs/training.md)
- [LeRobot training data format](docs/lerobot-training-data.md)
- [Inference and viewer](docs/inference.md)
- [Privacy and release checklist](docs/privacy.md)
- [Security policy](SECURITY.md)

## Data-pipeline reproducibility

The public release provides an implementation reference, not a one-command
reproduction of the production data generator. Use MINT directly for inference;
for pipeline reconstruction, supply the licensed upstream code and assets and
complete the local integration described in [Data pipeline](docs/data-pipeline.md).

## Acknowledgements

MINT is made possible by the following research projects, models, and datasets.

- **VITRA** — MINT's data-processing architecture, egocentric reconstruction workflow, world-space camera/hand annotations, and LeRobot conversion conventions evolved from the VITRA and VITRA-1M data engine.
- **[LingBot-Map](https://github.com/robbyant/lingbot-map)** — provides the core model architecture and the upstream source adapted for MINT camera-and-hand training and inference.
- **[HaWoR](https://github.com/ThunderVVV/HaWoR)** — provides monocular hand motion reconstruction, MANO estimation, tracking, and world-space hand-processing components used by the optional data pipeline. Its use remains subject to the upstream non-commercial, no-derivatives license.
- **Camera, depth, and tracking research** — [GeoCalib](https://github.com/cvg/GeoCalib), [MoGe](https://github.com/microsoft/MoGe), [Mega-SAM](https://github.com/mega-sam/mega-sam), [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM), [UniDepth](https://github.com/lpiccinelli-eth/UniDepth), [Metric3D](https://github.com/YvanYin/Metric3D), [DeepCalib](https://github.com/alexvbogdan/DeepCalib), [DINOv2](https://github.com/facebookresearch/dinov2), [VGGT](https://github.com/facebookresearch/vggt), InfiniteVGGT, and [PyTorch3D](https://github.com/facebookresearch/pytorch3d).
- **Hand models, simulation, and retargeting** — [MANO](https://mano.is.tue.mpg.de), [SMPL-X](https://smpl-x.is.tue.mpg.de), [MuJoCo](https://mujoco.org), and the Wuji hand description and retargeting components used by the optional Viewer panels.
- **Datasets and benchmarks** — [HOT3D](https://github.com/facebookresearch/hot3d), [ARCTIC](https://arctic.is.tue.mpg.de), [Ego4D](https://ego4d-data.org), [EPIC-KITCHENS](https://epic-kitchens.github.io), and [EgoDex](https://ego-dex.github.io). Dataset access and redistribution remain governed by each dataset's own terms.

We thank all upstream authors and maintainers. This acknowledgement does not
replace their citation or license requirements; see
[Third-party notices](THIRD_PARTY_NOTICES.md) before use or distribution.

## License

wuji-ego-mint's original code is released under the MIT License. Upstream models,
datasets, MANO assets, vendored LingBot-Map files, and optional research
backends retain their own licenses. Review [Third-party notices](THIRD_PARTY_NOTICES.md)
before distribution.
