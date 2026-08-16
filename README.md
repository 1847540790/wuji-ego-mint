# wuji-ego-mint

[中文说明](README_ZH.md)

MINT is an open toolkit for egocentric video processing, camera-and-hand model
training, model-effect visualization, and benchmarking. It packages the Ray data
pipeline and training stack behind a small, reproducible command surface.

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

Open `http://127.0.0.1:8011`, then:

1. In **Model and Sample**, keep `checkpoints/model.safetensors` or select another compatible checkpoint.
2. Click **Load Model** and wait until the model status is ready.
3. Select a LeRobot episode and choose the camera, hand-window, and geometry settings.
4. Click **Start Inference**.
5. Inspect the synchronized GT/Pred 2D view, fixed-world and camera-frame 3D panels, per-frame values, losses, exports, and optional benchmark tools.

![MINT Web Viewer after loading the model and running inference](data/samples/mint-web-viewer.png)

For a remote server, forward the Viewer port before opening the same URL:

```bash
ssh -L 8011:127.0.0.1:8011 user@server
```

## What is included

| Area | Entry point | Purpose |
| --- | --- | --- |
| Data | `python -m mint pipeline` | Process videos into training-ready LeRobot v3 datasets with Ray. |
| Train | `python -m mint train` | Train the camera-and-hand MINT model with Accelerate/DDP. |
| Infer and view | `python -m mint viewer` | Use the original model_effect web UI for LeRobot GT, predictions, 2D/3D trajectories, and frame metrics. |
| Benchmark | `python eval/model_effect/benchmark/run.py` | Run the open benchmark CLI with user-provided data and environment. |
| Headless infer | `python -m mint infer` | Export overlays and numeric results for scripts and batch jobs. |
| Audit | `python -m mint doctor` | Verify the environment, optional backends, assets, and GPU runtime. |

The viewer starts directly in `data/samples/lerobot_v3/` and retains the
original directory browser and LeRobot ground-truth comparison workflow. Its
Benchmark button can launch local-GPU or Aliyun evaluation, and the same
benchmark remains available through the independent CLI.

## Installation Profiles and Viewer Options

MINT uses Python 3.10 and PyTorch 2.8 with CUDA 12.8. Both environments are
resolved from clean specifications; no existing Conda environment is cloned.
The environment script tries the system Conda channels first, unless their
resolved configuration contains a known-blocked TUNA or HIT endpoint. If the
system solve is skipped or fails, `--override-channels` strictly isolates the
retry and uses only the official conda-forge channel. No configured channel can
leak into the fallback, and global settings are never changed. The `full` and
`inference` profiles install the exact same inference foundation. NumPy,
PyTorch, and general Python packages use the USTC PyPI mirror by default; on
Linux x86_64, the pinned PyTorch 2.8.0 package installs its CUDA 12.8 runtime
dependencies. The full profile adds training, data, and development
dependencies only after that shared layer is fixed.

The viewer uses these project paths by default:

- sample: `data/samples/lerobot_v3/`;
- model: a training checkpoint discovered under `output/model_train/`, or `--ckpt`;
- configuration: `configs/training/stage2_resume_worldengine_camera_only.yaml`;
- cache: `wuji-viewer-cache/` under the system temporary directory, or `--cache-dir`.

Open `http://127.0.0.1:8011`. The viewer displays LeRobot GT and, after the
explicit Load Model and Start Inference actions, renders GT/prediction overlays,
fixed-world and camera-frame 3D, numeric values, and frame losses. The server
binds to `0.0.0.0` by default; use SSH forwarding or an authenticated reverse
proxy for remote access.

Run `bash scripts/download_assets.sh` if the default model is absent. Override
the checkpoint only when using another compatible model:

```bash
python -m mint viewer --ckpt /path/to/model.safetensors
```

The inference profile omits Ray, dataset conversion, training loggers, and
data-pipeline research backends. Mesh rendering also requires separately
licensed MANO files. See [Installation](docs/installation.md) for CUDA, MANO,
and offline installation details.

### Optional headless inference

`mint infer` is a command-line alternative to the viewer, not a prerequisite.
Use it for automation or direct artifact export:

```bash
python -m mint infer \
  --input /path/to/video.mp4 \
  --checkpoint checkpoints/model.safetensors \
  --output artifacts/example
```

## Data processing and training

Data processing and training require the full environment:

```bash
bash scripts/create_env.sh full
conda activate mint
python -m mint doctor --profile full
```

GeoCalib, MoGe, and Mega-SAM source is bundled under `third_party/`. The adapted
HaWoR infra copy is available locally but remains Git-ignored because its
CC BY-NC-ND terms prohibit redistribution of modifications. Weights, MANO, and
other separately licensed assets are not bundled. Review `THIRD_PARTY_NOTICES.md`,
register the local source, and verify the data profile:

```bash
bash scripts/install_data_backends.sh
python -m mint doctor --profile data
```

### Process approved videos with Ray

```bash
python -m mint pipeline \
  --input data/samples \
  --output output/processed \
  --num-gpus 1
```

Run `python -m mint doctor --profile data` before a long job and begin with one short,
non-sensitive clip.

`mint pipeline` uses Ray to turn source videos into training-ready LeRobot v3
data. It does not start the viewer or display model predictions.

### Train

Only the two configurations associated with the selected checkpoints are kept. `step_00019000` is Stage 1; `step_00004500` is the Stage 2 WorldEngine camera-only adaptation initialized from Stage 1:

```bash
python -m mint train --config configs/training/stage1_lingbotmap_distill_axis_angle_refine.yaml --inspect
python -m mint train --config configs/training/stage1_lingbotmap_distill_axis_angle_refine.yaml

python -m mint train --config configs/training/stage2_resume_worldengine_camera_only.yaml --inspect
python -m mint train --config configs/training/stage2_resume_worldengine_camera_only.yaml
```

The Stage 2 `train.init_from` points to the Stage 1
`step_00019000/model.safetensors`. `--inspect` constructs the model without
loading a dataset and is the recommended first configuration check.

`mint train` consumes the processed dataset and writes training checkpoints;
it does not require the viewer. To inspect a trained model interactively, run
`python -m mint viewer --ckpt /path/to/checkpoint` afterward.

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

```bash
python eval/model_effect/benchmark/run.py \
  --ckpt /path/to/checkpoint \
  --config configs/training/stage2_resume_worldengine_camera_only.yaml \
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
- [Inference and viewer](docs/inference.md)
- [Privacy and release checklist](docs/privacy.md)
- [Security policy](SECURITY.md)

## Acknowledgements

MINT is made possible by the following research projects, open-source libraries,
tools, and datasets. The first three acknowledgements reflect the primary
technical foundations of this repository.

- **VITRA** — MINT's data-processing architecture, egocentric reconstruction workflow, world-space camera/hand annotations, and LeRobot conversion conventions evolved from the VITRA and VITRA-1M data engine.
- **[LingBot-Map](https://github.com/robbyant/lingbot-map)** — provides the core model architecture and the upstream source adapted for MINT camera-and-hand training and inference.
- **[HaWoR](https://github.com/ThunderVVV/HaWoR)** — provides monocular hand motion reconstruction, MANO estimation, tracking, and world-space hand-processing components used by the optional data pipeline. Its use remains subject to the upstream non-commercial, no-derivatives license.
- **Camera, depth, and tracking research** — [GeoCalib](https://github.com/cvg/GeoCalib), [MoGe](https://github.com/microsoft/MoGe), [Mega-SAM](https://github.com/mega-sam/mega-sam), [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM), [UniDepth](https://github.com/lpiccinelli-eth/UniDepth), [Metric3D](https://github.com/YvanYin/Metric3D), [DeepCalib](https://github.com/alexvbogdan/DeepCalib), [DINOv2](https://github.com/facebookresearch/dinov2), [VGGT](https://github.com/facebookresearch/vggt), InfiniteVGGT, and [PyTorch3D](https://github.com/facebookresearch/pytorch3d).
- **Hand models, simulation, and retargeting** — [MANO](https://mano.is.tue.mpg.de), [SMPL-X](https://smpl-x.is.tue.mpg.de), [MuJoCo](https://mujoco.org), and the Wuji hand description and retargeting components used by the optional Viewer panels.
- **Model training and distribution** — [PyTorch](https://pytorch.org), [TorchVision](https://github.com/pytorch/vision), [TorchAO](https://github.com/pytorch/ao), [timm](https://github.com/huggingface/pytorch-image-models), [einops](https://github.com/arogozhnikov/einops), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [Accelerate](https://github.com/huggingface/accelerate), [Hugging Face Hub](https://github.com/huggingface/huggingface_hub), [ModelScope](https://github.com/modelscope/modelscope), [Safetensors](https://github.com/huggingface/safetensors), and [Weights & Biases](https://wandb.ai).
- **Data, orchestration, and media** — [Ray](https://github.com/ray-project/ray), [LeRobot](https://github.com/huggingface/lerobot), [NumPy](https://numpy.org), [SciPy](https://scipy.org), [pandas](https://pandas.pydata.org), [Apache Arrow/PyArrow](https://arrow.apache.org), [OpenCV](https://opencv.org), [Decord](https://github.com/dmlc/decord), [FFmpeg](https://ffmpeg.org), [PyYAML](https://pyyaml.org), [tqdm](https://github.com/tqdm/tqdm), [joblib](https://joblib.readthedocs.io), [natsort](https://github.com/SethMMorton/natsort), [psutil](https://github.com/giampaolo/psutil), and [NVIDIA ML Python](https://pypi.org/project/nvidia-ml-py/).
- **Viewer, evaluation, and optional integrations** — [Flask](https://flask.palletsprojects.com), [Matplotlib](https://matplotlib.org), [Ultralytics](https://github.com/ultralytics/ultralytics), [TensorFlow](https://www.tensorflow.org), and [Project Aria Tools](https://github.com/facebookresearch/projectaria_tools).
- **Datasets and benchmarks** — [HOT3D](https://github.com/facebookresearch/hot3d), [ARCTIC](https://arctic.is.tue.mpg.de), [Ego4D](https://ego4d-data.org), [EPIC-KITCHENS](https://epic-kitchens.github.io), and [EgoDex](https://ego-dex.github.io). Dataset access and redistribution remain governed by each dataset's own terms.
- **Development and packaging tools** — [pytest](https://pytest.org), [Ruff](https://github.com/astral-sh/ruff), [pre-commit](https://pre-commit.com), [setuptools](https://github.com/pypa/setuptools), [wheel](https://github.com/pypa/wheel), [CMake](https://cmake.org), and [Ninja](https://ninja-build.org).

We thank all upstream authors and maintainers. This acknowledgement does not
replace their citation or license requirements; see
[Third-party notices](THIRD_PARTY_NOTICES.md) before use or distribution.

## License

wuji-ego-mint's original code is released under the MIT License. Upstream models,
datasets, MANO assets, vendored LingBot-Map files, and optional research
backends retain their own licenses. Review [Third-party notices](THIRD_PARTY_NOTICES.md)
before distribution.
