# wuji-ego-mint

[English](README.md)

MINT 是一个面向第一视角视频的数据处理、相机与手部模型训练、推理、模型效果可视化和 benchmark 工具包。项目将 Ray 数据管线与训练代码统一封装为稳定、可复现的命令行接口，便于研究者从零安装并快速验证。

公开 Git 仓库包含一个小于 20 MB 的 Hot3D LeRobot v3 样例和 benchmark 实现，但不包含完整数据集、模型权重、MANO 资产、访问凭证或私有基础设施配置。公开 MINT 模型托管于 Hugging Face 和 ModelScope，由下载脚本保存到本地忽略目录。Benchmark 所需数据、依赖和本地或云端运行环境由使用者自行准备。

## 快速开始：Web Viewer

Web Viewer 是 MINT 的主要入口，可以在一个界面中选择 checkpoint、加载 MINT 模型、对 LeRobot episode 运行推理，并查看 GT/预测叠加、相机轨迹、手部运动、逐帧指标和 benchmark 结果。

安装推理环境、下载公开 MINT 模型并启动 Viewer：

```bash
git clone git@github.com:1847540790/wuji-ego-mint.git
cd wuji-ego-mint
bash scripts/create_env.sh inference
conda activate mint-inference
bash scripts/download_assets.sh
python -m mint doctor --profile inference
python -m mint viewer
```

打开 `http://127.0.0.1:8011`，然后依次操作：

1. 在 **模型与样本** 中保留 `checkpoints/model.safetensors`，或者选择其他兼容 checkpoint。
2. 点击 **加载模型**，等待模型状态变为就绪。
3. 选择 LeRobot episode，并设置相机推理、手部拼窗和几何参数。
4. 点击 **开始推理**。
5. 查看同步的 GT/Pred 2D、固定世界和当前相机 3D、逐帧数值、loss、导出及可选 benchmark 工具。

![MINT Web Viewer 加载模型并完成推理后的界面](data/samples/mint-web-viewer.png)

远程服务器可先转发 Viewer 端口，再打开同一地址：

```bash
ssh -L 8011:127.0.0.1:8011 user@server
```

## 功能范围

| 模块 | 命令 | 用途 |
| --- | --- | --- |
| 数据处理 | `python -m mint pipeline` | 使用 Ray 将视频转换为可训练的 LeRobot v3 数据集。 |
| 模型训练 | `python -m mint train` | 使用 Accelerate/DDP 训练相机与手部 MINT 模型。 |
| 推理与可视化 | `python -m mint viewer` | 使用原版 model_effect Web 界面查看 LeRobot GT、模型预测、2D/3D 轨迹与逐帧指标。 |
| 模型评测 | `python eval/model_effect/benchmark/run.py` | 运行开源 benchmark CLI；数据集和运行环境由使用者配置。 |
| 无界面推理 | `python -m mint infer` | 面向脚本和批处理，对单个视频导出渲染视频和数值结果。 |
| 环境检查 | `python -m mint doctor` | 检查依赖、可选后端、模型资产和运行环境。 |

Viewer 默认直接进入 `data/samples/lerobot_v3/`，并保留原版目录浏览和 LeRobot GT 对照能力。顶栏的 Benchmark 按钮可直接启动本地 GPU 或 Aliyun 评测，同一套评测也可通过独立 CLI 使用。

## 安装配置与 Viewer 选项

MINT 使用 Python 3.10、PyTorch 2.8 和 CUDA 12.8。两个环境均通过独立配置从零求解，不会克隆任何已有 Conda 环境。
环境脚本首先使用系统现有的 Conda channels；若系统配置包含已知不可用的清华或 HIT 源，则直接跳过。创建失败后，脚本会通过 `--override-channels` 严格隔离系统源，并仅使用官方 conda-forge 重试。回退阶段不会混入任何已配置源，也不会修改用户的全局 Conda 配置。`full` 与 `inference` 共用完全相同的推理依赖层：NumPy、PyTorch 和普通 Python 包默认使用 USTC PyPI；Linux x86_64 上的 PyTorch 2.8.0 包会安装其锁定的 CUDA 12.8 运行时依赖。完整环境随后才追加训练、数据与开发依赖。

Viewer 默认读取以下项目路径：

- 样例：`data/samples/lerobot_v3/`
- 模型：`output/model_train/` 中可发现的训练 checkpoint，或命令行 `--ckpt`
- 配置：`configs/training/stage2_resume_worldengine_camera_only.yaml`
- 缓存：系统临时目录下的 `wuji-viewer-cache/`，或命令行 `--cache-dir`

浏览器打开 `http://127.0.0.1:8011`。Viewer 会展示 LeRobot 的 GT，并在显式点击“加载模型”和“开始推理”后生成 GT/Pred 2D overlay、固定世界/当前相机 3D、逐帧数值和 loss。服务默认监听 `0.0.0.0`；远程使用时建议通过 SSH 转发或带认证的反向代理访问。

默认模型不存在时，先运行 `bash scripts/download_assets.sh`；使用其他兼容模型时才需要覆盖路径：

```bash
python -m mint viewer --ckpt /path/to/model.safetensors
```

最小环境不安装 Ray、数据转换组件、训练日志组件和数据管线研究后端。网格渲染还需要按许可证单独放置 MANO 文件；CUDA、MANO 和离线安装说明见 [安装指南](docs/installation.md)。

### 可选：无界面推理

`mint infer` 是单视频的无界面推理入口，不是 Viewer 的前置步骤。自动化任务或只需要导出文件时可直接运行：

```bash
python -m mint infer \
  --input /path/to/video.mp4 \
  --checkpoint checkpoints/model.safetensors \
  --output artifacts/example
```

## 数据处理与训练

数据处理和训练需要完整环境：

```bash
bash scripts/create_env.sh full
conda activate mint
python -m mint doctor --profile full
```

GeoCalib、MoGe、Mega-SAM 源码随项目放在 `third_party/`。当前机器的 HaWoR
适配源码也位于该目录，但受 CC BY-NC-ND 限制而保持 Git 忽略；公开或商业分发需另行授权。
所有权重和 MANO 等单独授权资产均不随仓库发布。先阅读 `THIRD_PARTY_NOTICES.md`，再注册本地源码并检查数据管线：

```bash
bash scripts/install_data_backends.sh
python -m mint doctor --profile data
```

### 使用 Ray 处理已审核视频

```bash
python -m mint pipeline \
  --input data/samples \
  --output output/processed \
  --num-gpus 1
```

完整管线需要较多 GPU 资源。长任务开始前请运行 `python -m mint doctor --profile data`，并先使用一段短且不含敏感信息的视频验证流程。

`mint pipeline` 使用 Ray 调度研究后端，将原始视频转换为可供训练使用的 LeRobot v3 数据；它不会启动 Viewer，也不负责模型推理展示。

### 训练模型

训练配置只保留与两个指定 checkpoint 对应的两阶段配置。`step_00019000` 是 Stage 1，`step_00004500` 是从 Stage 1 权重初始化、仅训练 WorldEngine 相机头的 Stage 2：

```bash
python -m mint train --config configs/training/stage1_lingbotmap_distill_axis_angle_refine.yaml --inspect
python -m mint train --config configs/training/stage1_lingbotmap_distill_axis_angle_refine.yaml

python -m mint train --config configs/training/stage2_resume_worldengine_camera_only.yaml --inspect
python -m mint train --config configs/training/stage2_resume_worldengine_camera_only.yaml
```

Stage 2 配置中的 `train.init_from` 已指向 Stage 1 的 `step_00019000/model.safetensors`。`--inspect` 不读取数据集，仅构建模型并输出结构，建议将它作为训练配置的第一项检查。

`mint train` 消费数据管线生成的数据集并产出训练 checkpoint；它同样不需要 Viewer。训练完成后，如需交互检查新模型，再使用 `python -m mint viewer --ckpt /path/to/checkpoint`。

## LeRobot 样例与隐私

仓库只保留 `data/samples/lerobot_v3/` 这一套小样例：将 Hot3D 序列排序后取正中间的两个序列，分别截取居中 15 秒，合并为包含 2 个 episode、900 帧的标准 LeRobot v3 数据集。它包含视频、Hot3D 相机/双手标注、任务文本和 episode 索引，整体小于 20 MB；样例不记录参与者 ID 或原始序列名。

可从本地已有的三个完整 export 重建：

```bash
python scripts/build_sample_lerobot.py \
  --source-root /path/to/hot3d_to_lerobot \
  --output data/samples/lerobot_v3
```

能够获取数据不等于拥有再次分发的权利；发布者仍需确认许可证、参与者授权并完成逐帧隐私检查。详细要求见 [隐私说明](docs/privacy.md)。

## Benchmark

`eval/model_effect/benchmark/` 与对应测试完整开源，并已接入 Viewer 顶栏的 Benchmark 面板。使用前需自行下载hot3d和arctic并按各 adapter 要求组织 benchmark 数据、安装可选依赖；也可独立运行 CLI：

```bash
python eval/model_effect/benchmark/run.py \
  --ckpt /path/to/checkpoint \
  --config configs/training/stage2_resume_worldengine_camera_only.yaml \
  --data-root /path/to/benchmark-data
```

相机轨迹数据也可通过 `CAMERA_TRAJECTORY_ROOT` 配置。Aliyun 分布式执行的 `defaults.yaml` 仅提供占位模板，workspace、resource、镜像、CPFS、凭证及环境均由使用者自行配置；项目不代建或维护 benchmark 环境。

## 仓库结构

```text
mint/
|-- configs/          两阶段训练配置与推理设置
|-- data/samples/     已审核的 Hot3D LeRobot v3 小样例
|-- eval/model_effect 原版可视化、推理适配器与 benchmark
|-- docs/             架构与运行文档（英文）
|-- environments/     完整环境与最小推理环境定义
|-- mint/             CLI、推理引擎、渲染器和 Viewer
|-- model_train/      训练引擎、模型、损失函数和数据加载器
|-- ray_pipeline/     Ray 调度、Actor、模型后端、轨迹清理、Manifest 和数据导出
|-- scripts/          环境、资产、隐私和样例处理脚本
`-- third_party/      可分发源码快照；HaWoR 适配源码仅本地，权重和授权资产不包含
```

## 文档

- [架构](docs/architecture.md)
- [安装](docs/installation.md)
- [数据管线](docs/data-pipeline.md)
- [训练](docs/training.md)
- [推理与 Viewer](docs/inference.md)
- [隐私与发布检查](docs/privacy.md)
- [安全策略](SECURITY.md)

## 致谢

MINT 的实现离不开以下研究项目、开源库、工具与数据集。前三项按照它们对本仓库的主要技术贡献排序。

- **VITRA** — MINT 的数据处理架构、第一视角重建流程、世界坐标系相机/手部标注以及 LeRobot 转换规范均由 VITRA 与 VITRA-1M 数据引擎演进而来。
- **[LingBot-Map](https://github.com/robbyant/lingbot-map)** — 提供 MINT 相机与手部训练、推理所使用的核心模型架构和上游适配源码。
- **[HaWoR](https://github.com/ThunderVVV/HaWoR)** — 为可选数据管线提供单目手部运动重建、MANO 估计、跟踪和世界坐标系手部处理组件；使用时仍须遵守其上游非商业、禁止演绎许可。
- **相机、深度与跟踪研究** — [GeoCalib](https://github.com/cvg/GeoCalib)、[MoGe](https://github.com/microsoft/MoGe)、[Mega-SAM](https://github.com/mega-sam/mega-sam)、[DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM)、[UniDepth](https://github.com/lpiccinelli-eth/UniDepth)、[Metric3D](https://github.com/YvanYin/Metric3D)、[DeepCalib](https://github.com/alexvbogdan/DeepCalib)、[DINOv2](https://github.com/facebookresearch/dinov2)、[VGGT](https://github.com/facebookresearch/vggt)、InfiniteVGGT 和 [PyTorch3D](https://github.com/facebookresearch/pytorch3d)。
- **手部模型、仿真与重定向** — [MANO](https://mano.is.tue.mpg.de)、[SMPL-X](https://smpl-x.is.tue.mpg.de)、[MuJoCo](https://mujoco.org)，以及 Viewer 可选面板使用的 Wuji hand description 与 retargeting 组件。
- **模型训练与分发** — [PyTorch](https://pytorch.org)、[TorchVision](https://github.com/pytorch/vision)、[TorchAO](https://github.com/pytorch/ao)、[timm](https://github.com/huggingface/pytorch-image-models)、[einops](https://github.com/arogozhnikov/einops)、[FlashInfer](https://github.com/flashinfer-ai/flashinfer)、[Accelerate](https://github.com/huggingface/accelerate)、[Hugging Face Hub](https://github.com/huggingface/huggingface_hub)、[ModelScope](https://github.com/modelscope/modelscope)、[Safetensors](https://github.com/huggingface/safetensors) 和 [Weights & Biases](https://wandb.ai)。
- **数据、调度与媒体处理** — [Ray](https://github.com/ray-project/ray)、[LeRobot](https://github.com/huggingface/lerobot)、[NumPy](https://numpy.org)、[SciPy](https://scipy.org)、[pandas](https://pandas.pydata.org)、[Apache Arrow/PyArrow](https://arrow.apache.org)、[OpenCV](https://opencv.org)、[Decord](https://github.com/dmlc/decord)、[FFmpeg](https://ffmpeg.org)、[PyYAML](https://pyyaml.org)、[tqdm](https://github.com/tqdm/tqdm)、[joblib](https://joblib.readthedocs.io)、[natsort](https://github.com/SethMMorton/natsort)、[psutil](https://github.com/giampaolo/psutil) 和 [NVIDIA ML Python](https://pypi.org/project/nvidia-ml-py/)。
- **Viewer、评测与可选集成** — [Flask](https://flask.palletsprojects.com)、[Matplotlib](https://matplotlib.org)、[Ultralytics](https://github.com/ultralytics/ultralytics)、[TensorFlow](https://www.tensorflow.org) 和 [Project Aria Tools](https://github.com/facebookresearch/projectaria_tools)。
- **数据集与 benchmark** — [HOT3D](https://github.com/facebookresearch/hot3d)、[ARCTIC](https://arctic.is.tue.mpg.de)、[Ego4D](https://ego4d-data.org)、[EPIC-KITCHENS](https://epic-kitchens.github.io) 和 [EgoDex](https://ego-dex.github.io)；数据访问和再分发始终以各数据集自身条款为准。
- **开发与打包工具** — [pytest](https://pytest.org)、[Ruff](https://github.com/astral-sh/ruff)、[pre-commit](https://pre-commit.com)、[setuptools](https://github.com/pypa/setuptools)、[wheel](https://github.com/pypa/wheel)、[CMake](https://cmake.org) 和 [Ninja](https://ninja-build.org)。

感谢所有上游作者和维护者。致谢不能替代引用或许可证义务；使用或分发前请阅读[第三方声明](THIRD_PARTY_NOTICES.md)。

## 许可证

wuji-ego-mint 原创代码使用 MIT License。上游模型、数据集、MANO 资产、内置的 LingBot-Map 源文件及可选研究后端仍遵循各自许可证。发布前请阅读 [第三方声明](THIRD_PARTY_NOTICES.md)。
