# wuji-ego-mint

[English](README.md)

MINT 是一个面向第一视角相机与手部模型推理、训练、效果可视化和 benchmark 的开源工具包。公开仓库同时提供可复用的 Ray 调度、数据流、轨迹清理和 LeRobot 导出组件，但不是生产级数据生成管线的完整可分发副本。

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

Viewer 启动后会自动在默认浏览器打开 `http://127.0.0.1:8011`，然后依次操作：

1. 在 **模型与样本** 中保留 `checkpoints/model.safetensors`，或者选择其他兼容 checkpoint。
2. 点击 **加载模型**，等待模型状态变为就绪。
3. 选择 LeRobot episode，并设置相机推理、手部拼窗和几何参数。
4. 点击 **开始推理**。
5. 查看同步的 GT/Pred 2D、固定世界和当前相机 3D、逐帧数值、loss、导出及可选 benchmark 工具。

![MINT Web Viewer 加载模型并完成推理后的界面](data/samples/mint-web-viewer.png)

## 功能范围

| 模块 | 命令 | 用途 |
| --- | --- | --- |
| 管线参考 | `ray_pipeline/` | 在用户本地整合所需上游后端后，复用已开源的 Ray 调度、接口、轨迹清理和 LeRobot 导出代码。 |
| 模型训练 | `python -m mint train` | 使用 Accelerate/DDP 训练相机与手部 MINT 模型。 |
| 推理与可视化 | `python -m mint viewer` | 使用原版 model_effect Web 界面查看 LeRobot GT、模型预测、2D/3D 轨迹与逐帧指标。 |
| 模型评测 | `python eval/model_effect/benchmark/run.py` | 运行开源 benchmark CLI；数据集和运行环境由使用者配置。 |
| 环境检查 | `python -m mint doctor` | 检查依赖、可选后端、模型资产和运行环境。 |

Viewer 启动后会自动打开浏览器。模型与 checkpoint 选择、样本和 LeRobot episode 浏览、模型加载、推理、GT/Pred 对比、2D/3D 可视化、逐帧数值与 loss、结果导出以及 Benchmark 均可直接在 Viewer 面板中完成，无需额外的命令行可视化步骤。

## 安装配置与 Viewer 选项

所有可视化操作均可在 Viewer 面板中完成；安装、CUDA、MANO 与离线部署细节见 [安装指南](docs/installation.md)。

## 模型训练与可选管线复现

模型训练或本地数据管线开发需要完整环境：

```bash
bash scripts/create_env.sh full
conda activate mint
python -m mint doctor --profile full
```

公开版本以 Viewer 作为 MINT 模型推理、可视化和结果导出的统一入口。本项目仅提供第三方许可证允许公开发布的代码；受许可证限制的第三方适配与内部集成不包含在本仓库中，因此本仓库不包含完整的生产数据流程。

GeoCalib、MoGe 和 Mega-SAM 的源码快照位于 `third_party/`，但生产管线使用的部分第三方适配代码受上游许可证限制，无法公开。其中，当前机器上修改过的 HaWoR 源码因 CC BY-NC-ND 禁止分发修改版而保持 Git 忽略。所有权重、MANO 文件和其他需单独授权的资产也不随仓库发布。

如果确实需要复现数据生成管线，请先阅读 `THIRD_PARTY_NOTICES.md`，按各自许可条款自行获取和安装所有上游库与资产，再在本地补充必要的兼容适配。`ray_pipeline/` 已开源的代码可作为调度、接口、数据流、轨迹清理、Manifest 和 LeRobot 导出协议的实现参考。可使用 AI 编程工具帮助理解上游 API 差异并完成兼容层，但整合结果及其许可合规性由使用者负责。只有在完成这些本地整合后，才应将 `python -m mint doctor --profile data` 和 `python -m mint pipeline` 视为可用入口。

### 训练模型

训练配置只保留与两个指定 checkpoint 对应的两阶段配置。`step_00019000` 是 Stage 1，`step_00004500` 是从 Stage 1 权重初始化、仅训练 WorldEngine 相机头的 Stage 2：

```bash
python -m mint train --config configs/training/mint_step1.yaml --inspect
python -m mint train --config configs/training/mint_step1.yaml

python -m mint train --config configs/training/mint_step2.yaml --inspect
python -m mint train --config configs/training/mint_step2.yaml
```

Stage 2 配置中的 `train.init_from` 已指向 Stage 1 的 `step_00019000/model.safetensors`。`--inspect` 不读取数据集，仅构建模型并输出结构，建议将它作为训练配置的第一项检查。

`mint train` 消费由使用者单独准备的兼容 LeRobot 数据集并产出训练 checkpoint。训练完成后，可直接在 Viewer 面板中选择并检查新的 checkpoint。

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

**Benchmark 结果诚信声明。** 我们承诺，本项目报告的每一项测试指标均为按照所述评测协议实际运行所得的真实结果，不会对原始数值进行任何人为修改。如需精确复现其他方法或 baseline 的具体数值，请直接使用对应方法的官方仓库与原始环境。MINT 使用的指标定义、对齐规则、聚合逻辑和报告代码均公开在 `eval/model_effect/benchmark/` 中，可直接检查具体计算方式。

```bash
python eval/model_effect/benchmark/run.py \
  --ckpt /path/to/checkpoint \
  --config configs/training/mint_step2.yaml \
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
- [LeRobot 训练数据格式](docs/lerobot-training-data.md)
- [推理与 Viewer](docs/inference.md)
- [隐私与发布检查](docs/privacy.md)
- [安全策略](SECURITY.md)

## 数据管线复现说明

公开版本提供的是实现参考，而不是生产数据生成器的一键复现。请直接使用 MINT 进行模型推理；如需复现数据管线，请自行准备已授权的上游源码和资产，并按[数据管线文档](docs/data-pipeline.md)完成本地整合。

## 致谢

MINT 的实现离不开以下研究项目、模型与数据集。

- **VITRA** — MINT 的数据处理架构、第一视角重建流程、世界坐标系相机/手部标注以及 LeRobot 转换规范均由 VITRA 与 VITRA-1M 数据引擎演进而来。
- **[LingBot-Map](https://github.com/robbyant/lingbot-map)** — 提供 MINT 相机与手部训练、推理所使用的核心模型架构和上游适配源码。
- **[HaWoR](https://github.com/ThunderVVV/HaWoR)** — 为可选数据管线提供单目手部运动重建、MANO 估计、跟踪和世界坐标系手部处理组件；使用时仍须遵守其上游非商业、禁止演绎许可。
- **相机、深度与跟踪研究** — [GeoCalib](https://github.com/cvg/GeoCalib)、[MoGe](https://github.com/microsoft/MoGe)、[Mega-SAM](https://github.com/mega-sam/mega-sam)、[DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM)、[UniDepth](https://github.com/lpiccinelli-eth/UniDepth)、[Metric3D](https://github.com/YvanYin/Metric3D)、[DeepCalib](https://github.com/alexvbogdan/DeepCalib)、[DINOv2](https://github.com/facebookresearch/dinov2)、[VGGT](https://github.com/facebookresearch/vggt)、InfiniteVGGT 和 [PyTorch3D](https://github.com/facebookresearch/pytorch3d)。
- **手部模型、仿真与重定向** — [MANO](https://mano.is.tue.mpg.de)、[SMPL-X](https://smpl-x.is.tue.mpg.de)、[MuJoCo](https://mujoco.org)，以及 Viewer 可选面板使用的 Wuji hand description 与 retargeting 组件。
- **数据集与 benchmark** — [HOT3D](https://github.com/facebookresearch/hot3d)、[ARCTIC](https://arctic.is.tue.mpg.de)、[Ego4D](https://ego4d-data.org)、[EPIC-KITCHENS](https://epic-kitchens.github.io) 和 [EgoDex](https://ego-dex.github.io)；数据访问和再分发始终以各数据集自身条款为准。

感谢所有上游作者和维护者。致谢不能替代引用或许可证义务；使用或分发前请阅读[第三方声明](THIRD_PARTY_NOTICES.md)。

## 许可证

wuji-ego-mint 原创代码使用 MIT License。上游模型、数据集、MANO 资产、内置的 LingBot-Map 源文件及可选研究后端仍遵循各自许可证。发布前请阅读 [第三方声明](THIRD_PARTY_NOTICES.md)。
