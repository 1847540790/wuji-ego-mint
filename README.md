# wuji-ego-mint

[English](README_EN.md)

MINT 是一个面向第一视角视频的数据处理、相机与手部模型训练、推理和结果可视化工具包。项目将 Ray 数据管线与训练代码统一封装为稳定、可复现的命令行接口，便于研究者从零安装并快速验证。

公开 Git 仓库不会包含数据集、模型权重、MANO 资产、访问凭证、云平台启动脚本、私有路径或基础设施配置。公开学生模型托管于 Hugging Face 和 ModelScope，由下载脚本保存到本地忽略目录。

## 功能范围

| 模块 | 命令 | 用途 |
| --- | --- | --- |
| 数据处理 | `python -m mint pipeline` | 使用 Ray 将视频转换为可训练的 LeRobot v3 数据集。 |
| 模型训练 | `python -m mint train` | 使用 Accelerate/DDP 训练相机与手部学生模型。 |
| 模型推理 | `python -m mint infer` | 对单个视频执行纯预测推理并导出渲染视频和数值结果。 |
| 结果查看 | `python -m mint viewer` | 在简洁的本地 Web 界面中浏览已审核样例和预测结果。 |
| 环境检查 | `python -m mint doctor` | 检查依赖、可选后端、模型资产和运行环境。 |

Viewer 不会读取或展示真值，也不能浏览配置样例目录之外的文件。这是项目明确的隐私与安全边界。

## 安装

MINT 使用 Python 3.10、PyTorch 2.8 和 CUDA 12.8。两个环境均通过独立配置从零求解，不会克隆任何已有 Conda 环境。
环境脚本首先使用系统现有的 Conda channels；若系统配置包含已知不可用的清华或 HIT 源，则直接跳过。创建失败后，脚本会通过 `--override-channels` 严格隔离系统源，并依次尝试官方 conda-forge 和 USTC 镜像。回退阶段不会混入任何已配置源，也不会修改用户的全局 Conda 配置。

### 完整训练与数据环境

```bash
git clone git@github.com:1847540790/wuji-ego-mint.git
cd wuji-ego-mint
bash scripts/create_env.sh full
conda activate mint
python -m mint doctor --profile full
```

完整环境包含训练和 Ray 数据管线依赖。GeoCalib、MoGe、Mega-SAM、HaWoR 等研究后端及其权重不随仓库发布，确认各自许可证后再单独安装：

```bash
bash scripts/install_data_backends.sh
python -m mint doctor --profile data
```

### 最小推理环境

```bash
bash scripts/create_env.sh inference
conda activate mint-inference
python -m mint doctor --profile inference
```

最小环境不安装 Ray、数据转换组件、训练日志组件和数据管线研究后端。CUDA、MANO、权重和离线安装说明见 [安装指南](docs/installation.md)。

### 下载公开资产

```bash
bash scripts/download_assets.sh
```

该脚本会下载并校验两类文件：本仓库 GitHub Release 中可再分发的机器人手 URDF/mesh，以及 wuji-ego-mint 的 `model.safetensors`。学生模型优先从 [ModelScope](https://www.modelscope.cn/models/AsherZhu/mint_v1) 下载，再依次尝试 [Hugging Face](https://huggingface.co/ZZJAsher/mint_v1) 官方端点和 HF 镜像。脚本不会下载 LingBot-Map 预训练 backbone，也不会获取优化器或随机状态；所有大型文件均位于 Git 忽略目录。

若 Hugging Face 仓库要求身份验证，请先设置 `HF_TOKEN`。MANO 不会被脚本下载，必须按 [安装指南](docs/installation.md) 接受其独立许可证并手动放置。

## 快速开始

### 1. 处理已审核视频

```bash
python -m mint pipeline \
  --input data/samples \
  --output output/processed \
  --num-gpus 1
```

完整管线需要较多 GPU 资源。长任务开始前请运行 `python -m mint doctor --profile data`，并先使用一段短且不含敏感信息的视频验证流程。

### 2. 训练模型

在 `configs/training/lingbotmap_base.yaml` 中设置数据集根目录，然后运行：

```bash
python -m mint train --config configs/training/lingbotmap_base.yaml --inspect
python -m mint train --config configs/training/lingbotmap_base.yaml
```

`--inspect` 不读取数据集和权重，仅构建模型并输出结构，建议将它作为训练配置的第一项检查。

### 3. 执行推理

```bash
python -m mint infer \
  --input data/samples/example.mp4 \
  --checkpoint checkpoints/model.safetensors \
  --output artifacts/example
```

### 4. 启动 Viewer

```bash
python -m mint viewer \
  --samples data/samples \
  --checkpoint checkpoints/model.safetensors \
  --config configs/training/lingbotmap_base.yaml
```

浏览器打开 `http://127.0.0.1:7860`。服务默认仅监听本机；如需监听其他地址，请在受控网络或带认证的反向代理后使用。

## 测试样例与隐私

Ego4D、EPIC-KITCHENS 和 EgoDex 数据不会被自动打包或重新分发。能够获取数据不等于拥有再次分发的权利。只有在许可证、参与者授权和隐私检查均通过后，才能将样例放入 `data/samples/`。

```bash
python scripts/prepare_samples.py \
  --input /path/to/approved-clips \
  --output data/samples \
  --review-manifest data/samples/review.json
python scripts/privacy_audit.py --strict
```

样例准备工具会移除容器元数据、统一公开文件名、限制时长与分辨率，并支持应用人工指定的隐私遮罩。自动处理不能替代逐帧人工审核。详细要求见 [隐私说明](docs/privacy.md)。

## 仓库结构

```text
mint/
|-- configs/          可公开、可移植的配置模板
|-- data/samples/     仅存放已审核样例，默认为空
|-- data_cleaning/    轨迹清理与平滑算法
|-- docs/             架构与运行文档（英文）
|-- environments/     完整环境与最小推理环境定义
|-- mint/             CLI、推理引擎、渲染器和 Viewer
|-- model_train/      训练引擎、模型、损失函数和数据加载器
|-- modules/          数据管线模型适配层
|-- ray_pipeline/     Ray 调度、Actor、Manifest 和数据导出
|-- scripts/          环境、资产、隐私和样例处理脚本
`-- third_party/      仅保留清单；第三方源码和权重默认忽略
```

## 文档

- [架构](docs/architecture.md)
- [安装](docs/installation.md)
- [数据管线](docs/data-pipeline.md)
- [训练](docs/training.md)
- [推理与 Viewer](docs/inference.md)
- [隐私与发布检查](docs/privacy.md)
- [安全策略](SECURITY.md)

## 许可证

wuji-ego-mint 原创代码使用 MIT License。上游模型、数据集、MANO 资产、内置的 LingBot-Map 源文件及可选研究后端仍遵循各自许可证。发布前请阅读 [第三方声明](THIRD_PARTY_NOTICES.md)。
