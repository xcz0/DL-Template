# 深度学习项目模板

基于 PyTorch Lightning + Hydra 的深度学习项目模板，支持灵活的配置管理和实验跟踪。

## 特性

- 🔧 **Hydra 配置管理** - 模块化配置，命令行覆盖，多运行支持
- ⚡ **PyTorch Lightning** - 简化训练循环，自动混合精度，多 GPU 支持
- 📊 **实验跟踪** - TensorBoard、WandB、CSV 日志
- 🔍 **超参数搜索** - Optuna 集成
- 📦 **DVC 数据管理** - 数据版本控制
- 🛠️ **开发工具** - Ruff 代码检查，Make 命令

> 说明：MNIST 模型现已支持 `test` 流程（`test/loss`、`test/acc` 指标）。

## 项目结构

```
├── configs/                 # Hydra 配置文件
│   ├── _base.yaml          # 基础配置 (paths, hydra, callbacks, logger, trainer)
│   ├── train.yaml          # 训练入口配置
│   ├── eval.yaml           # 评估入口配置
│   ├── predict.yaml        # 预测入口配置
│   ├── debug.yaml          # 调试入口配置
│   ├── data/               # 数据集配置 (cifar10, mnist)
│   ├── debug/              # 调试配置 (default, limit, profiler)
│   ├── experiment/         # 实验配置
│   ├── hparams_search/     # 超参搜索配置 (optuna)
│   └── model/              # 模型配置
├── src/                     # 源代码
│   ├── train.py            # 训练脚本
│   ├── eval.py             # 评估脚本
│   ├── predict.py          # 预测脚本
│   ├── data/               # 数据模块
│   ├── models/             # 模型定义
│   └── utils/              # 工具函数
├── data/                    # 数据目录
├── logs/                    # 日志目录
└── saved_models/            # 保存的模型
```

## 快速开始

### 1. 安装依赖

```bash
make install
```

### 2. 训练模型

```bash
# 默认训练 (CIFAR-10 + ResNet)
make train

# 使用特定实验配置
make train-exp EXP=cifar_densenet

# 或直接使用 Python
uv run python src/train.py experiment=cifar_densenet trainer.max_epochs=50
```

### 3. 快速调试

```bash
# 调试模式 (CPU, fast_dev_run, 无日志)
make debug

# 限制步数调试
make debug-limit
```

### 4. 评估模型

```bash
make eval CKPT=/path/to/checkpoint.ckpt
```

### 5. 预测/推理

```bash
make predict CKPT=/path/to/checkpoint.ckpt INPUT=/path/to/images
```

### 6. 超参数搜索

```bash
make hparams-cifar   # CIFAR-10 超参搜索
make hparams-mnist   # MNIST 超参搜索
```

## Makefile 命令

运行 `make help` 查看所有可用命令：

| 命令 | 说明 |
|------|------|
| `make install` | 安装依赖并初始化 DVC |
| `make train` | 默认训练 |
| `make train-exp EXP=xxx` | 指定实验配置训练 |
| `make debug` | 调试模式 |
| `make eval CKPT=xxx` | 评估模型 |
| `make predict CKPT=xxx INPUT=xxx` | 预测/推理 |
| `make hparams-cifar` | CIFAR-10 超参搜索 |
| `make lint` | 代码检查与格式化 |
| `make tb` | 启动 TensorBoard |
| `make clean` | 清理缓存文件 |

## 配置系统

### 配置组

| 配置组 | 说明 | 可选值 |
|--------|------|--------|
| `data` | 数据集 | `cifar10`, `mnist` |
| `model` | 模型 | `cifar_resnet`, `cifar_densenet`, `cifar_googlenet`, `cifar_resnet_preact`, `mnist_cnn` |
| `debug` | 调试 | `default`, `limit`, `profiler` |
| `experiment` | 实验 | `cifar_resnet`, `cifar_densenet`, `cifar_googlenet`, `cifar_resnet_preact`, `mnist_lr_search` |

> **注意**: `callbacks`、`logger`、`trainer`、`paths`、`hydra` 已整合到 `_base.yaml`，通常无需单独修改。如需调整可直接在命令行覆盖，如 `trainer.max_epochs=200`。

### 配置覆盖示例

```bash
# 查看完整配置
uv run python src/train.py --cfg job

# 覆盖单个参数
uv run python src/train.py trainer.max_epochs=200 data.batch_size=128

# 使用 GPU 训练（覆盖默认 auto）
uv run python src/train.py trainer.accelerator=gpu trainer.devices=1

# 多 GPU 分布式训练
uv run python src/train.py trainer.accelerator=gpu trainer.devices=auto trainer.strategy=ddp

# 禁用日志（快速测试）
uv run python src/train.py logger=null

# 禁用回调
uv run python src/train.py callbacks=null
```

## 添加新实验

1. **创建模型配置** `configs/model/your_model.yaml`:
   ```yaml
   _target_: src.models.your_module.YourModule
   # 模型参数...
   ```

2. **创建实验配置** `configs/experiment/your_exp.yaml`:
   ```yaml
   # @package _global_
   defaults:
     - override /data: cifar10
     - override /model: your_model
   
   tags: ["your_exp"]
   trainer:
     max_epochs: 100
   ```

3. **运行实验**:
   ```bash
   uv run python src/train.py experiment=your_exp
   ```

## 数据管理 (DVC)

```bash
# 跟踪数据
dvc add data/your_dataset
git add data/your_dataset.dvc .gitignore
git commit -m "Add dataset"

# 推送到远程存储
dvc remote add -d myremote s3://mybucket/dvcstore
dvc push

# 拉取数据
dvc pull
```

## 目录说明

| 目录 | 说明 |
|------|------|
| `logs/runs/` | 训练日志和 checkpoint |
| `logs/multiruns/` | 超参搜索日志 |
| `saved_models/` | 手动保存的模型 |
| `data/` | 数据集存储 |

## License

MIT
