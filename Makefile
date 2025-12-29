.PHONY: install debug train eval predict lint tb clean help

# 默认目标：显示帮助
.DEFAULT_GOAL := help

# ============== 环境设置 ==============

install:  ## 安装依赖并初始化 DVC
	uv sync
	dvc init --no-scm || true

# ============== 文件下载 ==============

download:  ## 下载需要的模型权重或文件
	./scripts/download.sh

# ============== 训练与评估 ==============

debug:  ## 调试模式 (CPU, fast_dev_run, 无日志)
	uv run python src/train.py debug=default

debug-limit:  ## 调试模式 (CPU, 限制步数)
	uv run python src/train.py debug=limit

train:  ## 默认训练 (CIFAR-10 + ResNet)
	uv run python src/train.py

train-exp:  ## 使用实验配置训练 (用法: make train-exp EXP=cifar_densenet)
	uv run python src/train.py experiment=$(EXP)

eval:  ## 评估模型 (用法: make eval CKPT=/path/to/checkpoint.ckpt)
	uv run python src/eval.py ckpt_path=$(CKPT)

predict:  ## 预测/推理 (用法: make predict CKPT=/path/to/ckpt INPUT=/path/to/images)
	uv run python src/predict.py ckpt_path=$(CKPT) input_path=$(INPUT)

# ============== 超参数搜索 ==============

hparams-cifar:  ## CIFAR-10 超参数搜索 (Optuna)
	uv run python src/train.py -m hparams_search=cifar_optuna

hparams-mnist:  ## MNIST 超参数搜索 (Optuna)
	uv run python src/train.py -m hparams_search=mnist_optuna

# ============== 代码质量 ==============

lint:  ## 代码检查与格式化
	uv run ruff check . --fix
	uv run ruff format .

# ============== 可视化与工具 ==============

tb:  ## 启动 TensorBoard
	./scripts/tensorboard.sh

compare:  ## 对比实验结果
	./scripts/compare_runs.sh

# ============== 清理 ==============

clean:  ## 清理缓存文件
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .ruff_cache 2>/dev/null || true

clean-logs:  ## 清理日志文件 (谨慎使用)
	rm -rf logs/runs/* 2>/dev/null || true
	rm -rf outputs/* 2>/dev/null || true

# ============== 帮助 ==============

help:  ## 显示帮助信息
	@echo "深度学习项目模板 - 可用命令:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "示例:"
	@echo "  make train                           # 默认训练"
	@echo "  make train-exp EXP=cifar_densenet    # 指定实验配置"
	@echo "  make eval CKPT=saved_models/best.ckpt"
	@echo "  make predict CKPT=model.ckpt INPUT=./images"
