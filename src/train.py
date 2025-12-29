"""训练入口脚本"""

import pyrootutils
from loguru import logger

# 自动定位项目根目录，加载 .env，并将根目录加入 pythonpath
root = pyrootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True, dotenv=True)

import torch  # noqa: E402

# 优化 Tensor Cores 性能（适用于 RTX 20/30/40 系列等支持 Tensor Cores 的 GPU）
torch.set_float32_matmul_precision("medium")

import hydra
from hydra.utils import instantiate
from lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from omegaconf import DictConfig

from src.utils import (
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)


@task_wrapper
def train(cfg: DictConfig) -> dict:
    """训练流程。

    Args:
        cfg: Hydra 配置对象

    Returns:
        包含所有 metric 的字典
    """
    # 设置随机种子
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    # 1. 实例化 DataModule
    logger.info(f"Instantiating DataModule: <{cfg.data._target_}>")
    datamodule: LightningDataModule = instantiate(cfg.data)

    # 2. 实例化 Model
    logger.info(f"Instantiating Model: <{cfg.model._target_}>")
    model: LightningModule = instantiate(cfg.model)

    # 3. 实例化 Callbacks
    logger.info("Instantiating callbacks...")
    callbacks = instantiate_callbacks(cfg.get("callbacks"))

    # 4. 实例化 Loggers
    logger.info("Instantiating loggers...")
    loggers = instantiate_loggers(cfg.get("logger"))

    # 5. 实例化 Trainer
    logger.info(f"Instantiating Trainer: <{cfg.trainer._target_}>")
    trainer: Trainer = instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=loggers,
    )

    # 6. 记录超参数
    if loggers:
        logger.info("Logging hyperparameters...")
        log_hyperparameters(cfg, model, trainer)

    # 7. 训练
    if cfg.get("train"):
        logger.info("Starting training...")
        ckpt_path = cfg.get("ckpt_path")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    # 8. 获取最佳模型路径
    train_metrics = trainer.callback_metrics
    best_ckpt_path = None
    if trainer.checkpoint_callback is not None:
        best_ckpt_path = trainer.checkpoint_callback.best_model_path
        if best_ckpt_path:
            logger.info(f"Best checkpoint saved at: {best_ckpt_path}")

    # 9. 测试
    if cfg.get("test"):
        logger.info("Starting testing...")
        # 优先使用最佳模型，否则使用当前权重
        test_ckpt_path = best_ckpt_path or cfg.get("ckpt_path")
        if test_ckpt_path:
            logger.info(f"Testing with checkpoint: {test_ckpt_path}")
        else:
            logger.warning("No checkpoint found, using current model weights for testing.")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=test_ckpt_path)

    # 10. 合并所有 metrics
    test_metrics = trainer.callback_metrics
    all_metrics = {**train_metrics, **test_metrics}

    return all_metrics


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> float | None:
    """主入口函数。

    Args:
        cfg: Hydra 配置对象

    Returns:
        优化指标值（用于超参数搜索）
    """
    # 应用额外配置（忽略警告、打印配置等）
    extras(cfg)

    # 执行训练
    metrics = train(cfg)

    # 获取优化指标（用于 Optuna 等超参数搜索）
    metric_value = get_metric_value(metrics, cfg.get("optimized_metric"))

    return metric_value


if __name__ == "__main__":
    main()
