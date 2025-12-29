"""评估入口脚本"""

import pyrootutils
from loguru import logger

# 自动定位项目根目录，加载 .env，并将根目录加入 pythonpath
root = pyrootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True, dotenv=True)

import hydra
from hydra.utils import instantiate
from lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from omegaconf import DictConfig

from src.utils import (
    extras,
    instantiate_callbacks,
    instantiate_loggers,
    task_wrapper,
)


@task_wrapper
def evaluate(cfg: DictConfig) -> dict:
    """评估流程。

    Args:
        cfg: Hydra 配置对象

    Returns:
        包含测试 metric 的字典
    """
    assert cfg.ckpt_path, "Checkpoint path (ckpt_path) must be specified for evaluation!"

    # 设置随机种子
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    # 1. 实例化 DataModule
    logger.info(f"Instantiating DataModule: <{cfg.data._target_}>")
    datamodule: LightningDataModule = instantiate(cfg.data)

    # 2. 实例化 Model
    logger.info(f"Instantiating Model: <{cfg.model._target_}>")
    model: LightningModule = instantiate(cfg.model)

    # 3. 实例化 Callbacks（评估时通常不需要太多回调）
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

    # 6. 执行测试
    logger.info(f"Starting evaluation with checkpoint: {cfg.ckpt_path}")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    # 7. 返回测试指标
    metrics = trainer.callback_metrics
    logger.info(f"Evaluation metrics: {metrics}")

    return metrics


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> dict:
    """主入口函数。

    Args:
        cfg: Hydra 配置对象

    Returns:
        评估指标字典
    """
    # 应用额外配置
    extras(cfg)

    # 执行评估
    metrics = evaluate(cfg)

    return metrics


if __name__ == "__main__":
    main()
