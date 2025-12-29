"""工具函数集合"""

import warnings
from importlib.util import find_spec
from typing import Any, Callable

from omegaconf import DictConfig


def extras(cfg: DictConfig) -> None:
    """应用配置中指定的额外功能。

    Args:
        cfg: Hydra 配置对象
    """
    # 忽略警告
    if cfg.get("ignore_warnings"):
        warnings.filterwarnings("ignore")

    # 打印完整配置
    if cfg.get("print_config"):
        print_config_tree(cfg)


def print_config_tree(cfg: DictConfig, resolve: bool = True) -> None:
    """使用 Rich 库打印配置树。

    Args:
        cfg: Hydra 配置对象
        resolve: 是否解析配置中的插值
    """
    from omegaconf import OmegaConf

    try:
        from rich import print as rich_print
        from rich.syntax import Syntax
        from rich.tree import Tree

        tree = Tree("CONFIG", style="bold blue")

        # 添加主要配置组
        queue = []
        for field in ["data", "model", "callbacks", "logger", "trainer", "paths"]:
            if field in cfg:
                queue.append(field)

        # 添加其他字段
        for field in cfg:
            if field not in queue:
                queue.append(field)

        for field in queue:
            branch = tree.add(field, style="bold green")
            config_group = cfg.get(field)
            if isinstance(config_group, DictConfig):
                content = OmegaConf.to_yaml(config_group, resolve=resolve)
            else:
                content = str(config_group)
            branch.add(Syntax(content, "yaml", theme="monokai", line_numbers=False))

        rich_print(tree)

    except ImportError:
        # 如果没有 rich，使用简单打印
        from omegaconf import OmegaConf

        print(OmegaConf.to_yaml(cfg, resolve=resolve))


def get_metric_value(metric_dict: dict[str, Any], metric_name: str | None) -> float | None:
    """从 metric 字典中安全地获取指定 metric 的值。

    Args:
        metric_dict: 包含所有 metric 的字典
        metric_name: 要获取的 metric 名称

    Returns:
        metric 值，如果不存在则返回 None
    """
    if metric_name is None:
        return None

    if metric_name not in metric_dict:
        raise ValueError(
            f"Metric '{metric_name}' not found in metric_dict. Available metrics: {list(metric_dict.keys())}"
        )

    metric_value = metric_dict[metric_name].item()
    return metric_value


def task_wrapper(task_func: Callable) -> Callable:
    """装饰器，用于包装任务函数，提供异常处理。

    Args:
        task_func: 要包装的任务函数

    Returns:
        包装后的函数
    """

    def wrapper(cfg: DictConfig) -> Any:
        try:
            return task_func(cfg)
        except Exception as ex:
            from loguru import logger

            logger.exception(f"Task failed with exception: {ex}")
            raise

    return wrapper


def instantiate_callbacks(callbacks_cfg: DictConfig) -> list:
    """从配置实例化回调列表。

    Args:
        callbacks_cfg: 回调配置

    Returns:
        回调对象列表
    """
    from hydra.utils import instantiate
    from lightning.pytorch.callbacks import Callback

    callbacks: list[Callback] = []
    if not callbacks_cfg:
        return callbacks

    for _, cb_conf in callbacks_cfg.items():
        if cb_conf is not None and "_target_" in cb_conf:
            callbacks.append(instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> list:
    """从配置实例化日志记录器列表。

    Args:
        logger_cfg: 日志配置

    Returns:
        日志记录器对象列表
    """
    from hydra.utils import instantiate
    from lightning.pytorch.loggers import Logger

    loggers: list[Logger] = []
    if not logger_cfg:
        return loggers

    for _, lg_conf in logger_cfg.items():
        if lg_conf is not None and "_target_" in lg_conf:
            loggers.append(instantiate(lg_conf))

    return loggers


def log_hyperparameters(
    cfg: DictConfig,
    model: Any,
    trainer: Any,
) -> None:
    """将超参数记录到所有 logger。

    Args:
        cfg: Hydra 配置
        model: Lightning 模型
        trainer: Lightning Trainer
    """
    from omegaconf import OmegaConf

    hparams = {}

    # 选择要记录的配置项
    for key in ["model", "data", "trainer", "callbacks", "tags", "seed"]:
        if key in cfg:
            if isinstance(cfg[key], DictConfig):
                hparams[key] = OmegaConf.to_container(cfg[key], resolve=True)
            else:
                hparams[key] = cfg[key]

    # 记录到所有 logger
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)
