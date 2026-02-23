"""预测/推理入口脚本"""

from pathlib import Path
from typing import Any

import pyrootutils
from loguru import logger

# 自动定位项目根目录，加载 .env，并将根目录加入 pythonpath
root = pyrootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True, dotenv=True)

import hydra  # noqa: E402
import torch  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from omegaconf import DictConfig  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision import transforms  # noqa: E402

from src.utils import extras, task_wrapper  # noqa: E402


def get_device(device_cfg: str) -> torch.device:
    """根据配置获取设备。

    Args:
        device_cfg: 设备配置字符串

    Returns:
        torch.device 对象
    """
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def load_image(image_path: str, transform: transforms.Compose) -> torch.Tensor:
    """加载并预处理图片。

    Args:
        image_path: 图片路径
        transform: 图片预处理变换

    Returns:
        预处理后的图片张量
    """
    image = Image.open(image_path).convert("RGB")
    tensor: torch.Tensor = transform(image)
    return tensor


def get_image_paths(input_path: str) -> list[str]:
    """获取输入路径下的所有图片路径。

    Args:
        input_path: 输入路径（可以是文件或目录）

    Returns:
        图片路径列表
    """
    path = Path(input_path)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}

    if path.is_file():
        return [str(path)]
    elif path.is_dir():
        image_paths: list[Path] = []
        for ext in image_extensions:
            image_paths.extend(path.glob(f"*{ext}"))
            image_paths.extend(path.glob(f"*{ext.upper()}"))
        return [str(p) for p in sorted(image_paths)]
    else:
        raise ValueError(f"Input path does not exist: {path}")


# CIFAR-10 类别标签
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# MNIST 类别标签
MNIST_CLASSES = [str(i) for i in range(10)]


@task_wrapper
def predict(cfg: DictConfig) -> list[dict[str, Any]]:
    """预测流程。

    Args:
        cfg: Hydra 配置对象

    Returns:
        预测结果列表
    """
    assert cfg.ckpt_path, "Checkpoint path (ckpt_path) must be specified for prediction!"
    assert cfg.input_path, "Input path (input_path) must be specified for prediction!"

    # 1. 获取设备
    device = get_device(cfg.device)
    logger.info(f"Using device: {device}")

    # 2. 加载模型
    logger.info(f"Loading model from checkpoint: {cfg.ckpt_path}")
    model = instantiate(cfg.model)

    # 从 checkpoint 加载权重
    checkpoint = torch.load(cfg.ckpt_path, map_location=device, weights_only=False)
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    if hasattr(model, "freeze"):
        model.freeze()

    # 3. 定义预处理（根据模型类型选择）
    # 判断是 CIFAR 还是 MNIST 模型
    model_name = cfg.model.get("model_name", "")
    if model_name in ["GoogleNet", "ResNet", "DenseNet"]:
        # CIFAR-10 预处理
        data_means = [0.49139968, 0.48215841, 0.44653091]
        data_stds = [0.24703223, 0.24348513, 0.26158784]
        transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(data_means, data_stds),
            ]
        )
        class_names = CIFAR10_CLASSES
    else:
        # MNIST 预处理
        transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        class_names = MNIST_CLASSES

    # 4. 获取图片路径
    image_paths = get_image_paths(cfg.input_path)
    logger.info(f"Found {len(image_paths)} images to predict")

    # 5. 创建输出目录
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 6. 批量预测
    results: list[dict[str, Any]] = []
    batch_size = cfg.batch_size

    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            batch_images = []
            loaded_paths: list[str] = []

            for img_path in batch_paths:
                try:
                    img = load_image(img_path, transform)
                    batch_images.append(img)
                    loaded_paths.append(img_path)
                except Exception as e:
                    logger.warning(f"Failed to load image {img_path}: {e}")
                    continue

            if not batch_images:
                continue

            # 堆叠成批次
            batch_tensor = torch.stack(batch_images).to(device)

            # 前向传播
            logits = model(batch_tensor)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            # 收集结果
            for j, img_path in enumerate(loaded_paths):
                pred_idx = int(preds[j].item())
                pred_prob = float(probs[j, pred_idx].item())
                result: dict[str, Any] = {
                    "image": img_path,
                    "prediction": pred_idx,
                    "class_name": class_names[pred_idx],
                    "confidence": pred_prob,
                    "probabilities": {class_names[k]: float(probs[j, k].item()) for k in range(len(class_names))},
                }
                results.append(result)
                logger.info(f"{Path(img_path).name}: {class_names[pred_idx]} (confidence: {pred_prob:.4f})")

    # 7. 保存结果
    import json

    results_file = output_dir / "predictions.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Predictions saved to: {results_file}")

    return results


@hydra.main(version_base="1.3", config_path="../configs", config_name="predict.yaml")
def main(cfg: DictConfig) -> list[dict[str, Any]]:
    """主入口函数。

    Args:
        cfg: Hydra 配置对象

    Returns:
        预测结果列表
    """
    # 应用额外配置
    extras(cfg)

    # 执行预测
    results = predict(cfg)

    return results


if __name__ == "__main__":
    main()
