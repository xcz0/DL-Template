"""预测/推理入口脚本"""

import json
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

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
CIFAR_MODEL_NAMES = {"GoogleNet", "ResNet", "DenseNet"}

# CIFAR-10 类别标签
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# MNIST 类别标签
MNIST_CLASSES = [str(i) for i in range(10)]


def get_device(device_cfg: str) -> torch.device:
    """根据配置获取设备。"""
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def load_image(image_path: str, transform: transforms.Compose) -> torch.Tensor:
    """加载并预处理图片。"""
    with Image.open(image_path) as image:
        return transform(image.convert("RGB"))


def get_image_paths(input_path: str) -> list[str]:
    """获取输入路径下的所有图片路径。"""
    path = Path(input_path)

    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image file extension: {path.suffix}")
        return [str(path)]

    if path.is_dir():
        return [str(p) for p in sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)]

    raise ValueError(f"Input path does not exist: {path}")


def _get_predict_meta(model_name: str) -> tuple[transforms.Compose, list[str]]:
    if model_name in CIFAR_MODEL_NAMES:
        transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784)),
            ]
        )
        return transform, CIFAR10_CLASSES

    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    return transform, MNIST_CLASSES


@task_wrapper
def predict(cfg: DictConfig) -> list[dict[str, Any]]:
    """预测流程。"""
    assert cfg.ckpt_path, "Checkpoint path (ckpt_path) must be specified for prediction!"
    assert cfg.input_path, "Input path (input_path) must be specified for prediction!"

    device = get_device(cfg.device)
    logger.info(f"Using device: {device}")

    logger.info(f"Loading model from checkpoint: {cfg.ckpt_path}")
    model = instantiate(cfg.model)

    checkpoint = torch.load(cfg.ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device).eval()

    transform, class_names = _get_predict_meta(cfg.model.get("model_name", ""))

    image_paths = get_image_paths(cfg.input_path)
    logger.info(f"Found {len(image_paths)} images to predict")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []

    with torch.inference_mode():
        for i in range(0, len(image_paths), cfg.batch_size):
            batch_paths = image_paths[i : i + cfg.batch_size]
            valid_paths: list[str] = []
            batch_images: list[torch.Tensor] = []

            for img_path in batch_paths:
                try:
                    batch_images.append(load_image(img_path, transform))
                    valid_paths.append(img_path)
                except Exception as e:
                    logger.warning(f"Failed to load image {img_path}: {e}")

            if not batch_images:
                continue

            logits = model(torch.stack(batch_images).to(device))
            probs = torch.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)

            for j, img_path in enumerate(valid_paths):
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

    results_file = output_dir / "predictions.json"
    with results_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Predictions saved to: {results_file}")

    return results


@hydra.main(version_base="1.3", config_path="../configs", config_name="predict.yaml")
def main(cfg: DictConfig) -> list[dict[str, Any]]:
    """主入口函数。"""
    extras(cfg)
    return predict(cfg)


if __name__ == "__main__":
    main()
