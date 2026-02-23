import torch
import torch.nn as nn
import torch.optim as optim
from jaxtyping import Float, Int
from lightning import LightningModule
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from .components.DenseNet import DenseNet
from .components.GoogleNet import GoogleNet
from .components.ResNet import ResNet

MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "GoogleNet": GoogleNet,
    "ResNet": ResNet,
    "DenseNet": DenseNet,
}


def create_model(model_name: str, model_hparams: dict) -> nn.Module:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f'Unknown model name "{model_name}". Available models are: {list(MODEL_REGISTRY.keys())}')
    return MODEL_REGISTRY[model_name](**model_hparams)


class CIFARModule(LightningModule):
    def __init__(
        self,
        model_name: str,
        model_hparams: dict | DictConfig,
        optimizer_name: str,
        optimizer_hparams: dict | DictConfig,
    ):
        super().__init__()
        # Hydra 传入的 DictConfig 转换为普通 dict 以确保 checkpoint 兼容性
        model_hparams = (
            OmegaConf.to_container(model_hparams, resolve=True)
            if isinstance(model_hparams, DictConfig)
            else dict(model_hparams)
        )
        optimizer_hparams = (
            OmegaConf.to_container(optimizer_hparams, resolve=True)
            if isinstance(optimizer_hparams, DictConfig)
            else dict(optimizer_hparams)
        )

        self.save_hyperparameters(
            {
                "model_name": model_name,
                "model_hparams": model_hparams,
                "optimizer_name": optimizer_name,
                "optimizer_hparams": optimizer_hparams,
            }
        )

        self.model = create_model(model_name, model_hparams)
        self.loss_module = nn.CrossEntropyLoss()
        self.example_input_array = torch.zeros((1, 3, 32, 32), dtype=torch.float32)

    def forward(self, imgs: Float[Tensor, "B 3 H W"]) -> Float[Tensor, "B num_classes"]:
        return self.model(imgs)

    def configure_optimizers(self) -> tuple[list[optim.Optimizer], list[optim.lr_scheduler.LRScheduler]]:
        optimizer_name = self.hparams["optimizer_name"]
        optimizer_hparams = self.hparams["optimizer_hparams"]
        optimizer_registry: dict[str, type[optim.Optimizer]] = {
            "Adam": optim.AdamW,
            "SGD": optim.SGD,
        }

        if optimizer_name not in optimizer_registry:
            raise ValueError(f'Unknown optimizer: "{optimizer_name}". Supported: {list(optimizer_registry)}')

        optimizer = optimizer_registry[optimizer_name](self.parameters(), **optimizer_hparams)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        return [optimizer], [scheduler]

    def training_step(
        self,
        batch: tuple[Float[Tensor, "B 3 H W"], Int[Tensor, "B"]],
        batch_idx: int,
    ) -> Float[Tensor, ""]:
        imgs, labels = batch
        preds = self.model(imgs)
        loss = self.loss_module(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(
        self,
        batch: tuple[Float[Tensor, "B 3 H W"], Int[Tensor, "B"]],
        batch_idx: int,
    ) -> None:
        imgs, labels = batch
        preds = self.model(imgs).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        self.log("val/acc", acc, prog_bar=True)

    def test_step(
        self,
        batch: tuple[Float[Tensor, "B 3 H W"], Int[Tensor, "B"]],
        batch_idx: int,
    ) -> None:
        imgs, labels = batch
        preds = self.model(imgs).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        self.log("test/acc", acc)
