from typing import Callable

import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from lightning import LightningModule
from torch import Tensor
from torch.optim import Optimizer
from torchmetrics import Accuracy


class MNISTLitModule(LightningModule):
    def __init__(self, net: nn.Module, optimizer: Callable[..., Optimizer]):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.net = net
        self.accuracy = Accuracy(task="multiclass", num_classes=10)

    def forward(self, x: Float[Tensor, "B 1 H W"]) -> Float[Tensor, "B num_classes"]:
        return self.net(x)

    def training_step(
        self,
        batch: tuple[Float[Tensor, "B 1 H W"], Int[Tensor, "B"]],
        batch_idx: int,
    ) -> Float[Tensor, ""]:
        loss, preds, targets = self._model_step(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.accuracy(preds, targets), on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(
        self,
        batch: tuple[Float[Tensor, "B 1 H W"], Int[Tensor, "B"]],
        batch_idx: int,
    ) -> None:
        loss, preds, targets = self._model_step(batch)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", self.accuracy(preds, targets), prog_bar=True)

    def test_step(
        self,
        batch: tuple[Float[Tensor, "B 1 H W"], Int[Tensor, "B"]],
        batch_idx: int,
    ) -> None:
        loss, preds, targets = self._model_step(batch)
        self.log("test/loss", loss, prog_bar=True)
        self.log("test/acc", self.accuracy(preds, targets), prog_bar=True)

    def _model_step(
        self,
        batch: tuple[Float[Tensor, "B 1 H W"], Int[Tensor, "B"]],
    ) -> tuple[Float[Tensor, ""], Int[Tensor, "B"], Int[Tensor, "B"]]:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=1)
        return loss, preds, y

    def configure_optimizers(self) -> Optimizer:
        return self.hparams["optimizer"](params=self.parameters())
