import torch
from lightning import LightningModule
from torchmetrics import Accuracy


class MNISTLitModule(LightningModule):
    def __init__(self, net, optimizer):
        super().__init__()
        self.save_hyperparameters(logger=False)  # 保存 net 和 optimizer 参数到 ckpt
        self.net = net
        self.accuracy = Accuracy(task="multiclass", num_classes=10)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        loss, preds, targets = self.model_step(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.accuracy(preds, targets), on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self.model_step(batch)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", self.accuracy(preds, targets), prog_bar=True)

    def model_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def configure_optimizers(self):
        # 这里的 self.hparams.optimizer 是 hydra Partial 实例化的对象
        return self.hparams.optimizer(params=self.parameters())
