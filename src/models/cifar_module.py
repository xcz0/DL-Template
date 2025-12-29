import torch
import torch.nn as nn
import torch.optim as optim
from lightning import LightningModule
from omegaconf import OmegaConf

from .components.DenseNet import DenseNet
from .components.GoogleNet import GoogleNet
from .components.ResNet import ResNet

model_dict = {"GoogleNet": GoogleNet, "ResNet": ResNet, "DenseNet": DenseNet}


def create_model(model_name, model_hparams):
    if model_name in model_dict:
        return model_dict[model_name](**model_hparams)
    else:
        raise AssertionError(f'Unknown model name "{model_name}". Available models are: {str(model_dict.keys())}')


class CIFARModule(LightningModule):
    def __init__(self, model_name, model_hparams, optimizer_name, optimizer_hparams):
        """
        Inputs:
            model_name - Name of the model/CNN to run. Used for creating the model (see function below)
            model_hparams - Hyperparameters for the model, as dictionary.
            optimizer_name - Name of the optimizer to use. Currently supported: Adam, SGD
            optimizer_hparams - Hyperparameters for the optimizer, as dictionary. This includes learning rate, weight decay, etc.
        """
        super().__init__()
        # Hydra may pass DictConfig/ListConfig; store plain containers to keep checkpoints safe to load.
        model_hparams = OmegaConf.to_container(model_hparams, resolve=True) if model_hparams is not None else {}
        optimizer_hparams = (
            OmegaConf.to_container(optimizer_hparams, resolve=True) if optimizer_hparams is not None else {}
        )

        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters(
            {
                "model_name": model_name,
                "model_hparams": model_hparams,
                "optimizer_name": optimizer_name,
                "optimizer_hparams": optimizer_hparams,
            }
        )
        # Create model
        self.model = create_model(model_name, model_hparams)
        # Create loss module
        self.loss_module = nn.CrossEntropyLoss()
        # Example input for visualizing the graph in Tensorboard
        self.example_input_array = torch.zeros((1, 3, 32, 32), dtype=torch.float32)

    def forward(self, imgs):
        # Forward function that is run when visualizing the graph
        return self.model(imgs)

    def configure_optimizers(self):
        # We will support Adam or SGD as optimizers.
        if self.hparams.optimizer_name == "Adam":
            # AdamW is Adam with a correct implementation of weight decay (see here for details: https://arxiv.org/pdf/1711.05101.pdf)
            optimizer = optim.AdamW(self.parameters(), **self.hparams.optimizer_hparams)
        elif self.hparams.optimizer_name == "SGD":
            optimizer = optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)
        else:
            raise AssertionError(f'Unknown optimizer: "{self.hparams.optimizer_name}"')

        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        imgs, labels = batch
        preds = self.model(imgs)
        loss = self.loss_module(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        return loss  # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        # By default logs it per epoch (weighted average over batches)
        self.log("val/acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        # By default logs it per epoch (weighted average over batches), and returns it afterwards
        self.log("test/acc", acc)
