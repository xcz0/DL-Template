from lightning import LightningDataModule
from loguru import logger

DATA_MEANS = [0.49139968, 0.48215841, 0.44653091]
DATA_STDS = [0.24703223, 0.24348513, 0.26158784]


class CIFARDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        val_ratio: float = 0.1,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_ratio = val_ratio
        self.pin_memory = pin_memory

    def prepare_data(self):
        # Download CIFAR dataset if not already present
        from torchvision.datasets import CIFAR10

        CIFAR10(root=self.data_dir, train=True, download=True)
        CIFAR10(root=self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        from torch.utils.data import random_split
        from torchvision import transforms
        from torchvision.datasets import CIFAR10

        test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(DATA_MEANS, DATA_STDS)])
        train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Normalize(DATA_MEANS, DATA_STDS),
            ]
        )

        if stage == "fit" or stage is None:
            train_dataset = CIFAR10(root=self.data_dir, train=True, transform=train_transform, download=True)
            val_dataset = CIFAR10(root=self.data_dir, train=True, transform=test_transform, download=True)
            val_size = int(self.val_ratio * len(train_dataset))
            train_size = len(train_dataset) - val_size
            self.train_set, _ = random_split(train_dataset, [train_size, val_size])
            _, self.val_set = random_split(val_dataset, [train_size, val_size])

        if stage == "test" or stage is None:
            self.test_set = CIFAR10(root=self.data_dir, train=False, transform=test_transform)

    def train_dataloader(self):
        from torch.utils.data import DataLoader

        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        from torch.utils.data import DataLoader

        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        from torch.utils.data import DataLoader

        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )


if __name__ == "__main__":
    # Example usage
    data_module = CIFARDataModule(val_ratio=0.1, data_dir="./data", batch_size=64, num_workers=8)
    data_module.prepare_data()
    data_module.setup(stage="fit")
    train_loader = data_module.train_dataloader()
    images, labels = next(iter(train_loader))
    logger.info(
        f"images.shape: {images.shape}, labels.shape: {labels.shape}, Batch mean:{images.mean(dim=[0, 2, 3])}, Batch std: {images.std(dim=[0, 2, 3])}"
    )
    val_loader = data_module.val_dataloader()
    images, labels = next(iter(val_loader))
    logger.info(
        f"images.shape: {images.shape}, labels.shape: {labels.shape}, Batch mean:{images.mean(dim=[0, 2, 3])}, Batch std: {images.std(dim=[0, 2, 3])}"
    )
    data_module.setup(stage="test")
    test_loader = data_module.test_dataloader()
    images, labels = next(iter(test_loader))
    logger.info(
        f"images.shape: {images.shape}, labels.shape: {labels.shape}, Batch mean:{images.mean(dim=[0, 2, 3])}, Batch std: {images.std(dim=[0, 2, 3])}"
    )
