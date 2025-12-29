"""MNIST 数据模块"""

from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST


class MNISTDataModule(LightningDataModule):
    """MNIST 数据模块。

    Args:
        data_dir: 数据存储目录
        batch_size: 批次大小
        num_workers: 数据加载进程数
        pin_memory: 是否将数据固定在内存中
        val_ratio: 验证集比例
    """

    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        val_ratio: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.val_ratio = val_ratio

        # 数据变换
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        self.train_set = None
        self.val_set = None
        self.test_set = None

    def prepare_data(self):
        """下载数据集（仅在单进程中调用）"""
        MNIST(root=self.data_dir, train=True, download=True)
        MNIST(root=self.data_dir, train=False, download=True)

    def setup(self, stage: str = None):
        """设置数据集（每个进程都会调用）

        Args:
            stage: 训练阶段 (fit, test, predict)
        """
        if stage == "fit" or stage is None:
            mnist_full = MNIST(
                root=self.data_dir,
                train=True,
                transform=self.transform,
            )
            # 划分训练集和验证集
            val_size = int(len(mnist_full) * self.val_ratio)
            train_size = len(mnist_full) - val_size
            self.train_set, self.val_set = random_split(mnist_full, [train_size, val_size])

        if stage == "test" or stage is None:
            self.test_set = MNIST(
                root=self.data_dir,
                train=False,
                transform=self.transform,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


if __name__ == "__main__":
    # 测试数据模块
    from loguru import logger

    dm = MNISTDataModule(data_dir="./data", batch_size=64, num_workers=0)
    dm.prepare_data()
    dm.setup(stage="fit")

    train_loader = dm.train_dataloader()
    images, labels = next(iter(train_loader))
    logger.info(f"Train batch - images: {images.shape}, labels: {labels.shape}")

    val_loader = dm.val_dataloader()
    images, labels = next(iter(val_loader))
    logger.info(f"Val batch - images: {images.shape}, labels: {labels.shape}")

    dm.setup(stage="test")
    test_loader = dm.test_dataloader()
    images, labels = next(iter(test_loader))
    logger.info(f"Test batch - images: {images.shape}, labels: {labels.shape}")
