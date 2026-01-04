"""简单 CNN 网络组件"""

from jaxtyping import Float
from torch import Tensor, nn


class SimpleCNN(nn.Module):
    """简单的 CNN 网络，用于 MNIST 分类。"""

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 1,
    ):
        super().__init__()

        self.conv_layers = nn.Sequential(
            # 第一层卷积: 1 -> 32 channels, 28x28 -> 26x26 -> 13x13
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 第二层卷积: 32 -> 64 channels, 13x13 -> 11x11 -> 5x5
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 第三层卷积: 64 -> 128 channels, 5x5 -> 3x3
            nn.Conv2d(64, 128, kernel_size=3, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: Float[Tensor, "B C H W"]) -> Float[Tensor, "B num_classes"]:
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


if __name__ == "__main__":
    import torch

    model = SimpleCNN(num_classes=10)
    x = torch.randn(2, 1, 28, 28)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
