from dataclasses import dataclass
from typing import Callable

from jaxtyping import Float
from torch import Tensor, nn

from ..blocks import act_fn_by_name
from ..blocks.Inception import InceptionBlock


@dataclass(frozen=True)
class GoogleNetConfig:
    """GoogleNet 超参数配置"""

    num_classes: int
    act_fn_name: str
    act_fn: Callable[[], nn.Module]


class GoogleNet(nn.Module):
    def __init__(self, num_classes: int = 10, act_fn_name: str = "relu", **kwargs):
        super().__init__()
        self.config = GoogleNetConfig(
            num_classes=num_classes, act_fn_name=act_fn_name, act_fn=act_fn_by_name[act_fn_name]
        )
        self._create_network()
        self._init_params()

    def _create_network(self) -> None:
        # A first convolution on the original image to scale up the channel size
        self.input_net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), self.config.act_fn()
        )
        # Stacking inception blocks
        self.inception_blocks = nn.Sequential(
            InceptionBlock(
                64,
                c_red={"3x3": 32, "5x5": 16},
                c_out={"1x1": 16, "3x3": 32, "5x5": 8, "max": 8},
                act_fn=self.config.act_fn,
            ),
            InceptionBlock(
                64,
                c_red={"3x3": 32, "5x5": 16},
                c_out={"1x1": 24, "3x3": 48, "5x5": 12, "max": 12},
                act_fn=self.config.act_fn,
            ),
            nn.MaxPool2d(3, stride=2, padding=1),  # 32x32 => 16x16
            InceptionBlock(
                96,
                c_red={"3x3": 32, "5x5": 16},
                c_out={"1x1": 24, "3x3": 48, "5x5": 12, "max": 12},
                act_fn=self.config.act_fn,
            ),
            InceptionBlock(
                96,
                c_red={"3x3": 32, "5x5": 16},
                c_out={"1x1": 16, "3x3": 48, "5x5": 16, "max": 16},
                act_fn=self.config.act_fn,
            ),
            InceptionBlock(
                96,
                c_red={"3x3": 32, "5x5": 16},
                c_out={"1x1": 16, "3x3": 48, "5x5": 16, "max": 16},
                act_fn=self.config.act_fn,
            ),
            InceptionBlock(
                96,
                c_red={"3x3": 32, "5x5": 16},
                c_out={"1x1": 32, "3x3": 48, "5x5": 24, "max": 24},
                act_fn=self.config.act_fn,
            ),
            nn.MaxPool2d(3, stride=2, padding=1),  # 16x16 => 8x8
            InceptionBlock(
                128,
                c_red={"3x3": 48, "5x5": 16},
                c_out={"1x1": 32, "3x3": 64, "5x5": 16, "max": 16},
                act_fn=self.config.act_fn,
            ),
            InceptionBlock(
                128,
                c_red={"3x3": 48, "5x5": 16},
                c_out={"1x1": 32, "3x3": 64, "5x5": 16, "max": 16},
                act_fn=self.config.act_fn,
            ),
        )
        # Mapping to classification output
        self.output_net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(128, self.config.num_classes)
        )

    def _init_params(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity=self.config.act_fn_name)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Float[Tensor, "B 3 H W"]) -> Float[Tensor, "B num_classes"]:
        x = self.input_net(x)
        x = self.inception_blocks(x)
        x = self.output_net(x)
        return x
