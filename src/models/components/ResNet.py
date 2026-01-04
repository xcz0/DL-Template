from dataclasses import dataclass
from typing import Callable

from jaxtyping import Float
from torch import Tensor, nn

from ..blocks import act_fn_by_name


@dataclass(frozen=True)
class ResNetConfig:
    """ResNet 超参数配置"""

    num_classes: int
    c_hidden: tuple[int, ...]
    num_blocks: tuple[int, ...]
    act_fn_name: str
    act_fn: Callable[[], nn.Module]
    block_class: type["ResNetBlock"] | type["PreActResNetBlock"]


class ResNetBlock(nn.Module):
    def __init__(self, c_in: int, act_fn: Callable[[], nn.Module], subsample: bool = False, c_out: int = -1):
        super().__init__()
        if not subsample:
            c_out = c_in

        self.net = nn.Sequential(
            nn.Conv2d(
                c_in,
                c_out,
                kernel_size=3,
                padding=1,
                stride=1 if not subsample else 2,
                bias=False,
            ),
            nn.BatchNorm2d(c_out),
            act_fn(),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
        )

        self.downsample = nn.Conv2d(c_in, c_out, kernel_size=1, stride=2) if subsample else None
        self.act_fn = act_fn()

    def forward(self, x: Float[Tensor, "B C H W"]) -> Float[Tensor, "B C_out H_out W_out"]:
        z = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)
        out = z + x
        out = self.act_fn(out)
        return out


class PreActResNetBlock(nn.Module):
    def __init__(self, c_in: int, act_fn: Callable[[], nn.Module], subsample: bool = False, c_out: int = -1):
        super().__init__()
        if not subsample:
            c_out = c_in

        self.net = nn.Sequential(
            nn.BatchNorm2d(c_in),
            act_fn(),
            nn.Conv2d(
                c_in,
                c_out,
                kernel_size=3,
                padding=1,
                stride=1 if not subsample else 2,
                bias=False,
            ),
            nn.BatchNorm2d(c_out),
            act_fn(),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
        )

        self.downsample = (
            nn.Sequential(
                nn.BatchNorm2d(c_in),
                act_fn(),
                nn.Conv2d(c_in, c_out, kernel_size=1, stride=2, bias=False),
            )
            if subsample
            else None
        )

    def forward(self, x: Float[Tensor, "B C H W"]) -> Float[Tensor, "B C_out H_out W_out"]:
        z = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)
        out = z + x
        return out


resnet_blocks_by_name = {
    "ResNetBlock": ResNetBlock,
    "PreActResNetBlock": PreActResNetBlock,
}


class ResNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        num_blocks: list[int] | tuple[int, ...] = (3, 3, 3),
        c_hidden: list[int] | tuple[int, ...] = (16, 32, 64),
        act_fn_name: str = "relu",
        block_name: str = "ResNetBlock",
        **kwargs,
    ):
        super().__init__()
        if block_name not in resnet_blocks_by_name:
            raise ValueError(f"Unknown ResNet block: {block_name}. Available: {list(resnet_blocks_by_name.keys())}")

        self.config = ResNetConfig(
            num_classes=num_classes,
            c_hidden=tuple(c_hidden),
            num_blocks=tuple(num_blocks),
            act_fn_name=act_fn_name,
            act_fn=act_fn_by_name[act_fn_name],
            block_class=resnet_blocks_by_name[block_name],
        )

        self._create_network()
        self._init_params()

    def _create_network(self) -> None:
        c_hidden = self.config.c_hidden

        if self.config.block_class == PreActResNetBlock:
            self.input_net = nn.Sequential(
                nn.Conv2d(3, c_hidden[0], kernel_size=3, padding=1, bias=False),
            )
        else:
            self.input_net = nn.Sequential(
                nn.Conv2d(3, c_hidden[0], kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c_hidden[0]),
                self.config.act_fn(),
            )

        blocks: list[nn.Module] = []
        for block_idx, block_count in enumerate(self.config.num_blocks):
            for block_in_group_idx in range(block_count):
                subsample = block_in_group_idx == 0 and block_idx > 0
                blocks.append(
                    self.config.block_class(
                        c_in=c_hidden[block_idx if not subsample else (block_idx - 1)],
                        act_fn=self.config.act_fn,
                        subsample=subsample,
                        c_out=c_hidden[block_idx],
                    )
                )
        self.blocks = nn.Sequential(*blocks)

        self.output_net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(c_hidden[-1], self.config.num_classes),
        )

    def _init_params(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity=self.config.act_fn_name)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Float[Tensor, "B 3 H W"]) -> Float[Tensor, "B num_classes"]:
        x = self.input_net(x)
        x = self.blocks(x)
        x = self.output_net(x)
        return x
