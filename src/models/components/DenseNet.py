from types import SimpleNamespace

import torch
from torch import nn

from ..blocks import act_fn_by_name


class DenseLayer(nn.Module):
    def __init__(self, c_in: int, bn_size: int, growth_rate: int, act_fn):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(c_in),
            act_fn(),
            nn.Conv2d(c_in, bn_size * growth_rate, kernel_size=1, bias=False),
            nn.BatchNorm2d(bn_size * growth_rate),
            act_fn(),
            nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x):
        out = self.net(x)
        out = torch.cat([out, x], dim=1)
        return out


class DenseBlock(nn.Module):
    def __init__(self, c_in: int, num_layers: int, bn_size: int, growth_rate: int, act_fn):
        super().__init__()
        layers: list[nn.Module] = []
        for layer_idx in range(num_layers):
            layers.append(
                DenseLayer(
                    c_in=c_in + layer_idx * growth_rate,
                    bn_size=bn_size,
                    growth_rate=growth_rate,
                    act_fn=act_fn,
                )
            )
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class TransitionLayer(nn.Module):
    def __init__(self, c_in: int, c_out: int, act_fn):
        super().__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(c_in),
            act_fn(),
            nn.Conv2d(c_in, c_out, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.transition(x)


class DenseNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        num_layers: list[int] = [6, 6, 6, 6],
        bn_size: int = 2,
        growth_rate: int = 16,
        act_fn_name: str = "relu",
        **kwargs,
    ):
        super().__init__()
        self.hparams = SimpleNamespace(
            num_classes=num_classes,
            num_layers=num_layers,
            bn_size=bn_size,
            growth_rate=growth_rate,
            act_fn_name=act_fn_name,
            act_fn=act_fn_by_name[act_fn_name],
        )

        self._create_network()
        self._init_params()

    def _create_network(self):
        c_hidden = self.hparams.growth_rate * self.hparams.bn_size

        self.input_net = nn.Sequential(
            nn.Conv2d(3, c_hidden, kernel_size=3, padding=1),
        )

        blocks: list[nn.Module] = []
        for block_idx, num_layers_in_block in enumerate(self.hparams.num_layers):
            blocks.append(
                DenseBlock(
                    c_in=c_hidden,
                    num_layers=num_layers_in_block,
                    bn_size=self.hparams.bn_size,
                    growth_rate=self.hparams.growth_rate,
                    act_fn=self.hparams.act_fn,
                )
            )
            c_hidden = c_hidden + num_layers_in_block * self.hparams.growth_rate
            if block_idx < len(self.hparams.num_layers) - 1:
                blocks.append(
                    TransitionLayer(
                        c_in=c_hidden,
                        c_out=c_hidden // 2,
                        act_fn=self.hparams.act_fn,
                    )
                )
                c_hidden = c_hidden // 2

        self.blocks = nn.Sequential(*blocks)

        self.output_net = nn.Sequential(
            nn.BatchNorm2d(c_hidden),
            self.hparams.act_fn(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(c_hidden, self.hparams.num_classes),
        )

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity=self.hparams.act_fn_name)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input_net(x)
        x = self.blocks(x)
        x = self.output_net(x)
        return x
