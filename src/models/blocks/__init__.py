# 可复用层
from typing import Callable

import torch.nn as nn

act_fn_by_name: dict[str, Callable[[], nn.Module]] = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "mish": nn.Mish,
}
