

import torch
from torch import nn, Tensor

from typing import Any, Callable, List, Optional, Tuple, Union

class Conv2dNormActivation(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]] = 3,
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Optional[Union[int, Tuple[int, ...], str]] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
        dilation: Union[int, Tuple[int, ...]] = 1,
        bias: Optional[bool] = None,
        inplace: Optional[bool] = True,
    ) -> None:
        if padding is None:
            if dilation > 1:
                padding = dilation * (kernel_size - 1) // 2
            else:
                padding = kernel_size // 2

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6

        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                bias,
            ),
            norm_layer(out_channels),
            activation_layer(inplace=inplace),
        ]

        super().__init__(*layers)


class InvertedResidual(nn.Module):
    def __init__(self, inp: int, oup: int, stride: int, expand_ratio: int) -> None:
        super().__init__()

        hidden_dim = int(round(inp * expand_ratio))
        self.use_residual = inp == oup

        layers: List[nn.Module] = []

        if expand_ratio != 1:
            # pointwise expansion
            layers.append(
                Conv2dNormActivation(inp, hidden_dim, kernel_size=1, bias=False)
            )

        def conv_dw(inp, oup, stride):
            # inp, oup = int(inp * self.alpha), int(oup * self.alpha)
            return nn.Sequential(
                # depth wise
                Conv2dNormActivation(inp, inp, 3, 1, stride, groups=inp, bias=False),
                # pointwise
                nn.Conv2d(inp, oup, 1, 1, padding=0, bias=False),
                nn.BatchNorm2d(oup),
            )

        # depth-wise convolution:
        layers.append(conv_dw(hidden_dim, oup, stride))
        self.conv = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        if self.use_residual:
            out = x + self.conv(x)
        else:
            out = self.conv(x)
        print(out.shape)
        return out