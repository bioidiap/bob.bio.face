"""
@author: Jun Wang
@date: 20210322
@contact: jun21wangustc@gmail.com
"""

# based on:
# https://github.com/clovaai/rexnet/blob/master/rexnetv1.py
"""
ReXNet
Copyright (c) 2020-present NAVER Corp.
MIT license
"""

from math import ceil

import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


# Memory-efficient Siwsh using torch.jit.script borrowed from the code in (https://twitter.com/jeremyphoward/status/1188251041835315200)
# Currently use memory-efficient Swish as default:
USE_MEMORY_EFFICIENT_SWISH = True

if USE_MEMORY_EFFICIENT_SWISH:

    @torch.jit.script
    def swish_fwd(x):
        return x.mul(torch.sigmoid(x))

    @torch.jit.script
    def swish_bwd(x, grad_output):
        x_sigmoid = torch.sigmoid(x)
        return grad_output * (x_sigmoid * (1.0 + x * (1.0 - x_sigmoid)))

    class SwishJitImplementation(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return swish_fwd(x)

        @staticmethod
        def backward(ctx, grad_output):
            x = ctx.saved_tensors[0]
            return swish_bwd(x, grad_output)

    def swish(x, inplace=False):
        return SwishJitImplementation.apply(x)

else:

    def swish(x, inplace=False):
        return x.mul_(x.sigmoid()) if inplace else x.mul(x.sigmoid())


class Swish(nn.Module):
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return swish(x, self.inplace)


def ConvBNAct(
    out,
    in_channels,
    channels,
    kernel=1,
    stride=1,
    pad=0,
    num_group=1,
    active=True,
    relu6=False,
):
    out.append(
        nn.Conv2d(
            in_channels,
            channels,
            kernel,
            stride,
            pad,
            groups=num_group,
            bias=False,
        )
    )
    out.append(nn.BatchNorm2d(channels))
    if active:
        out.append(nn.ReLU6(inplace=True) if relu6 else nn.ReLU(inplace=True))


def ConvBNSwish(
    out, in_channels, channels, kernel=1, stride=1, pad=0, num_group=1
):
    out.append(
        nn.Conv2d(
            in_channels,
            channels,
            kernel,
            stride,
            pad,
            groups=num_group,
            bias=False,
        )
    )
    out.append(nn.BatchNorm2d(channels))
    out.append(Swish())


class SE(nn.Module):
    def __init__(self, in_channels, channels, se_ratio=12):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(
                in_channels, channels // se_ratio, kernel_size=1, padding=0
            ),
            nn.BatchNorm2d(channels // se_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // se_ratio, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y


class LinearBottleneck(nn.Module):
    def __init__(
        self,
        in_channels,
        channels,
        t,
        stride,
        use_se=True,
        se_ratio=12,
        **kwargs,
    ):
        super(LinearBottleneck, self).__init__(**kwargs)
        self.use_shortcut = stride == 1 and in_channels <= channels
        self.in_channels = in_channels
        self.out_channels = channels

        out = []
        if t != 1:
            dw_channels = in_channels * t
            ConvBNSwish(out, in_channels=in_channels, channels=dw_channels)
        else:
            dw_channels = in_channels

        ConvBNAct(
            out,
            in_channels=dw_channels,
            channels=dw_channels,
            kernel=3,
            stride=stride,
            pad=1,
            num_group=dw_channels,
            active=False,
        )

        if use_se:
            out.append(SE(dw_channels, dw_channels, se_ratio))

        out.append(nn.ReLU6())
        ConvBNAct(
            out,
            in_channels=dw_channels,
            channels=channels,
            active=False,
            relu6=True,
        )
        self.out = nn.Sequential(*out)

    def forward(self, x):
        out = self.out(x)
        if self.use_shortcut:
            out[:, 0 : self.in_channels] += x

        return out


class ReXNetV1(nn.Module):
    def __init__(
        self,
        input_ch=16,
        final_ch=180,
        width_mult=1.0,
        depth_mult=1.0,
        use_se=True,
        se_ratio=12,
        out_h=7,
        out_w=7,
        feat_dim=512,
        dropout_ratio=0.2,
        bn_momentum=0.9,
    ):
        super(ReXNetV1, self).__init__()

        layers = [1, 2, 2, 3, 3, 5]
        strides = [1, 2, 2, 2, 1, 2]
        use_ses = [False, False, True, True, True, True]

        layers = [ceil(element * depth_mult) for element in layers]
        strides = sum(
            [
                [element] + [1] * (layers[idx] - 1)
                for idx, element in enumerate(strides)
            ],
            [],
        )
        if use_se:
            use_ses = sum(
                [
                    [element] * layers[idx]
                    for idx, element in enumerate(use_ses)
                ],
                [],
            )
        else:
            use_ses = [False] * sum(layers[:])
        ts = [1] * layers[0] + [6] * sum(layers[1:])

        self.depth = sum(layers[:]) * 3
        stem_channel = 32 / width_mult if width_mult < 1.0 else 32
        inplanes = input_ch / width_mult if width_mult < 1.0 else input_ch

        features = []
        in_channels_group = []
        channels_group = []

        # The following channel configuration is a simple instance to make each layer become an expand layer.
        for i in range(self.depth // 3):
            if i == 0:
                in_channels_group.append(int(round(stem_channel * width_mult)))
                channels_group.append(int(round(inplanes * width_mult)))
            else:
                in_channels_group.append(int(round(inplanes * width_mult)))
                inplanes += final_ch / (self.depth // 3 * 1.0)
                channels_group.append(int(round(inplanes * width_mult)))

        # ConvBNSwish(features, 3, int(round(stem_channel * width_mult)), kernel=3, stride=2, pad=1)
        ConvBNSwish(
            features,
            3,
            int(round(stem_channel * width_mult)),
            kernel=3,
            stride=1,
            pad=1,
        )

        for block_idx, (in_c, c, t, s, se) in enumerate(
            zip(in_channels_group, channels_group, ts, strides, use_ses)
        ):
            features.append(
                LinearBottleneck(
                    in_channels=in_c,
                    channels=c,
                    t=t,
                    stride=s,
                    use_se=se,
                    se_ratio=se_ratio,
                )
            )

        # pen_channels = int(1280 * width_mult)
        pen_channels = int(512 * width_mult)
        ConvBNSwish(features, c, pen_channels)

        # features.append(nn.AdaptiveAvgPool2d(1))
        self.features = nn.Sequential(*features)
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(dropout_ratio),
            Flatten(),
            nn.Linear(512 * out_h * out_w, feat_dim),
            nn.BatchNorm1d(feat_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.output_layer(x)
        return x
