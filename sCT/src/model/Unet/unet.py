"""来源于IDDPM代码，修改去除fp16，按照simple diffusion的形式修改"""

from abc import abstractmethod

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


class TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, x):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x):
        for layer in self:
            x = layer(x)
        return x


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        # self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv = nn.ConvTranspose2d(channels, channels, 4, 2, 1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        # x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        stride = 2
        self.op = nn.Conv2d(channels, channels, 3, stride=stride, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    def __init__(self, channels:int,dropout,out_c=None, use_scale_shift_norm=False):
        super().__init__()
        self.channels = channels
        self.dropout = dropout
        self.out_c = out_c or channels
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_c, 3, padding=1),
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.out_c),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                nn.Conv2d(self.out_c, self.out_c, 3, padding=1)
            ),
        )

        if self.out_c == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_c, 3, padding=1)

    def forward(self, x):
        h = self.in_layers(x)
        h = self.out_layers(h)

        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.attention = QKVAttention()
        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))

    def forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])
        h = self.attention(qkv)
        h = h.reshape(b, -1, h.shape[-1])
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention.
    """

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (C * 3) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x C x T] tensor after attention.
        """
        ch = qkv.shape[1] // 3
        q, k, v = th.split(qkv, ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterward
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        return th.einsum("bts,bcs->bct", weight, v)


class UNet(nn.Module):
    def __init__(self, in_c, out_c, model_c, channel_mult, num_res_blocks, attention_res, num_heads=1, dropout=0,
                 dropout_start_res=16,use_scale_shift_norm=True, use_skip_connection_coef=False, use_first_down=False,
                 in_size=0):  # in_size 为输入的尺寸
        super().__init__()

        self.in_c = in_c
        self.model_c = model_c
        self.out_c = out_c
        self.num_res_blocks = num_res_blocks
        self.attention_res = attention_res
        self.dropout = dropout
        self.dropout_start_res = dropout_start_res
        self.channel_mult = channel_mult
        self.num_heads = num_heads
        self.skip_connection_coef = 2 ** (-0.5) if use_skip_connection_coef else 1

        conv = nn.Conv2d(in_c, model_c, 3, stride=2, padding=1) if use_first_down else nn.Conv2d(in_c, model_c, 3, padding=1)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv)]
        )
        input_block_chans = [model_c]
        ch = model_c
        ds = in_size // 2 if use_first_down else in_size
        layer_dropout = dropout if ds <= dropout_start_res else 0
        for level, mult in enumerate(channel_mult):
            num_res_block = num_res_blocks[level]
            for _ in range(num_res_block):
                layers: list[nn.Module] = [
                    ResBlock(
                        ch,
                        layer_dropout,
                        out_c=mult * model_c,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_c
                if ds in attention_res:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch))
                )
                input_block_chans.append(ch)
                ds //= 2
                layer_dropout = dropout if ds <= dropout_start_res else 0
            # end if
        # end for


        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                layer_dropout,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(ch, num_heads=num_heads),
            ResBlock(
                ch,
                layer_dropout,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            num_res_block = num_res_blocks[level]
            for i in range(num_res_block + 1):
                layers = [
                    ResBlock(
                        ch + input_block_chans.pop(),
                        layer_dropout,
                        out_c=model_c * mult,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_c * mult
                if ds in attention_res:
                    layers.append(
                        AttentionBlock(ch, num_heads=num_heads,)
                    )
                if level and i == num_res_block:
                    layers.append(Upsample(ch))
                    ds *= 2
                    layer_dropout = dropout if ds <= dropout_start_res else 0
                self.output_blocks.append(TimestepEmbedSequential(*layers))
            # end for
        # end for

        conv = nn.ConvTranspose2d(model_c, out_c, 4, 2, 1) if use_first_down else nn.Conv2d(model_c, out_c, 3, padding=1)
        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            zero_module(conv),
        )


    def forward(self, y):
        hs = []

        h = y
        for module in self.input_blocks:
            h = module(h)
            hs.append(h)
        h = self.middle_block(h)
        for module in self.output_blocks:
            temp = hs.pop() * self.skip_connection_coef
            cat_in = th.cat([h, temp], dim=1)
            h = module(cat_in)
        h = self.out(h)
        return h


