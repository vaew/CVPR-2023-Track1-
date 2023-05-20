# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS
from ..utils import resize


@MODELS.register_module()
class SegformerHead(BaseDecodeHead):
    """The all mlp Head of segformer.

    This head is the implementation of
    `Segformer <https://arxiv.org/abs/2105.15203>` _.

    Args:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'bilinear'.
    """

    def __init__(self, interpolate_mode='bilinear', **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs) # 多个level特征放入一个List中, 传入 decder_head

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        self.convs = nn.ModuleList()
        for i in range(num_inputs): # 4-level输入，4个1x1卷积
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg)
                )

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs, # concat起来，之后1x1 conv 来 4*256 -> 256
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg
        )

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs) # 通过 self.in_inputs 到 inputs中取索引
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append( # 先每个level变成256，然后双线性插值到原始输入的 1/4
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:], # 这里默认inputs是原始输入的1/4
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners
                    )
                )

        out = self.fusion_conv(torch.cat(outs, dim=1)) # 全部concat然后fusion

        out = self.cls_seg(out) # 变成最后的分类

        return out
