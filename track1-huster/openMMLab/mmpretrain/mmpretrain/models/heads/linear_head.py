# Copyright (c) OpenMMLab. All rights reserved.
from re import L
from typing import Optional, Tuple

import torch
import torch.nn as nn

from mmpretrain.registry import MODELS
from .cls_head import ClsHead


@MODELS.register_module()
class LinearClsHead(ClsHead):
    """Linear classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        loss (dict): Config of classification loss. Defaults to
            ``dict(type='CrossEntropyLoss', loss_weight=1.0)``.
        topk (int | Tuple[int]): Top-k accuracy. Defaults to ``(1, )``.
        cal_acc (bool): Whether to calculate accuracy during training.
            If you use batch augmentations like Mixup and CutMix during
            training, it is pointless to calculate accuracy.
            Defaults to False.
        init_cfg (dict, optional): the config to control the initialization.
            Defaults to ``dict(type='Normal', layer='Linear', std=0.01)``.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 init_cfg: Optional[dict] = dict(
                     type='Normal', layer='Linear', std=0.01),
                 **kwargs):
        super(LinearClsHead, self).__init__(init_cfg=init_cfg, **kwargs)

        self.in_channels = in_channels
        self.num_classes = num_classes

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self.fc = nn.Linear(self.in_channels, self.num_classes)

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The process before the final classification head.

        The input ``feats`` is a tuple of tensor, and each tensor is the
        feature of a backbone stage. In ``LinearClsHead``, we just obtain the
        feature of the last stage.
        """
        # The LinearClsHead doesn't have other module, just return after
        # unpacking.
        return feats[-1]

    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The forward process."""
        pre_logits = self.pre_logits(feats)
        # The final classification head.
        cls_score = self.fc(pre_logits)
        return cls_score



@MODELS.register_module()
class MultiLinearClsHead(ClsHead):
    def __init__(self,
                 num_classes: int, # 196
                 in_channels: int, # 896
                 latent_channels = [('bn', 896), ('fc', 1024), ('bn', 1024), ('relu', 1024), ('fc', 896), ('bn', 896)], 
                 use_shortcut = True,
                 init_cfg: Optional[dict] = dict(
                     type='Normal', layer='Linear', std=0.01),
                 **kwargs):
        '''根据latent_channels自动生成MLP
           num_classes: 最后分类的数量
           in_channels: neck输出的通道数量
           latent_channels: 中间通道的设置
           use_shortcut: 最后输出的分类特征是否与backbone的特征相加
        '''
        super(MultiLinearClsHead, self).__init__(init_cfg=init_cfg, **kwargs)

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.latent_channels = latent_channels  # 隐藏层设置
        self.use_shortcut = use_shortcut

        if self.use_shortcut:
            assert latent_channels[0][0] == 'bn', 'first layer of latent channel should be BN if shortcut is used'
            assert latent_channels[0][1] == self.in_channels, f'first BN should has exactly same channel with neck output.'
            assert latent_channels[-1][1] == self.in_channels, f'out channels shoule equal to in_channels when use shortcut.'

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        last_feat_dim = self.latent_channels[-1][-1]
        self.fc = nn.Linear(last_feat_dim, self.num_classes) # 最后用于分类的头

        latent_layers_build_idx = 0
        self.fst_bn = None
        if self.use_shortcut: # 如果有shortcut, 则单独建立最先的BN结构
            fst_bn_cfg = self.latent_channels[0]
            latent_layers_build_idx += 1
            self.fst_bn = nn.BatchNorm1d(fst_bn_cfg[-1]) # 如果有shortcut一开始的bn

        latent_layers = []
        last_layer_channels = self.in_channels
        for idx in range(latent_layers_build_idx, len(self.latent_channels)): # 依次建立后面的层
            layer_cfg = self.latent_channels[idx]
            if layer_cfg[0] == 'fc':
                latent_layers.append(nn.Linear(last_layer_channels, layer_cfg[-1]))
                last_layer_channels = layer_cfg[-1]
            elif layer_cfg[0] == 'bn':
                latent_layers.append(nn.BatchNorm1d(last_layer_channels))
            elif layer_cfg[0] == 'relu':
                latent_layers.append(nn.ReLU())
            else:
                raise NotImplementedError(f'layer type {layer_cfg[0]} not implemented.')
        self.latent_layers = nn.Sequential(*latent_layers) # 所有的层

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The process before the final classification head.

        The input ``feats`` is a tuple of tensor, and each tensor is the
        feature of a backbone stage. In ``LinearClsHead``, we just obtain the
        feature of the last stage.
        """
        # The LinearClsHead doesn't have other module, just return after
        # unpacking.
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]
        if self.use_shortcut:
            assert self.fst_bn is not None, f'the first layer should be BN when use shortcut.'
            feats = self.fst_bn(feats)

        outs = self.latent_layers(feats)

        if self.use_shortcut:
            outs += feats

        return outs

    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The forward process."""
        pre_logits = self.pre_logits(feats)
        # The final classification head.
        cls_score = self.fc(pre_logits)
        return cls_score