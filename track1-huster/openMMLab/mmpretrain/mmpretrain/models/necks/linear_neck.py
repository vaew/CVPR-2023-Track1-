# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmengine.model import BaseModule

from mmpretrain.registry import MODELS


@MODELS.register_module()
class LinearNeck(BaseModule):
    """Linear neck with Dimension projection.

    Args:
        in_channels (int): Number of channels in the input.
        out_channels (int): Number of channels in the output.
        gap_dim (int): Dimensions of each sample channel, can be one of
            {0, 1, 2, 3}. Defaults to 0.
        norm_cfg (dict, optional): dictionary to construct and
            config norm layer. Defaults to dict(type='BN1d').
        act_cfg (dict, optional): dictionary to construct and
            config activate layer. Defaults to None.
        init_cfg (dict, optional): dictionary to initialize weights.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 gap_dim: int = 0,
                 norm_cfg: Optional[dict] = dict(type='BN1d'),
                 act_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg=init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_cfg = copy.deepcopy(norm_cfg)
        self.act_cfg = copy.deepcopy(act_cfg)

        assert gap_dim in [0, 1, 2, 3], 'GlobalAveragePooling dim only ' \
            f'support {0, 1, 2, 3}, get {gap_dim} instead.'
        if gap_dim == 0:
            self.gap = nn.Identity()
        elif gap_dim == 1:
            self.gap = nn.AdaptiveAvgPool1d(1)
        elif gap_dim == 2:
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
        elif gap_dim == 3:
            self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fc = nn.Linear(in_features=in_channels, out_features=out_channels)

        if norm_cfg:
            self.norm = build_norm_layer(norm_cfg, out_channels)[1]
        else:
            self.norm = nn.Identity()

        if act_cfg:
            self.act = build_activation_layer(act_cfg)
        else:
            self.act = nn.Identity()

    def forward(self, inputs: Union[Tuple,
                                    torch.Tensor]) -> Tuple[torch.Tensor]:
        """forward function.

        Args:
            inputs (Union[Tuple, torch.Tensor]): The features extracted from
                the backbone. Multiple stage inputs are acceptable but only
                the last stage will be used.

        Returns:
            Tuple[torch.Tensor]: A tuple of output features.
        """
        assert isinstance(inputs, (tuple, torch.Tensor)), (
            'The inputs of `LinearNeck` must be tuple or `torch.Tensor`, '
            f'but get {type(inputs)}.')
        if isinstance(inputs, tuple):
            inputs = inputs[-1]

        x = self.gap(inputs)
        x = x.view(x.size(0), -1)
        out = self.act(self.norm(self.fc(x)))
        return (out, )


@MODELS.register_module()
class MultiLinearNeck(BaseModule):
    def __init__(self,
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
        super(MultiLinearNeck, self).__init__(init_cfg=init_cfg, **kwargs)

        self.in_channels = in_channels
        self.latent_channels = latent_channels  # 隐藏层设置
        self.use_shortcut = use_shortcut

        if self.use_shortcut:
            assert latent_channels[0][0] == 'bn', 'first layer of latent channel should be BN if shortcut is used'
            assert latent_channels[0][1] == self.in_channels, f'first BN should has exactly same channel with neck output.'
            assert latent_channels[-1][1] == self.in_channels, f'out channels shoule equal to in_channels when use shortcut.'

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

        self.gap = nn.AdaptiveAvgPool2d((1,1))

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
        feats = self.gap(feats)
        feats = feats.view(feats.size(0), -1)

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
        return pre_logits # 返回用于cls的特征