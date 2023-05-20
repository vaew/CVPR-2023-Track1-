#coding=utf-8
#{Author: ZeLun Zhang, Date: 2023-5-15}


import copy, math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmengine.model import BaseModule
from mmcv.cnn import ConvModule
from torch.nn.init import normal_
import torch.nn.functional as F
import warnings
from torch.nn import init

from mmpretrain.registry import MODELS

def _trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)
    tensor.uniform_(2 * l - 1, 2 * u - 1)

    # Use inverse cdf transform for normal distribution to get truncated
    # standard normal
    tensor.erfinv_()

    # Transform to proper mean, std
    tensor.mul_(std * math.sqrt(2.))
    tensor.add_(mean)

    # Clamp to ensure it's in the proper range
    tensor.clamp_(min=a, max=b)
    return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    NOTE: this impl is similar to the PyTorch trunc_normal_, the bounds [a, b] are
    applied while sampling the normal with mean/std applied, therefore a, b args
    should be adjusted to match the range of mean, std args.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    with torch.no_grad():
        return _trunc_normal_(tensor, mean, std, a, b)

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    '''将一个batch中随机一些样本设置为全0，让这个样本没有经过某个path的计算，需要和shortcut联合使用 
    '''
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # (B, 1,..., 1), 对标batch维度
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob) # 随机选择一个作为0
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob) # 按照keepdim来放大整个batch的数值
    return x * random_tensor # 随机将一些样本的整个计算结果变成0，相当于这个样本没有参与这个path的计算

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self):
        return 'p={}'.format(self.drop_prob)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        '''transformer block的基础self-attn
           dim: 输入特征的原始维度
           num_heads: self-attn有几个头
           qkv_bias: 是否计算qkv时使用bias
           attn_drop: 对 attn map 使用dropout的概率
           proj_drop: 对cat后经过的linear使用dropout的概率
        '''
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape # B对应每个样本，N对应每个token，C对应每个token的channel
        qkv = self.qkv(x) # [B, N, 3*C], 最后的3对应qkv, 是8个head的qkv的concat, 理论上每个head的qkv都是在输入上做linear，可以一次性做了
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads) # [B, N, 3, num_heads, dim // num_heads]
        qkv = qkv.permute(2, 0, 3, 1, 4) # [3, B, num_heads, N, dim // num_heads]
        q, k, v = qkv.unbind(0)   # 分解qkv
        # q, k, v: [B, num_heads, N, dim // num_heads]
        attn = ((q * self.scale) @ k.transpose(-2, -1)) # [B, num_heads, N, N], 获得每个样本下每个token之间的attn
        attn = attn.softmax(dim=-1) # 1、先做softmax
        attn = self.attn_drop(attn) # 2、对softmax做drop, 同比例放大分数

        x = (attn @ v) # [B, num_heads, N, dim // num_heads], 乘v做self-attn
        x = x.transpose(1, 2) # [B, N, num_heads, dim // num_heads]
        x = x.reshape(B, N, C) # 每个头分开做self-attn，之后concat起来
        x = self.proj(x) # 1、经过linear
        x = self.proj_drop(x) # 2、经过dropout获得最终输出
        return x


class Mlp(nn.Module):
    '''transformer中经过一个mlp，transformer的最后一个阶段
    '''
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransformerBlock(nn.Module):
    '''基础的transformer block
    '''
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        '''dim: 这一层特征维度
           num_heads: self-attn 注意力头数量
           mlp_ratio: 第二阶段mlp的扩张维度数量
           qkv_bias: 是否使用qkv_bias
           drop: 第二阶段 MLP 中 dropout 概率, 和 self-attn 中 linear 的drop概率
           attn_drop: self-attn 的 attn dropout 概率
           drop_path: 
           act_layer: 用什么激活
           norm_layer: 用什么正则化
        '''
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

@MODELS.register_module()
class MultiScaleVIT(BaseModule):
    '''对不同level的特征cat后做Vit
    '''
    def __init__(
        self,
        in_channels, # backbone输出的每个level通道数
        in_hw_dims, # backbone输出的每个level的分辨率
        feat_num, # 每个level先映射到同一个级别
        transformer_layers, # 用多少层transformer block
        transformer_num_heads, # 每个trans block用几个head
        use_cls_token = True,
        use_position_embedding = True, # 是否使用 posi embedding
        use_level_embedding = True, # 是否使用level embedding
        init_cfg = dict(
            type='Xavier', layer='Conv2d', distribution='uniform'
        )
    ):

        super().__init__(init_cfg=init_cfg)

        assert len(in_hw_dims) == len(in_channels)
        assert use_cls_token, f'currently only support use cls token'
        self.num_levels = len(in_channels)
        self.in_channels = in_channels
        self.in_hw_dims = in_hw_dims
        self.feat_num = feat_num
        self.transformer_layers = transformer_layers
        self.use_position_embedding = use_position_embedding
        self.use_level_embedding = use_level_embedding
        self.transformer_num_heads = transformer_num_heads
        self.use_cls_token = use_cls_token


        # 首先构造channer mapper
        self.channer_mappers = nn.ModuleList()
        for in_channel in self.in_channels:
            self.channer_mappers.append(
                ConvModule(
                    in_channel,
                    self.feat_num,
                    kernel_size=1,
                    stride = 1,
                    padding=0,
                    conv_cfg=None, # conv2D卷积
                    act_cfg=None, # 不使用激活函数
                    norm_cfg=dict(type='GN', num_groups=32), # 使用GN
                )
            )

        # 其次构造ViT结构
        vit_lst = []
        for i in range(self.transformer_layers):
            vit_lst.append(
                TransformerBlock(
                    dim = self.feat_num, 
                    num_heads = self.transformer_num_heads
                )
            )
        self.vit = nn.Sequential(*vit_lst)

        # 构建各种embedding
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.Tensor(1,1, self.feat_num))
            normal_(self.cls_token)

        if self.use_position_embedding:
            self.posi_embedding = nn.ParameterList()
            for in_channel, in_hw_dim in zip(self.in_channels, self.in_hw_dims):
                h, w = in_hw_dim
                self.posi_embedding.append(
                    nn.Parameter(torch.Tensor(1, self.feat_num, h, w))
                )
            for embed in self.posi_embedding:
                normal_(embed)

        if self.use_level_embedding:
            self.level_embedding = nn.Parameter(torch.Tensor(self.num_levels, self.feat_num))
            normal_(self.level_embedding)

        # 最后输出的分类特征映射回原来backbone的维度
        self.last_linear = nn.Linear(self.feat_num, self.in_channels[-1]) 


    def forward(self, inputs: Tuple[torch.Tensor]):
        '''inputs: backbone输出的多尺度特征
        '''
        assert len(inputs) == len(self.channer_mappers)
        outs = [self.channer_mappers[i](inputs[i]) for i in range(self.num_levels)] # [b,feat_num,h,w]

        if self.use_position_embedding:
            outs = [out + posi_embed for out, posi_embed in zip(outs, self.posi_embedding)]

        bs, *_ = outs[0].shape
        outs = [outs[i].reshape(bs, self.feat_num, -1).permute(0, 2, 1) for i in range(self.num_levels)] # [bs, h*w, feat_num]

        if self.use_level_embedding:
            outs = [outs[i] + self.level_embedding[i] for i in range(self.num_levels)]

        outs = torch.cat(outs, dim = 1) # [bs, num_level * spatial_dim, feat_dim]

        if self.use_cls_token:
            cls_token = torch.tile(self.cls_token, (bs, 1, 1)) # [bs, 1, num_feats]
            outs = torch.cat([cls_token, outs], dim = 1) # [bs, num_level * spatial_dim + 1, num_feats]

        out = self.vit(outs)

        if self.use_cls_token:
            out = out[:, 0] # [bs, num_feats]

        out = self.last_linear(out) # 经过最后一个linear映射回backbone最后一个维度的特征

        return (out, )


class PatchMergeBlock(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        '''输入x: [B,N,C], 其中 N = H * W
        '''
        B, N, C = x.shape

        hw = int(math.sqrt(N))

        x = x.reshape([-1, hw, hw, C])

        pad_input = (hw % 2 == 1) or (hw % 2 == 1) # 如果一方面是奇数，则需要pad input
        if pad_input:
            x = F.pad(x, [0, 0, 0, hw % 2, 0, hw % 2, 0, 0], data_format='NHWC')
            hw += hw % 2
            hw += hw % 2

        x0 = x[:, 0::2, 0::2, :] # [b, hw/2, hw/2, C]
        x1 = x[:, 1::2, 0::2, :]  
        x2 = x[:, 0::2, 1::2, :] 
        x3 = x[:, 1::2, 1::2, :] 
        x = torch.cat([x0, x1, x2, x3], -1)  # [b, hw/2, hw/2, 4*C]
        x = x.reshape([-1, hw * hw // 4, 4 * C])  # [b, hw*hw/4, 4*C]

        x = self.norm(x)
        x = self.reduction(x) # [b, hw*hw/4, 2*C]

        return x

class ECAAttention(nn.Module):
    def __init__(self, kernel_size=3, use_cls_token=False):
        super().__init__()
        self.gap=nn.AdaptiveAvgPool2d(1)
        self.conv=nn.Conv1d(1,1,kernel_size=kernel_size,padding=(kernel_size-1)//2)
        self.sigmoid=nn.Sigmoid()
        self.use_cls_token = use_cls_token

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, H, W):
        if self.use_cls_token:
            cls_feat = x[:, :1, :]
            x = x[:, 1:, :]
        
        B = x.shape[0]
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        y = self.gap(x) #bs,c,1,1
        y = y.squeeze(-1).permute(0,2,1) #bs,1,c
        y = self.conv(y) #bs,1,c
        y = self.sigmoid(y) #bs,1,c
        y = y.permute(0,2,1).unsqueeze(-1) #bs,c,1,1
        y = x * y.expand_as(x)
        
        if self.use_cls_token:
            return torch.cat([cls_feat, y.reshape(B, -1, H * W).permute(0, 2, 1)], dim=1)
        else:
            return y.reshape(B, -1, H * W).permute(0, 2, 1)


class ChannelAttention(nn.Module):
    def __init__(self,channel,reduction=16):
        super().__init__()
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.se=nn.Sequential(
            nn.Conv2d(channel,channel//reduction,1,bias=False),
            nn.ReLU(),
            nn.Conv2d(channel//reduction,channel,1,bias=False)
        )
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result=self.maxpool(x)
        avg_result=self.avgpool(x)
        max_out=self.se(max_result)
        avg_out=self.se(avg_result)
        output=self.sigmoid(max_out+avg_out)
        return output

class SpatialAttention(nn.Module):
    def __init__(self,kernel_size=7):
        super().__init__()
        self.conv=nn.Conv2d(2,1,kernel_size=kernel_size,padding=kernel_size//2)
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result,_=torch.max(x,dim=1,keepdim=True)
        avg_result=torch.mean(x,dim=1,keepdim=True)
        result=torch.cat([max_result,avg_result],1)
        output=self.conv(result)
        output=self.sigmoid(output)
        return output


class CBAMBlock(nn.Module):
    def __init__(self, channel=512,reduction=16,kernel_size=7, use_cls_token=False):
        super().__init__()
        self.ca = ChannelAttention(channel=channel,reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)
        self.use_cls_token = use_cls_token


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, H, W):
        # import pdb; pdb.set_trace()
        if self.use_cls_token:
            cls_feat = x[:, :1, :]
            x = x[:, 1:, :]
        B = x.shape[0]
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        b, c, _, _ = x.size()
        residual = x
        out = x * self.ca(x)
        out = out * self.sa(out) + residual
        if self.use_cls_token:
            return torch.cat([cls_feat, out.reshape(B, -1, H * W).permute(0, 2, 1)], dim=1)
        else:
            return out.reshape(B, -1, H * W).permute(0, 2, 1)


class SEAttention(nn.Module):

    def __init__(self, channel=512, reduction=16, use_cls_token=False):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.use_cls_token = use_cls_token

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, H, W):
        if self.use_cls_token:
            cls_feat = x[:, :1, :]
            x = x[:, 1:, :]
        B = x.shape[0]
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = x * y.expand_as(x)
        
        if self.use_cls_token:
            return torch.cat([cls_feat, y.reshape(B, -1, H * W).permute(0, 2, 1)], dim=1)
        else:
            return y.reshape(B, -1, H * W).permute(0, 2, 1)

@MODELS.register_module()
class MultiScaleVIT_(BaseModule):

    def __init__(self, # 相当于是在backbone旁边拼接了一个小模型，但是共用backbone第一个level的特征
                in_features, # backbone输出的每层特征的维度
                in_spatial_dims, # 每层level特征的空间维度
                transformer_block_nums=1, # 每一层几个transformer block
                transformer_block_num_heads=2, # 每个trans block头数量
                gate_T=0.1, 
                reduct_after_use_attention=None, 
                use_position_embedding = True, # 是否在每一层使用posi_embedding
                use_shortcut = True, # 最后特征和gap特征相加
                init_cfg = dict(
                    type='Xavier', layer='Conv2d', distribution='uniform'
                )
            ):
        super(MultiScaleVIT_, self).__init__(init_cfg=init_cfg)
        
        assert len(in_features) == len(in_spatial_dims)
        self.in_features = in_features
        self.in_spatial_dims = in_spatial_dims
        self.transformer_block_nums = transformer_block_nums
        self.transformer_block_num_heads = transformer_block_num_heads
        self.reduct_after_use_attention = reduct_after_use_attention
        self.use_position_embedding = use_position_embedding
        self.use_shortcut = use_shortcut

        self.gate_T = gate_T
        
        # Gate参数，本质就是两个模型结果融合，通过Gate来控制两个结果的融合权重，类似MoE
        self.side_gate_params = nn.ParameterList(
            [nn.Parameter(torch.zeros(1)) for i in range(len(self.in_features)-1)] # 初始化1:1融合
        )
        if self.use_shortcut: # 用于控制最后的GAP + shortcut
            self.shortcut_gate_param = nn.Parameter(torch.zeros(1))

        # 构建每一层小模型用的transformer
        self.transformer_blocks = nn.ModuleList()
        for i in range(len(self.in_features)): # 对每个level的特征都需要子网络
            sub_blocks = nn.ModuleList()
            dim = self.in_features[i]
            for _ in range(self.transformer_block_nums):
                sub_blocks.append(TransformerBlock(dim=dim, num_heads=self.transformer_block_num_heads))
            self.transformer_blocks.append(sub_blocks)

        # 降低空间维度, down2top融合时降低空间分辨率
        self.downsample_spatial = nn.ModuleList()
        for in_feature in self.in_features[:-1]:
            self.downsample_spatial.append(
                PatchMergeBlock(dim = in_feature)
            )

        # 生成posi_embedding
        if self.use_position_embedding: # 对backbone中每个level特征用的posi_embedding
            self.posi_embedding = nn.ParameterList(
                [
                    torch.Tensor(1, c, h, w) for (c, (h, w)) in zip(self.in_features, self.in_spatial_dims)
                ]
            )
            for embed in self.posi_embedding:
                normal_(embed)
        
        if self.use_shortcut:
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.backbone_norm = nn.LayerNorm(self.in_features[-1])
            self.vit_norm = nn.LayerNorm(self.in_features[-1])

        self.last_proj = nn.Linear(self.in_features[-1], max(1024, self.in_features[-1])) # 添加一个最后的linear层, 映射到1024
    
        self.apply(self._init_weights)
    
    def forward(self, inputs: Tuple[torch.Tensor]):
        '''inputs: 来自backbone的multi-level特征, [B, C, H, W]
        '''
        #print('inputs:', [i.shape for i in inputs])
        if self.use_shortcut:
            backbone_last_out = inputs[-1]
        
        # 首先使用 posi_embedding
        inputs = [inp + posi_embed for inp, posi_embed in zip(inputs, self.posi_embedding)]

        # 把backbone的特征转换成transformer的形式
        inputs_ = []
        for inp in inputs:
            b, c, h, w = inp.shape
            inputs_.append(inp.reshape(b, c, -1).permute(0, 2, 1)) # [B, H*W, C]

        inputs = inputs_
        # 每一层的计算
        feats = None
        for i, block_feat in enumerate(inputs):
            if 0 == i:
                for transformer_block in self.transformer_blocks[i]:
                    feats = transformer_block(block_feat) # [B, H*W, C]
                feats = self.downsample_spatial[i](feats) # [B, H*W/4, 2*C]
                #print(feats.shape)
            else:
                gate = torch.sigmoid(self.side_gate_params[i - 1] / self.gate_T)

                feats = gate * feats + (1 - gate) * block_feat # MoE-like 的融合两个模型结果
                    
                for transformer_block in self.transformer_blocks[i]:
                    feats = transformer_block(feats)

                if i < (len(inputs) - 1):
                    feats = self.downsample_spatial[i](feats)
        # feats [B,N,C]
        
        feats = feats.mean(dim=1) # gap, [B,C]

        if self.use_shortcut:
            backbone_last_out = self.gap(backbone_last_out)
            b, c, *_ = backbone_last_out.shape
            backbone_last_out = backbone_last_out.reshape(b, c) # [B, C]
            backbone_last_out = self.backbone_norm(backbone_last_out)
            feats = self.vit_norm(feats)
            gate = torch.sigmoid(self.shortcut_gate_param / self.gate_T)
            feats = gate * feats + (1 - gate) * backbone_last_out

        out = self.last_proj(feats)

        return out
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)