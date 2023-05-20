#coding=utf-8
#{Author: ZeLun Zhang, Date: 2023.04.15}


import random
from numbers import Number
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.dist import barrier, broadcast, get_dist_info
from mmengine.logging import MessageHub
from mmengine.model import BaseDataPreprocessor, ImgDataPreprocessor
from mmengine.structures import PixelData
from mmengine.utils import is_seq_of
from torch import Tensor

from mmdet.models.utils import unfold_wo_center
from mmdet.models.utils.misc import samplelist_boxtype2tensor
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample
from mmdet.structures.mask import BitmapMasks
from mmdet.utils import ConfigType

from mmseg.models import HEADS as MMSEG_MODELS
from mmpretrain.models import HEADS as MMPRETRAIN_MODELS
from mmdet.registry import MODELS as MMDET_MODELS


try:
    import skimage
except ImportError:
    skimage = None

from torch import nn

@MODELS.register_module()
class MultiTaskDataPreprocessor(nn.Module):
    '''多个pp的封装
    '''
    def __init__(self, det, seg, cls):
        super().__init__()
        self.task2preprocesser = nn.ModuleDict()
        self.task2preprocesser['det'] = MMDET_MODELS.build(det)
        self.task2preprocesser['seg'] = MMSEG_MODELS.build(seg)
        self.task2preprocesser['cls'] = MMPRETRAIN_MODELS.build(cls)

    def __call__(self, data: dict, training: bool = False) -> dict:
        '''输出:
           {'det': 原来直接输入给检测的数据, 'seg': 原来直接输入给分割的数据, 'cls': 原来直接输入给分类的数据}
        '''
        rst_dict = {}
        if 'det' in data:
            det_rst= self.task2preprocesser['det'](data['det'], training)
            rst_dict['det'] = det_rst
        if 'seg' in data:
            seg_rst = self.task2preprocesser['seg'](data['seg'], training)
            rst_dict['seg'] = seg_rst
        if 'cls' in data:
            cls_rst = self.task2preprocesser['cls'](data['cls'], training)
            rst_dict['cls'] = cls_rst
        return rst_dict