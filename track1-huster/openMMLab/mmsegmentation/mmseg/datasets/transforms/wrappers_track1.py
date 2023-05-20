#coding=utf-8


from mmseg.registry import TRANSFORMS
from mmcv.transforms.base import BaseTransform
import mmengine
from mmcv.transforms import Compose

import numpy as np

from typing import Any, Callable, Dict, List, Optional, Sequence, Union
Transform = Union[Dict, Callable[[Dict], Dict]]

@TRANSFORMS.register_module()
class RandomChoiceSegTrack1(BaseTransform):
    '''针对track1 seg修改，随着iter的进行降低mosaic的概率
    '''

    def __init__(self,
                 transforms: List[Union[Transform, List[Transform]]],
                 prob: Optional[List[float]] = None,
                 total_iter_num = None, # 整个训练流程总iter数量
                 mosaic_shutdown_iter_ratio = [0.4, 0.5], # 在多少比例的iter时调整mosaic的ratio
                 mosaic_use_prob = [0.1, 0.0], # 与上面对应，在多少iter时以多少概率使用mosaic
                 ):

        super().__init__()

        if prob is not None:
            assert mmengine.is_seq_of(prob, float)
            assert len(transforms) == len(prob), \
                '``transforms`` and ``prob`` must have same lengths. ' \
                f'Got {len(transforms)} vs {len(prob)}.'
            assert sum(prob) == 1

        assert total_iter_num is not None

        self.prob = prob
        self.transforms = [Compose(transforms) for transforms in transforms]

        self.iter_cnt = 0 # 当前调用多少次，经历了多少iter
        self.mosaic_shutdown_iter_ratio = mosaic_shutdown_iter_ratio
        self.mosaic_use_prob = mosaic_use_prob
        self.total_iter_num = total_iter_num

        for idx in range(len(self.mosaic_shutdown_iter_ratio) - 1): # 保证 mosaic_shutdown_iter_ratio 中时递增的
            assert self.mosaic_shutdown_iter_ratio[idx] < self.mosaic_shutdown_iter_ratio[idx+1]

    def __iter__(self):
        return iter(self.transforms)

    def random_pipeline_index(self) -> int:
        """Return a random transform index."""
        indices = np.arange(len(self.transforms))
        return np.random.choice(indices, p=self.prob)

    def transform(self, results: Dict) -> Optional[Dict]:
        """Randomly choose a transform to apply."""
        self.modify_mosaic_prob()
        idx = self.random_pipeline_index()
        if self.iter_cnt % 50 == 0: # 仅仅是debug
            print('RandomChoiceSegTrack1 use prob:', self.prob)
            print('RandomChoiceSegTrack1 use idx:', idx)
        return self.transforms[idx](results)

    def modify_mosaic_prob(self): # 修改使用mosaic的概率
        self.iter_cnt += 1
        cur_iter_ratio = self.iter_cnt / self.total_iter_num # 计算当前的iter到哪个ratio了
        cur_ratio_idx = None
        for idx, ratio in enumerate(self.mosaic_shutdown_iter_ratio[::-1]):
            if cur_iter_ratio > ratio:
                cur_ratio_idx = idx
                break
        if cur_ratio_idx is not None:
            mosaic_prob = self.mosaic_use_prob[-cur_ratio_idx-1]
            self.prob = [mosaic_prob, 1. - mosaic_prob]
        

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(transforms = {self.transforms}'
        repr_str += f'prob = {self.prob})'
        return repr_str