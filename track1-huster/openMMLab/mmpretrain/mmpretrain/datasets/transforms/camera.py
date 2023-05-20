#coding=utf-8
#{Author: ZeLun Zhang, Date: 2023-05-12}


'''主要是针对采集环境做一些增强，
'''


import numpy as np

import imgaug.augmenters as iaa
from PIL import Image

from mmcv.transforms import BaseTransform, Compose

from mmpretrain.registry import TRANSFORMS

from collections.abc import Iterable
import random
import albumentations as A

@TRANSFORMS.register_module()
class Spatter(BaseTransform): # 溅出水花, 值最大5
    def __init__(self, severity_range = 1):
        if not isinstance(severity_range, Iterable): # 固定值
            self.severity_range = [severity_range, severity_range]
        self.severity_range = list(severity_range)
        assert len(self.severity_range) == 2, f'serverity should has length of 2'
        assert self.severity_range[1] >= self.severity_range[0], f'severity_range[1] must no smaller than severity_range[0]'

    def transform(self, results: dict) -> dict:
        img = results['img']
        results['img'] = self.__call(img)
        return results

    def __call(self, img):
        '''img本来就是np.ndarray
        '''
        theSeverity = random.randint(*self.severity_range)
        seq = iaa.Sequential([
            iaa.imgcorruptlike.Spatter(severity=theSeverity)
        ])
        image_aug = seq.augment_image(img)
        return image_aug

    def __repr__(self):
        return self.__class__.__name__ + '(severity={0})'.format(self.severity_range)

@TRANSFORMS.register_module()
class Rain(BaseTransform):
    drop_width = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
    blur_value = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
    brightness_coefficient = {1:0.9, 2:0.8, 3:0.7, 4:0.6, 5:0.5}
    def __init__(self, severity_range = 1):
        if not isinstance(severity_range, Iterable): # 固定值
            self.severity_range = [severity_range, severity_range]
        self.severity_range = list(severity_range)
        assert len(self.severity_range) == 2, f'serverity should has length of 2'
        assert self.severity_range[1] >= self.severity_range[0], f'severity_range[1] must no smaller than severity_range[0]'

    def transform(self, results: dict) -> dict:
        img = results['img']
        results['img'] = self.__call(img)
        return results

    def __call(self, img):
        theSeverity = random.randint(*self.severity_range)
        bc = self.brightness_coefficient[theSeverity]; dw = self.drop_width[theSeverity]; bv = self.blur_value[theSeverity]
        aug = A.RandomRain(brightness_coefficient=bc, drop_width=dw, blur_value=bv, p=1)
        img = aug(image = img)['image']
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(severity={0})'.format(self.severity_range)

@TRANSFORMS.register_module()
class Fog(BaseTransform):
    def __init__(self, severity_range = 1):
        if not isinstance(severity_range, Iterable): # 固定值
            self.severity_range = [severity_range, severity_range]
        self.severity_range = list(severity_range)
        assert len(self.severity_range) == 2, f'serverity should has length of 2'
        assert self.severity_range[1] >= self.severity_range[0], f'severity_range[1] must no smaller than severity_range[0]'

    def transform(self, results: dict) -> dict:
        img = results['img']
        results['img'] = self.__call(img)
        return results

    def __call(self, img):
        theSeverity = random.randint(*self.severity_range)
        seq = iaa.Sequential([
            iaa.imgcorruptlike.Fog(severity=theSeverity)
        ])
        image_aug = seq.augment_image(img)
        return image_aug

    def __repr__(self):
        return self.__class__.__name__ + '(severity={0})'.format(self.severity)

@TRANSFORMS.register_module()
class Frost(BaseTransform):
    def __init__(self, severity_range = 1):
        if not isinstance(severity_range, Iterable): # 固定值
            self.severity_range = [severity_range, severity_range]
        self.severity_range = list(severity_range)
        assert len(self.severity_range) == 2, f'serverity should has length of 2'
        assert self.severity_range[1] >= self.severity_range[0], f'severity_range[1] must no smaller than severity_range[0]'

    def transform(self, results: dict) -> dict:
        img = results['img']
        results['img'] = self.__call(img)
        return results

    def __call(self, img):
        theSeverity = random.randint(*self.severity_range)
        seq = iaa.Sequential([
            iaa.imgcorruptlike.Frost(severity=theSeverity)
        ])
        image_aug = seq.augment_image(img)
        return image_aug

    def __repr__(self):
        return self.__class__.__name__ + '(severity={0})'.format(self.severity)

@TRANSFORMS.register_module()
class Snow(BaseTransform):
    def __init__(self, severity_range = 1):
        if not isinstance(severity_range, Iterable): # 固定值
            self.severity_range = [severity_range, severity_range]
        self.severity_range = list(severity_range)
        assert len(self.severity_range) == 2, f'serverity should has length of 2'
        assert self.severity_range[1] >= self.severity_range[0], f'severity_range[1] must no smaller than severity_range[0]'

    def transform(self, results: dict) -> dict:
        img = results['img']
        results['img'] = self.__call(img)
        return results

    def __call(self, img):
        theSeverity = random.randint(*self.severity_range)
        seq = iaa.Sequential([
            iaa.imgcorruptlike.Snow(severity=theSeverity)
        ])
        image_aug = seq.augment_image(img)
        return image_aug

    def __repr__(self):
        return self.__class__.__name__ + '(severity={0})'.format(self.severity)