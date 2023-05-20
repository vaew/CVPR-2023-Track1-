# Copyright (c) OpenMMLab. All rights reserved.
import collections
import copy
from typing import Sequence, Union

from mmengine.dataset import BaseDataset, force_full_init

from mmdet.registry import DATASETS, TRANSFORMS


@DATASETS.register_module()
class MultiImageMixDataset:
    """A wrapper of multiple images mixed dataset.

    Suitable for training on multiple images mixed data augmentation like
    mosaic and mixup. For the augmentation pipeline of mixed image data,
    the `get_indexes` method needs to be provided to obtain the image
    indexes, and you can set `skip_flags` to change the pipeline running
    process. At the same time, we provide the `dynamic_scale` parameter
    to dynamically change the output image size.

    Args:
        dataset (:obj:`CustomDataset`): The dataset to be mixed.
        pipeline (Sequence[dict]): Sequence of transform object or
            config dict to be composed.
        dynamic_scale (tuple[int], optional): The image scale can be changed
            dynamically. Default to None. It is deprecated.
        skip_type_keys (list[str], optional): Sequence of type string to
            be skip pipeline. Default to None.
        max_refetch (int): The maximum number of retry iterations for getting
            valid results from the pipeline. If the number of iterations is
            greater than `max_refetch`, but results is still None, then the
            iteration is terminated and raise the error. Default: 15.
    """

    def __init__(self,
                 dataset: Union[BaseDataset, dict],
                 pipeline: Sequence[str],
                 skip_type_keys: Union[Sequence[str], None] = None,
                 max_refetch: int = 15,
                 lazy_init: bool = False) -> None:
        assert isinstance(pipeline, collections.abc.Sequence)
        if skip_type_keys is not None:
            assert all([
                isinstance(skip_type_key, str)
                for skip_type_key in skip_type_keys
            ])
        self._skip_type_keys = skip_type_keys

        self.pipeline = []
        self.pipeline_types = []
        for transform in pipeline:
            if isinstance(transform, dict):
                self.pipeline_types.append(transform['type'])
                transform = TRANSFORMS.build(transform)
                self.pipeline.append(transform)
            else:
                raise TypeError('pipeline must be a dict')

        self.dataset: BaseDataset
        if isinstance(dataset, dict):
            self.dataset = DATASETS.build(dataset)
        elif isinstance(dataset, BaseDataset):
            self.dataset = dataset
        else:
            raise TypeError(
                'elements in datasets sequence should be config or '
                f'`BaseDataset` instance, but got {type(dataset)}')

        self._metainfo = self.dataset.metainfo
        if hasattr(self.dataset, 'flag'):
            self.flag = self.dataset.flag
        self.num_samples = len(self.dataset)
        self.max_refetch = max_refetch

        self._fully_initialized = False
        if not lazy_init:
            self.full_init()

    @property
    def metainfo(self) -> dict:
        """Get the meta information of the multi-image-mixed dataset.

        Returns:
            dict: The meta information of multi-image-mixed dataset.
        """
        return copy.deepcopy(self._metainfo)

    def full_init(self):
        """Loop to ``full_init`` each dataset."""
        if self._fully_initialized:
            return

        self.dataset.full_init()
        self._ori_len = len(self.dataset)
        self._fully_initialized = True

    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index.

        Args:
            idx (int): Global index of ``ConcatDataset``.

        Returns:
            dict: The idx-th annotation of the datasets.
        """
        return self.dataset.get_data_info(idx)

    @force_full_init
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        results = copy.deepcopy(self.dataset[idx]) # 从包裹的dataset中取出数据，会经过包裹的dataset的pipeline
        for (transform, transform_type) in zip(self.pipeline,
                                               self.pipeline_types): # 依次执行wrapper的pipeline中的transform
            if self._skip_type_keys is not None and \
                    transform_type in self._skip_type_keys:
                continue

            if hasattr(transform, 'get_indexes'): # 当前transform是mix的，则需要从self.dataset中取数据，需要经过dataset的pipeline
                mix_results = self._get_mix_results(transform)
                results['mix_results'] = mix_results

            # 对于RandomChoice, 其无 'get_indexes', 外层有额外包裹
            from mmcv.transforms import RandomChoice, Compose
            if isinstance(transform, RandomChoice):
                for compose_transforms in transform.transforms:
                    assert isinstance(compose_transforms, Compose), f'compose_transforms has type {compose_transforms.__class__.__name__}'
                    for item_transform in compose_transforms.transforms:
                        if item_transform.__class__.__name__ == 'Mosaic':
                            mix_results = self._get_mix_results(item_transform)
                            results['mosaic_mix_results'] = mix_results
                        elif item_transform.__class__.__name__ == 'MixUp':
                            mix_results = self._get_mix_results(item_transform)
                            results['mixup_mix_results'] = mix_results

            for i in range(self.max_refetch):
                # To confirm the results passed the training pipeline
                # of the wrapper is not None.
                updated_results = transform(copy.deepcopy(results))
                if updated_results is not None:
                    results = updated_results
                    break
            else:
                raise RuntimeError(
                    'The training pipeline of the dataset wrapper'
                    ' always return None.Please check the correctness '
                    'of the dataset and its pipeline.')

            if 'mix_results' in results:
                results.pop('mix_results')

        return results

    def _get_mix_results(self, transform):
        for i in range(self.max_refetch):
            # Make sure the results passed the loading pipeline
            # of the original dataset is not None.
            indexes = transform.get_indexes(self.dataset)
            if not isinstance(indexes, collections.abc.Sequence):
                indexes = [indexes]
            mix_results = [
                copy.deepcopy(self.dataset[index]) for index in indexes
            ]
            if None not in mix_results:
                #results['mix_results'] = mix_results # 取出的结果放在results的'mix_results'这个key中
                break
        else:
            raise RuntimeError(
                'The loading pipeline of the original dataset'
                ' always return None. Please check the correctness '
                'of the dataset and its pipeline.')
        return mix_results

    def update_skip_type_keys(self, skip_type_keys):
        """Update skip_type_keys. It is called by an external hook.

        Args:
            skip_type_keys (list[str], optional): Sequence of type
                string to be skip pipeline.
        """
        assert all([
            isinstance(skip_type_key, str) for skip_type_key in skip_type_keys
        ])
        self._skip_type_keys = skip_type_keys
