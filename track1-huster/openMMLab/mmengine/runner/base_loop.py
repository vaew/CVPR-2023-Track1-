# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Union

from torch.utils.data import DataLoader

from mmengine.dataset import InfiniteSampler

class MultiTaskDataLoader(object):
    '''by ZeLun Zhang
       多任务模式下对每个任务的 DataLoader 的封装
    '''
    def __init__(self, task2dataloader:dict):
        '''task2dataloader: Dict[str: DataLoader], 这里要求每个dataloader的sampler需要是 InfiniteSampler
        '''
        for task, dataloader in task2dataloader:
            if not isinstance(dataloader, DataLoader):
                raise ValueError(f'dataloader in task2dataloader should be DataLoader instance.')
            if not isinstance(dataloader.sampler, InfiniteSampler):
                raise ValueError(f'Each dataloader of {self.__class__.__name__} should use InfiniteSampler')

        self.task2dataloader = {task: iter(dataloader) for task, dataloader in task2dataloader.items()}

    def __iter__(self): # 每个dataloader都是infinity，直接返回自己作为迭代器对象
        return self

    def __next__(self):
        '''加载数据格式 {'det': det_dataloader_rst, 'seg': seg_dataloader_rst, 'cls': cls_dataloader_rst}
        '''
        rst_batch = {}
        for task in self.task2dataloader:
            rst_batch[task] = next(self.task2dataloader[task])
        return rst_batch


class BaseLoop_deprecated(metaclass=ABCMeta):
    """by ZeLun Zhang
       可以建立multi-task dataloader
    """

    def __init__(self, runner, dataloader: Union[DataLoader, Dict]) -> None:
        self._runner = runner
        if isinstance(dataloader, dict):
            # Determine whether or not different ranks use different seed.
            diff_rank_seed = runner._randomness_cfg.get(
                'diff_rank_seed', False)

            if dataloader.get('dataset', None) is not None: # fields中有dataset，表明单任务dataloader
                self.dataloader = runner.build_dataloader(
                    dataloader, seed=runner.seed, diff_rank_seed=diff_rank_seed)
            else: # 多任务dataloader
                task2dataloader = {}
                for task, dataloader_cfg in dataloader.items():
                    task2dataloader[task] = runner.build_dataloader(
                        dataloader_cfg, seed=runner.seed, diff_rank_seed=diff_rank_seed)
                self.dataloader = MuiltiTaskDataloader(task2dataloader)
        else:
            self.dataloader = dataloader

    @property
    def runner(self):
        return self._runner

    @abstractmethod
    def run(self) -> Any:
        """Execute loop."""



class BaseLoop(metaclass=ABCMeta):
    """Base loop class.

    All subclasses inherited from ``BaseLoop`` should overwrite the
    :meth:`run` method.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): An iterator to generate one batch of
            dataset each iteration.
    """

    def __init__(self, runner, dataloader: Union[DataLoader, Dict]) -> None:
        self._runner = runner
        if isinstance(dataloader, dict):
            # Determine whether or not different ranks use different seed.
            diff_rank_seed = runner._randomness_cfg.get(
                'diff_rank_seed', False)
            self.dataloader = runner.build_dataloader(
                dataloader, seed=runner.seed, diff_rank_seed=diff_rank_seed)
        else:
            self.dataloader = dataloader

    @property
    def runner(self):
        return self._runner

    @abstractmethod
    def run(self) -> Any:
        """Execute loop."""