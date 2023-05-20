# Copyright (c) OpenMMLab. All rights reserved.
from .base_dataset import BaseDataset, Compose, force_full_init
from .dataset_wrapper import ClassBalancedDataset, ConcatDataset, RepeatDataset
from .sampler import DefaultSampler, InfiniteSampler
from .utils import (COLLATE_FUNCTIONS, default_collate, pseudo_collate,
                    worker_init_fn)

__all__ = [
    'BaseDataset', 'Compose', 'force_full_init', 'ClassBalancedDataset',
    'ConcatDataset', 'RepeatDataset', 'DefaultSampler', 'InfiniteSampler',
    'worker_init_fn', 'pseudo_collate', 'COLLATE_FUNCTIONS', 'default_collate'
]

from torch.utils.data import DataLoader

class MultiTaskDataLoader(object):
    '''by ZeLun Zhang
       多任务模式下对每个任务的 DataLoader 的封装, 用在train
    '''
    def __init__(self, task2dataloader:dict):
        '''task2dataloader: Dict[str: DataLoader], 这里要求每个dataloader的sampler需要是 InfiniteSampler
        '''
        for task, dataloader in task2dataloader.items():
            if not isinstance(dataloader, DataLoader):
                raise ValueError(f'dataloader in task2dataloader should be DataLoader instance.')
            #if not isinstance(dataloader.sampler, InfiniteSampler):
            #    raise ValueError(f'Each dataloader of {self.__class__.__name__} should use InfiniteSampler, but got {dataloader.sampler.__class__.__name__} instead.')

        self.src_task2dataloader = task2dataloader
        self.task2dataloader = {task: iter(dataloader) for task, dataloader in task2dataloader.items()}

        for task in self.src_task2dataloader:
            self.dataset = self.src_task2dataloader[task].dataset # 仅仅是为了有metainfo，随便弄个dataset
            break

    def __iter__(self): # 每个dataloader都是infinity，直接返回自己作为迭代器对象
        return self

    def __next__(self):
        '''加载数据格式 {'det': det_dataloader_rst, 'seg': seg_dataloader_rst, 'cls': cls_dataloader_rst}
        '''
        rst_batch = {}
        for task in self.task2dataloader:
            try:
                rst_batch[task] = next(self.task2dataloader[task]) # 每个dataloader都是无限的
            except StopIteration:
                self.task2dataloader[task] = iter(self.src_task2dataloader[task])
                rst_batch[task] = next(self.task2dataloader[task])
        return rst_batch

    def __len__(self):
        return 2000


class MultiTaskDataLoaderVal(object):
    '''by ZeLun Zhang
       多任务模式下对每个任务的 DataLoader 的封装, 用在val
       其会按照顺序依次从每个dataloader中加载数据
    '''
    def __init__(self, task2dataloader:dict):
        '''task2dataloader: Dict[str: DataLoader], 这里要求每个dataloader的sampler需要是 InfiniteSampler
        '''
        for task, dataloader in task2dataloader.items():
            if not isinstance(dataloader, DataLoader):
                raise ValueError(f'dataloader in task2dataloader should be DataLoader instance.')
            if not isinstance(dataloader.sampler, DefaultSampler):
                raise ValueError(f'Each dataloader of {self.__class__.__name__} should use DefaultSampler, but got {dataloader.sampler.__class__.__name__} instead.')

        # val只会进行一轮
        self.src_task2dataloader = task2dataloader
        self.task2dataloader = {task: iter(dataloader) for task, dataloader in self.src_task2dataloader.items()}

        self.task_lst = list(self.task2dataloader.keys())
        self.current_task_idx = 0

        data_num = 0
        for task in self.src_task2dataloader:
            data_num += len(self.src_task2dataloader[task].dataset)
        self.data_num = data_num

    def __iter__(self): # 每个dataloader都是infinity，直接返回自己作为迭代器对象
        self.current_task_idx = 0
        self.task2dataloader = {task: iter(dataloader) for task, dataloader in self.src_task2dataloader.items()}
        return self

    def __next__(self):
        '''加载数据格式 {'det': det_dataloader_rst} or {'seg': seg_dataloader_rst} or {'cls': cls_dataloader_rst}
           每次只会加载一种任务的数据来测试
        '''
        rst_batch = {}
        try:
            current_task = self.task_lst[self.current_task_idx]
            rst_batch[current_task] = next(self.task2dataloader[current_task])
        except StopIteration:
            self.current_task_idx += 1
            if self.current_task_idx == len(self.task_lst):
                raise
            current_task = self.task_lst[self.current_task_idx]
            rst_batch[current_task] = next(self.task2dataloader[current_task])
        return rst_batch

    def __len__(self):
        return self.data_num