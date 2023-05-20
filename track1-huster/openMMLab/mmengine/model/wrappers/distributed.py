# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, Union

import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel

from mmengine.optim import OptimWrapper
from mmengine.registry import MODEL_WRAPPERS
from ..utils import detect_anomalous_params
from typing import List
from mmengine.logging import MessageHub, print_log

MODEL_WRAPPERS.register_module(module=DistributedDataParallel)
MODEL_WRAPPERS.register_module(module=DataParallel)


@MODEL_WRAPPERS.register_module()
class MMDistributedDataParallel(DistributedDataParallel):
    """A distributed model wrapper used for training,testing and validation in
    loop.

    Different from DistributedDataParallel, MMDistributedDataParallel
    implements three methods :meth:`train_step`, :meth:`val_step` and
    :meth:`test_step`, which will be called by ``train_loop``, ``val_loop``
    and ``test_loop``.

    - ``train_step``: Called by ``runner.train_loop``, and implement
      default model forward, gradient back propagation, parameter updating
      logic. To take advantage of DistributedDataParallel's automatic gradient
      synchronization, ``train_step`` calls ``DistributedDataParallel.forward``
      to calculate the losses, and call other methods of :class:`BaseModel` to
      pre-process data and parse losses. Finally, update model parameters by
      :class:`OptimWrapper` and return the loss dictionary used
      for logging.

    - ``val_step``: Called by ``runner.val_loop`` and get the inference
      results. Since there is no gradient synchronization requirement,
      this procedure is equivalent to ``BaseModel.val_step``

    - ``test_step``: Called by ``runner.test_loop``, equivalent ``val_step``.

    Args:
        detect_anomalous_params (bool): This option is only used for
            debugging which will slow down the training speed.
            Detect anomalous parameters that are not included in
            the computational graph with `loss` as the root.
            There are two cases

            - Parameters were not used during forward pass.
            - Parameters were not used to produce loss.

            Defaults to False.

        **kwargs: keyword arguments passed to ``DistributedDataParallel``.

            - device_ids (List[int] or torch.device, optional): CUDA devices
              for module.
            - output_device (int or torch.device, optional): Device location of
              output for single-device CUDA modules.
            - dim (int): Defaults to 0.
            - broadcast_buffers (bool): Flag that enables syncing (
              broadcasting) buffers of the module at beginning of the
              ``forward`` function. Defaults to True
            - find_unused_parameters (bool): Whether to find parameters of
              module, which are not in the forward graph. Defaults to False.
            - process_group (ProcessGroup, optional): The process group to be
              used for distributed data all-reduction.
            - bucket_cap_mb (int): bucket size in MegaBytes (MB). Defaults
              to 25.
            - check_reduction (bool): This argument is deprecated. Defaults
              to False.
            - gradient_as_bucket_view (bool): Defaults to False.
            - static_graph (bool): Defaults to False.

    See more information about arguments in
    :class:`torch.nn.parallel.DistributedDataParallel`.

    Note:
        If model has multiple submodules and each module has
        separate optimization strategies,
        :class:`MMSeparateDistributedDataParallel` should be used to wrap
        the model.

    Note:
        If model itself has custom optimization strategy, rather than
        simply forward model and update model. A custom model wrapper
        inherit from ``MMDistributedDataParallel`` should be defined and
        override the ``train_step`` method.
    """

    def __init__(self,
                 module,
                 detect_anomalous_params: bool = False,
                 **kwargs):
        super().__init__(module=module, **kwargs)
        self.detect_anomalous_params = detect_anomalous_params

    def train_step(self, data: Union[dict, tuple, list], # data就是直接从dataloader中加载的结果
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        """Implements the default model training process including
        preprocessing, model forward propagation, loss calculation,
        optimization, and back-propagation.

        During non-distributed training. If subclasses do not override the
        :meth:`train_step`, :class:`EpochBasedTrainLoop` or
        :class:`IterBasedTrainLoop` will call this method to update model
        parameters. The default parameter update process is as follows:

        1. Calls ``self.data_processor(data, training=False)`` to collect
           batch_inputs and corresponding data_samples(labels).
        2. Calls ``self(batch_inputs, data_samples, mode='loss')`` to get raw
           loss
        3. Calls ``self.parse_losses`` to get ``parsed_losses`` tensor used to
           backward and dict of loss tensor used to log messages.
        4. Calls ``optim_wrapper.update_params(loss)`` to update model.

        Args:
            data (dict or tuple or list): Data sampled from dataset.
            optim_wrapper (OptimWrapper): OptimWrapper instance
                used to update model parameters.

        Returns:
            Dict[str, torch.Tensor]: A ``dict`` of tensor for logging.
        """
        rst_log_vars = {}
        record_module_grad = [torch.zeros_like(param) for param in self.module.parameters()] # 记录本次iter模型更新参数时用的grad
        for task in data: # 依次对一个batch中每个task_batch单独计算
            task_data = {task: data[task]}
            # Enable automatic mixed precision training context.
            with optim_wrapper.optim_context(self):
                task_data = self.module.data_preprocessor(task_data, True) # 输入的data就是直接从dataloader中加载的结果
                # 输出变成 {'det': 原det PP输出结果, 'cls': 原cls PP输出结果, 'seg':原seg PP输出结果 }
                losses = self._run_forward(task_data, mode='loss')  # 对单个任务计算的损失

            parsed_losses, log_vars = self.module.parse_losses(losses) # 解析出需要backward的loss, 最后输出parsed_losses作为整体loss, 最后的backward全部依赖这个
            # parsed_losses: 单任务用来backward的loss
            # log_vars: 日志用，其中有 'loss' 需要整体更新
            if len(rst_log_vars) == 0:
                rst_log_vars.update(log_vars)
            else:
                rst_log_vars['loss'] += log_vars['loss']
                del log_vars['loss']
                rst_log_vars.update(log_vars)

            # 在这里backward但不更新参数
            parsed_losses = optim_wrapper.scale_loss(parsed_losses)
            optim_wrapper.backward(parsed_losses) # 释放单个任务计算图，计算单个任务的梯度，此时已存储到param.grad中
            #self._calc_backbone_grad_norm(f'{task}-before-clip')

            if optim_wrapper.clip_grad_kwargs:
                optim_wrapper._clip_grad(task = task, backbone_based = True, model = self) # TODO 基于backbone的梯度裁剪，缩放整个任务的grad，直到这个任务传给backbone的grad norm不大于一个scalar
                #self._calc_backbone_grad_norm(f'{task}-after-clip')
                for i, param in zip(record_module_grad, self.module.parameters()): # 累积这个任务梯度裁剪的结果
                    if param.grad is None: # 这部分没有参与训练
                        continue
                    i += param.grad
                optim_wrapper.zero_grad()

        for record_grad, param in zip(record_module_grad, self.module.parameters()):
            param.grad = record_grad

        #self._calc_backbone_grad_norm(f'all-before-step')
        optim_wrapper.optimizer.step()
        optim_wrapper.zero_grad()

        return rst_log_vars

    def train_step_deprecated2(self, data: Union[dict, tuple, list], # data就是直接从dataloader中加载的结果
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        """Implements the default model training process including
        preprocessing, model forward propagation, loss calculation,
        optimization, and back-propagation.

        During non-distributed training. If subclasses do not override the
        :meth:`train_step`, :class:`EpochBasedTrainLoop` or
        :class:`IterBasedTrainLoop` will call this method to update model
        parameters. The default parameter update process is as follows:

        1. Calls ``self.data_processor(data, training=False)`` to collect
           batch_inputs and corresponding data_samples(labels).
        2. Calls ``self(batch_inputs, data_samples, mode='loss')`` to get raw
           loss
        3. Calls ``self.parse_losses`` to get ``parsed_losses`` tensor used to
           backward and dict of loss tensor used to log messages.
        4. Calls ``optim_wrapper.update_params(loss)`` to update model.

        Args:
            data (dict or tuple or list): Data sampled from dataset.
            optim_wrapper (OptimWrapper): OptimWrapper instance
                used to update model parameters.

        Returns:
            Dict[str, torch.Tensor]: A ``dict`` of tensor for logging.
        """
        rst_log_vars = {}
        record_module_grad = [torch.zeros_like(param) for param in self.module.parameters()] # 记录本次iter模型更新参数时用的grad
        for task in data: # 依次对一个batch中每个task_batch单独计算
            task_data = {task: data[task]}
            # Enable automatic mixed precision training context.
            with optim_wrapper.optim_context(self):
                task_data = self.module.data_preprocessor(task_data, True) # 输入的data就是直接从dataloader中加载的结果
                # 输出变成 {'det': 原det PP输出结果, 'cls': 原cls PP输出结果, 'seg':原seg PP输出结果 }
                losses = self._run_forward(task_data, mode='loss')  # 对单个任务计算的损失

            parsed_losses, log_vars = self.module.parse_losses(losses) # 解析出需要backward的loss, 最后输出parsed_losses作为整体loss, 最后的backward全部依赖这个
            # parsed_losses: 单任务用来backward的loss
            # log_vars: 日志用，其中有 'loss' 需要整体更新
            if len(rst_log_vars) == 0:
                rst_log_vars.update(log_vars)
            else:
                rst_log_vars['loss'] += log_vars['loss']
                del log_vars['loss']
                rst_log_vars.update(log_vars)

            # 在这里backward但不更新参数
            parsed_losses = optim_wrapper.scale_loss(parsed_losses)
            optim_wrapper.backward(parsed_losses) # 释放单个任务计算图，计算单个任务的梯度，此时已存储到param.grad中
            #self._calc_backbone_grad_norm(f'{task}-before-clip')

            if optim_wrapper.clip_grad_kwargs:
                optim_wrapper._clip_grad(task) # 对单个任务单独梯度裁剪
                #self._calc_backbone_grad_norm(f'{task}-after-clip')
                for i, param in zip(record_module_grad, self.module.parameters()): # 累积这个任务梯度裁剪的结果
                    if param.grad is None: # 这部分没有参与训练
                        continue
                    i += param.grad
                optim_wrapper.zero_grad()

        for record_grad, param in zip(record_module_grad, self.module.parameters()):
            param.grad = record_grad

        #self._calc_backbone_grad_norm(f'all-before-step')
        optim_wrapper.optimizer.step()
        optim_wrapper.zero_grad()

        return rst_log_vars

    def _calc_backbone_grad_norm(self, prefix_str = ''):
        '''计算backbone的grad范数，看表现是否符合预期
        '''
        backbone_params = []
        for name, param in self.module.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
        backbone_params = list(filter(lambda p: (p.requires_grad) and (p.grad is not None), backbone_params))
        backbone_grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in backbone_params]), 2)
        print(f'{prefix_str}-backbone-grad-norm:', backbone_grad_norm)



    def _calc_backbone_task_grad_norm(self, new_iter, task):
        '''计算单次iter中回传到backbone
           new_iter: 当前iter是否为这个iter计算的第一个task
           task: 当前计算的task名
        '''
        backbone_params = []
        for name, param in self.module.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
        backbone_params = list(filter(lambda p: (p.requires_grad) and (p.grad is not None), backbone_params))
        # 选出已经计算出梯度的param

        if new_iter: # 用于记录之前任务计算的梯度
            self.record_backbone_grad = [torch.zeros_like(param) for param in backbone_params]

        cur_backbone_grad = [param.grad.detach() for param in backbone_params]
        task_backbone_grad = [(i - j) for i,j in zip(cur_backbone_grad, self.record_backbone_grad)] # 当前task对backbone产生的梯度

        task_backbone_grad_norm = torch.norm(torch.stack([torch.norm(p, 2) for p in task_backbone_grad]), 2)
        message_hub = MessageHub.get_current_instance()
        message_hub.update_scalar(f'{task}-backbone-norm', float(task_backbone_grad_norm))
        print(f'{task}-backbone-norm:', task_backbone_grad_norm)

        self.record_backbone_grad = [(i+j) for i,j in zip(self.record_backbone_grad, task_backbone_grad)]


    def train_step_deprecated1(self, data: Union[dict, tuple, list], # 每次完成一个任务的计算，对这个任务backward，释放计算图，其梯度累积在param.grad中，最后一起更新
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        """Implements the default model training process including
        preprocessing, model forward propagation, loss calculation,
        optimization, and back-propagation.

        During non-distributed training. If subclasses do not override the
        :meth:`train_step`, :class:`EpochBasedTrainLoop` or
        :class:`IterBasedTrainLoop` will call this method to update model
        parameters. The default parameter update process is as follows:

        1. Calls ``self.data_processor(data, training=False)`` to collect
           batch_inputs and corresponding data_samples(labels).
        2. Calls ``self(batch_inputs, data_samples, mode='loss')`` to get raw
           loss
        3. Calls ``self.parse_losses`` to get ``parsed_losses`` tensor used to
           backward and dict of loss tensor used to log messages.
        4. Calls ``optim_wrapper.update_params(loss)`` to update model.

        Args:
            data (dict or tuple or list): Data sampled from dataset.
            optim_wrapper (OptimWrapper): OptimWrapper instance
                used to update model parameters.

        Returns:
            Dict[str, torch.Tensor]: A ``dict`` of tensor for logging.
        """
        rst_log_vars = {}
        new_iter = True
        for task in data: # 依次对一个batch中每个task_batch单独计算
            task_data = {task: data[task]}
            # Enable automatic mixed precision training context.
            with optim_wrapper.optim_context(self):
                task_data = self.module.data_preprocessor(task_data, True) # 输入的data就是直接从dataloader中加载的结果
                # 输出变成 {'det': 原det PP输出结果, 'cls': 原cls PP输出结果, 'seg':原seg PP输出结果 }
                losses = self._run_forward(task_data, mode='loss')  # 对单个任务计算的损失

            parsed_losses, log_vars = self.module.parse_losses(losses) # 解析出需要backward的loss, 最后输出parsed_losses作为整体loss, 最后的backward全部依赖这个
            # parsed_losses: 单任务用来backward的loss
            # log_vars: 日志用，其中有 'loss' 需要整体更新
            if len(rst_log_vars) == 0:
                rst_log_vars.update(log_vars)
            else:
                rst_log_vars['loss'] += log_vars['loss']
                del log_vars['loss']
                rst_log_vars.update(log_vars)

            # 在这里backward但不更新参数
            parsed_losses = optim_wrapper.scale_loss(parsed_losses)
            optim_wrapper.backward(parsed_losses) # 释放单个任务计算图
            self._calc_backbone_task_grad_norm(new_iter, task)
            new_iter = False

        optim_wrapper.step()
        optim_wrapper.zero_grad()
        
        return rst_log_vars

    def train_step_deprecated0(self, data: Union[dict, tuple, list],
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]: # 最原始的官方的train_step，保存三个任务的计算图，然后一起backward更新参数
        """Interface for model forward, backward and parameters updating during
        training process.

        :meth:`train_step` will perform the following steps in order:

        - If :attr:`module` defines the preprocess method,
          call ``module.preprocess`` to pre-processing data.
        - Call ``module.forward(**data)`` and get losses.
        - Parse losses.
        - Call ``optim_wrapper.optimizer_step`` to update parameters.
        - Return log messages of losses.

        Args:
            data (dict or tuple or list): Data sampled from dataset.
            optim_wrapper (OptimWrapper): A wrapper of optimizer to
                update parameters.

        Returns:
            Dict[str, torch.Tensor]: A ``dict`` of tensor for logging.
        """
        # Enable automatic mixed precision training context.
        with optim_wrapper.optim_context(self):
            data = self.module.data_preprocessor(data, training=True)
            losses = self._run_forward(data, mode='loss')
        parsed_loss, log_vars = self.module.parse_losses(losses)
        optim_wrapper.update_params(parsed_loss)
        if self.detect_anomalous_params:
            detect_anomalous_params(parsed_loss, model=self)
        return log_vars

    def val_step(self, data: Union[dict, tuple, list]) -> list:
        """Gets the prediction of module during validation process.

        Args:
            data (dict or tuple or list): Data sampled from dataset.

        Returns:
            list: The predictions of given data.
        """
        return self.module.val_step(data)

    def test_step(self, data: Union[dict, tuple, list]) -> list:
        """Gets the predictions of module during testing process.

        Args:
            data (dict or tuple or list): Data sampled from dataset.

        Returns:
            list: The predictions of given data.
        """
        return self.module.test_step(data)

    def _run_forward(self, data: Union[dict, tuple, list], mode: str) -> Any:
        """Unpacks data for :meth:`forward`

        Args:
            data (dict or tuple or list): Data sampled from dataset.
            mode (str): Mode of forward.

        Returns:
            dict or list: Results of training or testing mode.
        """
        if isinstance(data, dict):
            results = self(**data, mode=mode)
        elif isinstance(data, (list, tuple)):
            results = self(*data, mode=mode)
        else:
            raise TypeError('Output of `data_preprocessor` should be '
                            f'list, tuple or dict, but got {type(data)}')
        return results
