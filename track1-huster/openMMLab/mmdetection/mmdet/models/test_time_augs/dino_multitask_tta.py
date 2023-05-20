# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch
from mmcv.ops import batched_nms
from mmengine.model import BaseTTAModel
from mmengine.registry import MODELS
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.structures import DetDataSample
from mmdet.structures.bbox import bbox_flip, bbox_mapping_back

from mmengine.structures import PixelData
from mmseg.structures import SegDataSample
from mmseg.utils import SampleList
import skimage
import numpy as np

from mmpretrain.structures import DataSample, data_sample

@MODELS.register_module()
class DINOMultiTaskTTAModel(BaseTTAModel):
    """Merge augmented detection results, only bboxes corresponding score under
    flipping and multi-scale resizing can be processed now.

    Examples:
        >>> tta_model = dict(
        >>>     type='DetTTAModel',
        >>>     tta_cfg=dict(nms=dict(
        >>>                     type='nms',
        >>>                     iou_threshold=0.5),
        >>>                     max_per_img=100))
        >>>
        >>> tta_pipeline = [
        >>>     dict(type='LoadImageFromFile',
        >>>          backend_args=None),
        >>>     dict(
        >>>         type='TestTimeAug',
        >>>         transforms=[[
        >>>             dict(type='Resize',
        >>>                  scale=(1333, 800),
        >>>                  keep_ratio=True),
        >>>         ], [
        >>>             dict(type='RandomFlip', prob=1.),
        >>>             dict(type='RandomFlip', prob=0.)
        >>>         ], [
        >>>             dict(
        >>>                 type='PackDetInputs',
        >>>                 meta_keys=('img_id', 'img_path', 'ori_shape',
        >>>                         'img_shape', 'scale_factor', 'flip',
        >>>                         'flip_direction'))
        >>>         ]])]
    """

    def __init__(self, det_tta_cfg=None, **kwargs):
        super().__init__(**kwargs)
        self.det_tta_cfg = det_tta_cfg

    def merge_aug_bboxes(self, aug_bboxes: List[Tensor],
                         aug_scores: List[Tensor],
                         img_metas: List[str]) -> Tuple[Tensor, Tensor]:
        """Merge augmented detection bboxes and scores.

        Args:
            aug_bboxes (list[Tensor]): shape (n, 4*#class)
            aug_scores (list[Tensor] or None): shape (n, #class)
        Returns:
            tuple[Tensor]: ``bboxes`` with shape (n,4), where
            4 represent (tl_x, tl_y, br_x, br_y)
            and ``scores`` with shape (n,).
        """
        detr_tta=False
        iou_thresh=0.5
        if detr_tta:
            aug0_bboxes=aug_bboxes[0]
            aug0_img_info=img_metas[0]
            aug0_score=aug_scores[0]

            recovered_bboxes = []
            for bboxes, img_info in zip(aug_bboxes[1:], img_metas[1:]):
                # from IPython import embed;embed()
                # 是否需要做FixedOffsetCrop还原
                offset = True if 'offset' in img_info.keys() else False
                
                # Flip还原
                if offset:
                    ori_shape = img_info['offset_img_shape']
                else:
                    ori_shape = img_info['ori_shape']
                flip = img_info['flip']
                flip_direction = img_info['flip_direction']
                if flip:
                    bboxes = bbox_flip(
                        bboxes=bboxes,
                        img_shape=ori_shape,
                        direction=flip_direction)
                
                if offset:
                    offset_pixel=img_info['offset']
                    bboxes[:,0::2] += offset_pixel[1]
                    bboxes[:,1::2] += offset_pixel[0]
                        
                recovered_bboxes.append(bboxes)
            
            # from IPython import embed;embed()
            for i in range(aug0_bboxes.shape[0]):
                num=1
                area_i = (aug0_bboxes[i,2] - aug0_bboxes[i,0] + 1) * (aug0_bboxes[i,3] - aug0_bboxes[i,1] + 1)
                for j in range(len(recovered_bboxes)):
                    xx1=torch.maximum(aug0_bboxes[i,0],recovered_bboxes[j][:,0])
                    yy1=torch.maximum(aug0_bboxes[i,1],recovered_bboxes[j][:,1])
                    xx2=torch.minimum(aug0_bboxes[i,2],recovered_bboxes[j][:,2])
                    yy2=torch.minimum(aug0_bboxes[i,3],recovered_bboxes[j][:,3])
                    w = torch.maximum(torch.zeros(1).type_as(xx1), xx2 - xx1 + 1)
                    h = torch.maximum(torch.zeros(1).type_as(xx1), yy2 - yy1 + 1)
                    area_j=(recovered_bboxes[j][:,2] - recovered_bboxes[j][:,0] + 1) * (recovered_bboxes[j][:,3] - recovered_bboxes[j][:,1] + 1)
                    over = (w * h) / (area_i + area_j - w * h)
                    index = torch.where(over >= iou_thresh)[0]
                    if index.shape[0]>0:
                        num+=1
                        aug0_bboxes[i]+=recovered_bboxes[j][torch.argmax(over)]
                aug0_bboxes[i]=aug0_bboxes[i]/num

            recovered_bboxes = [aug0_bboxes]
        
        else:
            recovered_bboxes = []
            for bboxes, img_info in zip(aug_bboxes, img_metas):
                # from IPython import embed;embed()
                # 是否需要做FixedOffsetCrop还原
                offset = True if 'offset' in img_info.keys() else False
                
                # Flip还原
                if offset:
                    ori_shape = img_info['offset_img_shape']
                else:
                    ori_shape = img_info['ori_shape']
                flip = img_info['flip']
                flip_direction = img_info['flip_direction']
                if flip:
                    bboxes = bbox_flip(
                        bboxes=bboxes,
                        img_shape=ori_shape,
                        direction=flip_direction)
                
                if offset:
                    offset_pixel=img_info['offset']
                    bboxes[:,0::2] += offset_pixel[1]
                    bboxes[:,1::2] += offset_pixel[0]
                        
                recovered_bboxes.append(bboxes)
        bboxes = torch.cat(recovered_bboxes, dim=0)
        if aug_scores is None:
            return bboxes
        else:
            if detr_tta:
                scores = aug0_score
            else:
                scores = torch.cat(aug_scores, dim=0)
            return bboxes, scores

    def merge_preds(self, data_samples_list: List[List[DetDataSample]]):
        """Merge batch predictions of enhanced data.

        Args:
            data_samples_list (List[List[DetDataSample]]): List of predictions
                of all enhanced data. The outer list indicates images, and the
                inner list corresponds to the different views of one image.
                Each element of the inner list is a ``DetDataSample``.
        Returns:
            List[DetDataSample]: Merged batch prediction.
        """
        merged_data_samples = []
        for data_samples in data_samples_list:
            merged_data_samples.append(self._merge_single_sample(data_samples))
        return merged_data_samples

    def det_merge_preds(self, data_samples_list: List[List[DetDataSample]]):
        """Merge batch predictions of enhanced data.

        Args:
            data_samples_list (List[List[DetDataSample]]): List of predictions
                of all enhanced data. The outer list indicates images, and the
                inner list corresponds to the different views of one image.
                Each element of the inner list is a ``DetDataSample``.
        Returns:
            List[DetDataSample]: Merged batch prediction.
        """
        merged_data_samples = []
        for data_samples in data_samples_list:
            merged_data_samples.append(self._det_merge_single_sample(data_samples))
        return {'det':merged_data_samples}
    
    def _det_merge_single_sample(
            self, data_samples: List[DetDataSample]) -> DetDataSample:
        """Merge predictions which come form the different views of one image
        to one prediction.

        Args:
            data_samples (List[DetDataSample]): List of predictions
            of enhanced data which come form one image.
        Returns:
            List[DetDataSample]: Merged prediction.
        """
        aug_bboxes = []
        aug_scores = []
        aug_labels = []
        img_metas = []
        # TODO: support instance segmentation TTA
        assert data_samples[0].pred_instances.get('masks', None) is None, \
            'TTA of instance segmentation does not support now.'
        # from IPython import embed;embed()
        for data_sample in data_samples:
            aug_bboxes.append(data_sample.pred_instances.bboxes)
            aug_scores.append(data_sample.pred_instances.scores)
            aug_labels.append(data_sample.pred_instances.labels)
            img_metas.append(data_sample.metainfo)

        merged_bboxes, merged_scores = self.merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas)
        merged_labels = torch.cat(aug_labels, dim=0)
        # merged_labels = data_samples[0].pred_instances.labels

        if merged_bboxes.numel() == 0:
            return data_samples[0]

        det_bboxes, keep_idxs = batched_nms(merged_bboxes, merged_scores,
                                            merged_labels, self.det_tta_cfg.nms)

        det_bboxes = det_bboxes[:self.det_tta_cfg.max_per_img]
        det_labels = merged_labels[keep_idxs][:self.det_tta_cfg.max_per_img]

        results = InstanceData()
        _det_bboxes = det_bboxes.clone()
        results.bboxes = _det_bboxes[:, :-1]
        results.scores = _det_bboxes[:, -1]
        results.labels = det_labels
        det_results = data_samples[0]
        det_results.pred_instances = results
        return det_results

    def seg_merge_preds(self, data_samples_list: List[SampleList]) -> SampleList:
        """Merge predictions of enhanced data to one prediction.

        Args:
            data_samples_list (List[SampleList]): List of predictions
                of all enhanced data.

        Returns:
            SampleList: Merged prediction.
        """
        # from IPython import embed;embed()
        predictions = []
        for data_samples in data_samples_list:
            seg_logits = data_samples[0].seg_logits.data
            logits = torch.zeros(seg_logits.shape).to(seg_logits)
            for data_sample in data_samples:
                seg_logit = data_sample.seg_logits.data
                if self.module.seg_head.out_channels > 1:
                    logits += seg_logit.softmax(dim=0)
                else:
                    logits += seg_logit.sigmoid()
            logits /= len(data_samples)
            if self.module.seg_head.out_channels == 1:
                seg_pred = (logits > self.module.decode_head.threshold
                            ).to(logits).squeeze(1)
            else:
                seg_pred = logits.argmax(dim=0)

                #seg_pred = seg_pred.cpu().numpy().astype(np.uint8)[:,:,None].repeat(3,axis=-1)
                ## 删除图像中小物体
                #seg_pred=skimage.morphology.remove_small_objects(seg_pred, min_size=1024, connectivity=1)
                ## # min_size: 最小连通区域尺寸，小于该尺寸的都将被删除。默认为64.
                ## # connectivity: 邻接模式，1表示4邻接(up,down,left,right)，2表示8邻接(up_left,up_right,down_left,down_right)
                #seg_pred = torch.from_numpy(seg_pred[:,:,0]).to(logits)

            data_sample = SegDataSample(
                **{
                    'pred_sem_seg': PixelData(data=seg_pred),
                    'gt_sem_seg': data_samples[0].gt_sem_seg,
                    'img_path': data_sample.img_path
                })
            predictions.append(data_sample)
        return {'seg':predictions}
    
    def cls_merge_preds(
        self,
        data_samples_list: List[List[DataSample]],
    ) -> List[DataSample]:
        """Merge predictions of enhanced data to one prediction.

        Args:
            data_samples_list (List[List[DataSample]]): List of predictions
                of all enhanced data.

        Returns:
            List[DataSample]: Merged prediction.
        """
        # from IPython import embed;embed()
        merged_data_samples = []
        for data_samples in data_samples_list:
            merged_data_samples.append(self._cls_merge_single_sample(data_samples))
        return {'cls':merged_data_samples}

    def _cls_merge_single_sample(self, data_samples):
        merged_data_sample: DataSample = data_samples[0].new()
        merged_score = sum(data_sample.pred_score
                           for data_sample in data_samples) / len(data_samples)
        merged_data_sample.set_pred_score(merged_score)
        #merged_data_sample.set_data({'img_path': data_sample.img_path})
        return merged_data_sample
