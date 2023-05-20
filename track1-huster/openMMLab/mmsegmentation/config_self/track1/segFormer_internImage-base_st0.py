dataset_type = 'mmseg.Track1SegDataset'
default_scope = 'mmseg'
train_multiscale_pipeline = [
    dict(
        type='RandomResize', # 从ratio_range中采样ratio -> ratio * scale = dst_scale -> 用 dst_scale 对图像 rescale, 如果keep_ratio，先把短边缩放到短边，再保证长边不超过长的
        scale=(1820, 1024), # (w, h)的形式, 设置预测分辨率1024
        ratio_range=(0.5, 2.0), # (910, 512) ~ (1820, 1024) ~ (1820, 1024)*2
        keep_ratio=True, # 先对图像做整体的缩放
    ),
    dict(
        type='mmseg.RandomCrop', # crop出 crop_size 的patch，形状定死, 如果 图像尺寸 < crop_size，则直接返回原图
        crop_size=(1024, 1024), # (h, w)形式，
        cat_max_ratio=0.75, # crop单个类别在crop_img中的最高比例
    ),
    dict(type='RandomFlip', prob=0.5),
    dict(type='mmseg.PhotoMetricDistortion'), # brightness, contractness, saturation 调节
    dict(
        type='RandomApply',
        transforms = dict(type='mmseg.RGB2Gray'),
        prob = 0.1,
    ),
    dict(type='mmseg.RandomRotate', prob = 0.2, degree = 15),
    dict(
        type='mmpretrain.Albu',
        transforms = [
            dict(
	            type='OneOf',
	                transforms=[
		                 dict(type='GaussNoise', var_limit=(0.0,50.0), p=1.0),
		                 dict(type='Blur', blur_limit=3, p=1.0),
		                 dict(type='Emboss', alpha=(0.2,0.5),strength=(0.2,0.7),p=0.5),
		                 dict(type='GaussianBlur', blur_limit=(3,5),sigma_limit=0,p=1),
		                 dict(type='MultiplicativeNoise', multiplier=(0.8, 1.2), elementwise=True,p=1.0),
	                ],
	            p=0.6
            ),
        ]
    ),
    dict(type='mmseg.PackSegInputs'),
]
train_mosaic_pipeline = [
    dict(
        type = 'mmseg.RandomMosaic',
        prob = 1.0,
        img_scale = (720, 1280), # 输出1420x2560, 如果设置1024x1024，进过mosaic会有大量空白
        center_ratio_range=(0.5, 1.5),
        pad_val=0,
        seg_pad_val=255,
    ),
    dict(
        type = 'RandomResize',
        scale = (720, 1280), # 基准1024, hw都有2张图，一张图基准512
        ratio_range = (0.5, 2.0), # 单张最大(720, 1280)，最小(180, 320), 缩小1/4
        keep_ratio = True,
    ),
    dict(
        type='mmseg.RandomCrop', # crop出 crop_size 的patch，形状定死, 如果 图像尺寸 < crop_size，则直接返回原图
        crop_size=(1024, 1024), # (h, w)形式，
        cat_max_ratio=0.75, # crop单个类别在crop_img中的最高比例
    ),
    dict(type='RandomFlip', prob=0.5),
    dict(type='mmseg.PhotoMetricDistortion'), # brightness, contractness, saturation 调节
    dict(
        type='mmpretrain.Albu',
        transforms = [
            dict(
	            type='OneOf',
	                transforms=[
		                 dict(type='GaussNoise', var_limit=(0.0,50.0), p=1.0),
		                 dict(type='Blur', blur_limit=3, p=1.0),
		                 dict(type='Emboss', alpha=(0.2,0.5),strength=(0.2,0.7),p=0.5),
		                 dict(type='GaussianBlur', blur_limit=(3,5),sigma_limit=0,p=1),
		                 dict(type='MultiplicativeNoise', multiplier=(0.8, 1.2), elementwise=True,p=1.0),
	                ],
	            p=0.6
            ),
        ]
    ),
    dict(type='mmseg.PackSegInputs'),
]
train_pipeline = [
    dict(
        type='RandomChoice',
        transforms = [
            train_mosaic_pipeline, # hard
            train_multiscale_pipeline, # easy
        ],
        prob = [0., 1.0],
    )
]
#train_pipeline = train_mosaic_pipeline

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1280, 720), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='mmseg.LoadAnnotations'),
    dict(type='mmseg.PackSegInputs'),
]
train_dataloader = dict(
    batch_size=20,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset = dict(
        type='mmseg.MultiImageMixDataset',
        dataset=dict(
            type=dataset_type,
            data_prefix=dict(
                img_path='/data/panxue/track1/track1_train_data/seg/images/train_and_val', 
                seg_map_path='/data/panxue/track1/track1_train_data/seg/label/train_and_val'
            ),
            img_suffix='.jpg',
            seg_map_suffix='.png',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='mmseg.LoadAnnotations'),
            ],
            serialize_data = False, # debug用
        ),
        pipeline=train_pipeline,
    )
)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_prefix=dict(
            img_path='/data/panxue/track1/track1_test_data/seg/images/test', 
            seg_map_path='/data/panxue/track1/track1_test_data/seg/images/test_label',
        ),
        img_suffix='.jpg',
        seg_map_suffix='.png',
        pipeline=test_pipeline,
        serialize_data = False, # debug
        ),
    )
test_dataloader = val_dataloader
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

#model settings
norm_cfg = dict(type='SyncBN', requires_grad=True) # SyncBN
data_preprocessor = dict(
    type='mmseg.SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0, # img的pad
    seg_pad_val=255, # label的pad
    size = (720, 1280), #(h, w)
    )
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='mmdet.InternImage',
        core_op='DCNv3',
        channels=112,
        depths=[4, 4, 21, 4],
        groups=[7, 14, 28, 56],
        mlp_ratio=4.,
        drop_path_rate=0.4,
        norm_layer='LN',
        layer_scale=1.0,
        offset_scale=1.0,
        post_norm=True,
        with_cp=False,
        out_indices=(0, 1, 2, 3),
        frozen_stages = 4, # 固定全部来自检测的参数，整个backbone的param全部freeze
        init_cfg=dict(
            type='Pretrained', 
            checkpoint='/data/zzl/work_dirs/dino_internimage-b_st2/epoch_80.pth',
            prefix = 'backbone',
            map_location = 'cpu'
        )
    ),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[112, 224, 448, 896],
        in_index=[0, 1, 2, 3],
        channels=256, # 输出的通道数
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
        ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

base_lr = 0.0001 # cfg里 var = val, 这样var会变成tuple，终究还是python
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=base_lr, betas=(0.9, 0.999), weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1), # 对backbone的差分学习率
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.) # 分割头lr*10
        }
    )
)

# learning policy
max_epochs = 100
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='LinearLR', 
        start_factor = 0.01, 
        end_factor = 1.0,
        begin=0,  
        end=5,
        by_epoch=True,  
        convert_to_iter_based=True,
    ), 
    dict(
        type='CosineAnnealingLR',
        T_max=95,
        eta_min=base_lr*1e-3,
        begin=5,
        end=100,
        by_epoch=True,
        convert_to_iter_based=True
    ),
]
#
# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (16 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(enable=True, base_batch_size=16) # 需要加上enable=True才自动打开

# runtime设置


default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50), #  默认50
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, by_epoch=True),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook')
    )

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=10, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False

fp16 = dict(loss_scale=512.)
work_dir = '/data/zzl/work_dirs/segformer_internimage-b_st0'

find_unused_parameters=True