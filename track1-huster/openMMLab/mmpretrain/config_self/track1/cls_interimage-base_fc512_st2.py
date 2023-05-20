# dataset settings
dataset_type = 'mmpretrain.CustomDataset'
default_scope = 'mmpretrain'
data_preprocessor = dict(
    type='mmpretrain.ClsDataPreprocessor',
    num_classes=196,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

bgr_mean = data_preprocessor['mean'][::-1]
bgr_std = data_preprocessor['std'][::-1]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='mmpretrain.RandomResizedCrop',
        crop_ratio_range = (0.9, 1.0), # 保证是原图大面积
        aspect_ratio_range = (3. / 4., 4. / 3.),
        scale=448,
        backend='cv2',
        interpolation='bilinear'
    ),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    #dict(
    #    type='mmpretrain.RandAugment',
    #    policies='timm_increasing',
    #    num_policies=2,
    #    total_level=10,
    #    magnitude_level=9,
    #    magnitude_std=0.5,
    #    hparams=dict(
    #        pad_val=[round(x) for x in bgr_mean], interpolation='bilinear')
    #),
    dict(
        type='RandomErasing',
        erase_prob=0.5,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=1 / 3,
        fill_color=bgr_mean,
        fill_std=bgr_std
    ),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale = (448, 448)),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=128,
    num_workers=6,
    dataset=dict(
        type=dataset_type,
        data_root = '/data/panxue/track1/track1_train_data/cls/train_and_val',
        ann_file='/data/panxue/track1/track1_train_data/cls/train_and_val-amend.txt',
        pipeline=train_pipeline,
        serialize_data = False, # debug用
        ),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=128,
    num_workers=6,
    dataset=dict(
        type=dataset_type,
        data_root = '/data/panxue/track1/track1_test_data/cls/test',
        ann_file='/data/panxue/track1/track1_train_data/cls/test/dataset-orginal-test-label.txt',
        pipeline=test_pipeline, 
        serialize_data = False, # debug用
        ),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='Accuracy', topk=(1, 5))

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator

# 模型设置
model = dict(
    type='mmpretrain.ImageClassifier',
    data_preprocessor=data_preprocessor,
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
        out_indices=(3,), # 只需要最后一层
        frozen_stages = 4, # 固定全部来自检测的参数，整个backbone的param全部freeze
        init_cfg=dict(
            type='Pretrained', 
            checkpoint='/data/zzl/work_dirs/dino_internimage-b_st2/epoch_80.pth',
            prefix = 'backbone',
            map_location = 'cpu'
        )
    ),
    neck=dict(
        type='mmpretrain.GlobalAveragePooling', # 输出 bs x 896
    ),
    head=dict(
        type='mmpretrain.MultiLinearClsHead', # 双层linear, 896 -> 512 -> 196, 可以实验多层
        num_classes=196,
        in_channels=896,
        latent_channels=[('bn', 896), ('fc', 1024), ('bn', 1024), ('relu', 1024), ('fc', 896), ('bn', 896)], # 仅仅使用一个fc效果很差
        use_shortcut = True,
        loss=dict(type='mmpretrain.CrossEntropyLoss', loss_weight=1.0),
        init_cfg=None,
    ),
    init_cfg=dict(type='TruncNormal', layer=['Conv2d', 'Linear'], std=.02, bias=0.),
)


base_lr = 0.001 # cfg里 var = val, 这样var会变成tuple，终究还是python
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=base_lr, betas=(0.9, 0.999), weight_decay=4e-5),
    #clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1), # 对backbone的差分学习率
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=1.) # 分割头lr*10
        }
    )
)

# 训练参数设置
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
auto_scale_lr = dict(enable=True, base_batch_size=128) # 需要加上enable=True才自动打开

# runtime设置


default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=5), #  默认50
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, by_epoch=True),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='mmpretrain.VisualizationHook')
    )

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='mmpretrain.UniversalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=10, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False

fp16 = dict(loss_scale=512.)
work_dir = '/data/zzl/work_dirs/cls_interimage-base_fc512_st2'

find_unused_parameters=True
sync_bn = 'torch'