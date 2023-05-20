
###### 以下是一些设置 ######

## 整个训练逻辑使用 IterBasedRunner，所以需要根据 det, seg, cls 这三个任务的数据多少，结合使用的batch_size，计算出一共需要多少iter

# 需要修改的部分
cls_train_data_num = 8144
det_train_data_num = 6103
seg_train_data_num = 7000

# 不需要修改的部分
total_epoch_num = 160
lr_warmup_epoch_num = 5
cls_train_batchsize = 8 * 8
det_train_batchsize = 1 * 8
seg_train_batchsize = 2 * 8

# 计算完成一个epoch至少需要多少iter
iter_num_per_epoch = max(cls_train_data_num // cls_train_batchsize+1, det_train_data_num // det_train_batchsize + 1, seg_train_data_num // seg_train_batchsize + 1)
total_iter_num = iter_num_per_epoch * total_epoch_num
lr_warmup_iter_num = lr_warmup_epoch_num * iter_num_per_epoch


## 训练标签, 需要改成系统中文件的对应路径

det_train_data_root = ''
det_train_data_file = '../../data/det_train_val.json'

seg_train_data_root = ''
seg_train_ann_file = ''

cls_train_data_root = ''
cls_train_ann_file = '../../data/cls_train_and_val-amend.txt'


## 测试标签，需要改成系统中文件的对应路径
# 检测任务测试数据集图像根目录
det_test_data_root = '/data/panxue/track1/track1_test_data/dec/test'
# 检测任务测试集标签文件
det_test_ann_file='/data/panxue/track1/track1_test_data/dec/test.json'

# 分割任务数据集图像根目录，数据一般以.jpg结尾
seg_test_data_root = '/data/panxue/track1/track1_test_data/seg/images/test'
# 分割任务数据集的标签，标签图像一般以.png结尾
seg_test_ann_file = '/data/panxue/track1/track1_test_data/seg/images/test_label'

# 分类任务数据集图像根目录
cls_test_data_root = '/data/panxue/track1/track1_test_data/cls/test'
# 分类任务数据集的图像标签文件
cls_test_ann_file = '/data/panxue/track1/track1_train_data/cls/test/dataset-orginal-test-label.txt'

###### 以上是文件路径配置 ######

###### 以下是一些TTA设置 ######
tta_model = dict(
    type='DINOMultiTaskTTAModel',
    det_tta_cfg=dict(nms=dict(type='nms', iou_threshold=0.65), max_per_img=100))

det_img_scales = [(1040, 1248), (1280, 1536), (1920, 2304)]
#det_img_scales = [(1024, 1024)]

det_tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                # ``RandomFlip`` must be placed before ``Pad``, otherwise
                # bounding box coordinates after flipping cannot be
                # recovered correctly.
                dict(type='RandomFlip', prob=0.),
                dict(type='RandomFlip', prob=1.)
                
            ],
            [
                dict(type='Resize', scale=s, keep_ratio=True)
                for s in det_img_scales
            ],
            [dict(type='LoadAnnotations', with_bbox=True)],
            [
                dict(
                    type='PackDetInputs',
                    meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                               'scale_factor', 'flip', 'flip_direction'))
            ]
        ])
]

seg_img_w,seg_img_h = 102400,1024 # 短边以1024为中心，长边不受限制，无论长宽比如何，都是把短边resize到 1024, 之后滑窗预测。
#seg_img_ratios = [1.0]
seg_img_ratios = [0.75, 1.0, 1.25]
seg_tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale=(seg_img_w*r,seg_img_h*r), keep_ratio=True)
                for r in seg_img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], 
            [dict(type='mmseg.LoadAnnotations')], 
            [dict(type='mmseg.PackSegInputs',
                  meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                               'scale_factor', 'flip', 'flip_direction',))],
        ])
]

crop_num=5
cls_img_w,cls_img_h = 512,512
cls_img_ratios = [1.0]
# cls_img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
cls_tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            #[
            #    dict(type='mmpretrain.RandomResizedCrop',
            #         scale=(cls_img_w,cls_img_h), 
            #         crop_ratio_range=(0.9,1.0))
            #    for i in range(crop_num)
            #] +
            #[
            #    dict(type='Resize', scale = (int(cls_img_w*r), int(cls_img_h*r)))
            #    for r in cls_img_ratios
            #],
            #[
            #    dict(type='RandomFlip', prob=0., direction='horizontal'),
            #    dict(type='RandomFlip', prob=1., direction='horizontal')
            #], 
            [dict(type='Resize', scale = (512, 512), keep_ratio=False)],
            [dict(type='mmpretrain.PackInputs')],
        ]
    )
]



###### 以上是TTA设置 ######

default_scope = 'mmdet'
custom_imports = dict(imports=['mmpretrain.datasets.transforms'], allow_failed_imports=False)
# 关于检测的 dataloader 设置
backend_args = None
# 分辨率(1280, 1536), hw
det_train_mosaic_pipeline = [
    dict(type='Mosaic', img_scale=(1536, 1280), pad_val=114.0), # img_scale有hw含义，wh形式, mmdet中是wh形式，mmseg中是hw形式
    dict(
        type='RandomAffine',
        max_rotate_degree = 0,
        max_shear_degree = 0,
        max_translate_ratio = 0,
        scaling_ratio_range=(0.2, 1.5),
        border=(-1536 // 2, -1280 // 2), # wh形式
    ),
    dict( 
        type='MixUp', # 需要增加概率
        img_scale=(1536, 1280), # 从dataset中采样图像的scale, wh形式
        ratio_range=(0.5, 1.5), 
        prob = 1.0,
        pad_val=114.0,
    ),
    dict(
        type='YOLOXHSVRandomAug',
    ),
    dict(type='Resize', scale=(1536,1280), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5), # bboxes也会翻转,
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
    dict(type='PackDetInputs', _scope_ = 'mmdet')
]
det_train_multiscale_pipeline = [
    dict(
        type='RandomChoice',
        transforms=[
            [ # 单纯对每张图片多尺度训练，对图像整体的 keep_ratio 的缩放, 本质在提高小目标的鲁棒性
                dict(
                    type='RandomChoiceResize', # 比如 (480, 1333)，要求图像的 长边不大于1333 and 图像短边不大于480，等价先把短边缩放到480，如果长边大于1333，则把长边缩短到1333
                    scales=[(800, 1536), (832, 1536), (864, 1536), (896, 1536), (928, 1536), (960, 1536), (992, 1536), (1024, 1536), (1056, 1536), (1088, 1536),
                            (1120, 1536), (1152, 1536), (1184, 1536), (1216, 1536), (1248, 1536), (1280, 1536)], 
                            # 缩放比例 [0.78125, 0.8125, 0.84375, 0.875, 0.90625, 0.9375, 0.96875, 1.0]
                    keep_ratio=True # keep_ratio则是rescale
                )
            ],
            [
                dict( # 这里是在增加大目标的鲁棒性
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(832, 1536), (928, 1536), (1024, 1536), (1120, 1536), (1280, 1536)], # 先随机整体缩放到 832, 928, 1024,
                    keep_ratio=True
                ),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range', # crop_h = [crop_size[0], min(h, crop_size[1])], crop_w = [crop_size[0], min(w, crop_size[1])]
                    crop_size=(640, 1280),
                    allow_negative_crop=True, # 允许一个crop中啥都没有，完全是背景，可以抑制FP
                    _scope_='mmdet',
                ),
                dict(
                    type='RandomChoiceResize',
                    scales=[(800, 1536), (832, 1536), (864, 1536), (896, 1536), (928, 1536), (960, 1536), (992, 1536), (1024, 1536), (1056, 1536), (1088, 1536),
                            (1120, 1536), (1152, 1536), (1184, 1536), (1216, 1536), (1248, 1536), (1280, 1536)], 
                    keep_ratio=True
                )
            ]
        ]
    ),
    dict(type='RandomFlip', prob=0.5), # bboxes也会翻转,
    dict(
        type='RandomApply',
        transforms = dict(
                        type='mmdet.AutoAugment', # NOTE 注意使用的track1增强
                    ),
        prob = 0.6, # 以0.6概率用autoaug
    ),
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
    dict(type='PackDetInputs', _scope_ = 'mmdet')  # 得加上这个，坑爹货，法克
    # results中的imgs这个field变成inputs, 只有inputs和data_samples这两个field，其中inputs作为模型输入，data_samples是标注信息，会根据每个data_sample中ingore_gt的设置删除invalid的框
]
det_train_pipeline = [
    dict(
        type='RandomChoice',
        transforms = [
            det_train_mosaic_pipeline, # hard
            det_train_multiscale_pipeline # easy
        ],
        prob = [0.8, 0.2],
    )
]
#det_train_pipeline = det_train_mosaic_pipeline
det_test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1280, 1536), keep_ratio=True), # 测试时就是 1024x1024
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor')
        )
]
det_train_dataloader = dict(
    batch_size=1,
    num_workers=3,
    persistent_workers=True, # debug True
    sampler=dict(type='InfiniteSampler', shuffle=True),
    #batch_sampler=dict(type='AspectRatioBatchSampler', _scope_ = 'mmdet'), # 不需要，数据集全是2048x2048
    dataset=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type='Track1DetDataset',
            data_root=det_train_data_root,
            ann_file=det_train_data_file,
            data_prefix=dict(img=''),
            filter_cfg=dict(filter_empty_gt=False), # filter_empty_gt=True, min_size=32,
            pipeline=[
                dict(type='LoadImageFromFile', backend_args=backend_args), 
                dict(type='LoadAnnotations', with_bbox=True), # 原始data_list[idx]中是instances:List，从中解析出gt_bboxs: np.ndarray(N, 4), gt_bboxes_labels: np.ndarray(N, ) 等等
            ],
            backend_args=backend_args,
            serialize_data = False, # for debug
            _scope_ = 'mmdet',
        ),
        pipeline=det_train_pipeline,
    )
)

det_val_dataloader = dict(
    batch_size=12,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False), # test_dataset必须DefaultSampler
    dataset=dict(
        type='Track1DetDataset',
        data_root=det_test_data_root,
        ann_file=det_test_ann_file,
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=det_test_pipeline,
        backend_args=backend_args,
        serialize_data = False, # for debug
        _scope_ = 'mmdet',
        )
    )
det_test_dataloader = det_val_dataloader

# 关于分割的dataloader设置
seg_weather_aug = dict(
    type = 'RandomChoice',
    transforms = [
        [
            dict(type='mmpretrain.Spatter', severity_range=(1,3)),
            dict(type='mmpretrain.Rain', severity_range=(1,3)),
        ],
        [
            dict(type='mmpretrain.Frost', severity_range=(2,4)),
        ],
        [
            dict(type='mmpretrain.Fog', severity_range=(2,4)),
        ],
        [
            dict(type='mmpretrain.Fog', severity_range=(1,3)),
            dict(type='mmpretrain.Rain', severity_range=(1,3)),
        ],
        [
            dict(type='mmpretrain.Fog', severity_range=(1,3)),
            dict(type='mmpretrain.Spatter', severity_range=(1,3)),
        ],
        [
            dict(type='mmpretrain.ColorJitter', brightness = 0.8), # 随机黑夜 + 随机强光过曝
        ],
    ],
)

seg_train_multiscale_pipeline = [
    dict(
        type='RandomResize', # 从ratio_range中采样ratio -> ratio * scale = dst_scale -> 用 dst_scale 对图像 rescale, 如果keep_ratio，先把短边缩放到短边，再保证长边不超过长的
        scale=(4096, 1024), # 设置最大短边长边比1:4, 仅仅保证短边被resize到1024，长边不做限制
        ratio_range=(0.5, 2.0), # (910, 512) ~ (1820, 1024) ~ (1820, 1024)*2
        keep_ratio=True, # True则scale没有宽高的意义
    ),
    dict(
        type='mmseg.RandomCrop', # crop出 crop_size 的patch，形状定死, 如果 图像尺寸 < crop_size，则直接返回原图
        crop_size=(1024, 1024), # (h, w)形式，
        cat_max_ratio=0.75, # crop单个类别在crop_img中的最高比例
    ),
    dict(
        type = 'RandomApply',
        transforms = seg_weather_aug,
        prob = 0.1 # 0.1概率使用天气增强
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
seg_train_mosaic_pipeline = [
    dict(
        type = 'mmseg.RandomMosaic', # 输出形状固定 hw(720x2, 1280x2)
        prob = 1.0,
        img_scale = (720, 1280), # (h, w) 形式, 有宽高的含义, 采样的4张图像首先都会宽高对应rescale到这个范围内
        center_ratio_range=(1.0, 1.0),
        pad_val=0,
        seg_pad_val=255,
    ),
    dict(
        type = 'RandomResize',
        scale = (1024, 1820), # 基准1024, 来自mosaic，相当于obj_scale最小为0.5
        ratio_range = (1.0, 2.0), # 基准1024, obj_scale最小0.5, 最大1.0，专门对小目标的增强
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
seg_train_pipeline = [
    dict(
        type='RandomChoice',
        transforms = [
            seg_train_mosaic_pipeline,
            seg_train_multiscale_pipeline,
        ],
        prob = [0.4, 0.6], # 全程0.4, 0.6概率使用mosaic
    )
]
seg_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1024000, 1024), keep_ratio=True), # 无论图像长宽比如何，把图像短边resize到1024，之后滑窗预测
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='mmseg.LoadAnnotations'),
    dict(type='mmseg.PackSegInputs'),
]
seg_train_dataloader = dict(
    batch_size=2,
    num_workers=6,
    persistent_workers=True, # debug True
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset = dict(
        type='mmseg.MultiImageMixDataset',
        dataset=dict(
            type='mmseg.Track1SegDataset',
            data_prefix=dict(
                img_path=seg_train_data_root, 
                seg_map_path=seg_train_ann_file
            ),
            img_suffix='.jpg',
            seg_map_suffix='.png',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='mmseg.LoadAnnotations'),
            ],
            serialize_data = False, # debug用
        ),
        pipeline=seg_train_pipeline,
    )
)
seg_val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='mmseg.Track1SegDataset',
        data_prefix=dict(
            img_path=seg_test_data_root, 
            seg_map_path=seg_test_ann_file,
        ),
        img_suffix='.jpg',
        seg_map_suffix='.png',
        pipeline=seg_test_pipeline,
        serialize_data = False, # debug
        #indices = 3, # debug
        ),
    )
seg_test_dataloader = seg_val_dataloader

# 关于分类的dataloader

bgr_mean = [123.675, 116.28, 103.53][::-1]
bgr_std = [58.395, 57.12, 57.375][::-1]

cls_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale = (512, 512)),
    dict(
        type='RandomApply',
        transforms = dict( # 以0.5概率用autoaug
                        type='mmpretrain.RandomResizedCrop',
                        scale = (512, 512),
                        crop_ratio_range = (0.98, 1.0),
                    ),
        prob = 0.2,
    ),
    dict(
        type='mmpretrain.RandomErasing',
        erase_prob=0.5,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=1 / 3,
        fill_color=bgr_mean,
        fill_std=bgr_std
    ),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandomApply',
        transforms = dict( # 以0.5概率用autoaug
                        type='mmpretrain.AutoAugment',
                        policies='imagenet',
                        hparams=dict(
                        pad_val=[round(x) for x in bgr_mean], interpolation='bilinear')
                    ),
        prob = 0.8,
    ),
    dict(
        type='mmpretrain.Albu',
        transforms = [
            dict(
                type='ShiftScaleRotate',
                shift_limit=0.0,
                scale_limit=[-0.1, 0.2],
                rotate_limit=30,
                interpolation=1,
                border_mode=1,#复制边缘像素
                p=0.5
                ),
            dict(
                type='Perspective',
                pad_mode=1,
                p=0.5
            ),
            dict(
	            type='OneOf',
	                transforms=[
		                 dict(type='GaussNoise', var_limit=(0.0,50.0), p=1.0),
		                 dict(type='Blur', blur_limit=3, p=1.0),
		                 dict(type='Emboss', alpha=(0.2,0.5),strength=(0.2,0.7),p=0.5),
		                 dict(type='GaussianBlur', blur_limit=(3,5),sigma_limit=0,p=1),
		                 dict(type='MultiplicativeNoise', multiplier=(0.8, 1.2), elementwise=True,p=1.0),
		                 dict(type='Superpixels',p_replace=0.2,p=0.5),
	                ],
	            p=0.6
            ),
        ]
    ),
    dict(type='mmpretrain.PackInputs'),
]

cls_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale = (512, 512)),
    dict(type='mmpretrain.PackInputs'),
]

cls_train_dataloader = dict(
    batch_size=8,
    num_workers=6,
    dataset=dict(
        type='mmpretrain.CustomDataset',
        data_root = cls_train_data_root,
        ann_file = cls_train_ann_file,
        pipeline=cls_train_pipeline,
        serialize_data = False, # debug用
        #indices = 1, # debug 用
        ),
    sampler=dict(type='InfiniteSampler', shuffle=True),
)

cls_val_dataloader = dict(
    batch_size=36,
    num_workers=6,
    dataset=dict(
        type='mmpretrain.CustomDataset',
        data_root = cls_test_data_root,
        ann_file = cls_test_ann_file,
        pipeline =cls_test_pipeline, 
        serialize_data = False, # debug用
        ),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
cls_test_dataloader = cls_val_dataloader

# 最后整体的dataloader
# 训练用dataloader, 每次同时从 det, seg, cls 中加载数据
train_dataloader = dict(
    mode='train',
    det=det_train_dataloader,
    seg=seg_train_dataloader,
    cls=cls_train_dataloader,
)
#train_dataloader = seg_train_dataloader
# 测试用dataloader, 依次从 det, seg, cls 中加载数据
val_dataloader = dict(
    mode='val',
    det = det_val_dataloader,
    seg = seg_val_dataloader,
    cls = cls_val_dataloader,
)
test_dataloader = val_dataloader

# 设置 metric
# det的metric
det_val_evaluator = dict(
    type='mmdet.CocoMetric',
    ann_file=det_test_ann_file,
    classwise = True,
    metric='bbox',
    format_only=False,
    backend_args=backend_args
    )
det_test_evaluator = det_val_evaluator

# seg的metric
seg_val_evaluator = dict(type='mmseg.IoUMetric', iou_metrics=['mIoU'])
seg_test_evaluator = seg_val_evaluator

# cls的metric
cls_val_evaluator = dict(type='mmpretrain.Accuracy', topk=(1, 5))
cls_test_evaluator = cls_val_evaluator

# 最终的 evaluator
val_evaluator = dict(
    det = det_val_evaluator,
    seg = seg_val_evaluator,
    cls = cls_val_evaluator
)
test_evaluator = val_evaluator

# 对模型中data_preprocesser的设置：
det_data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=1,
    _scope_ = 'mmdet', 
)
seg_data_preprocessor = dict(
    type='mmseg.SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0, # img的pad
    seg_pad_val=255, # label的pad
    size = (1024, 1024), #(h, w), 原(720, 1280)有问题，加载出来的1024x1024图像被padding成 1024x1280的尺寸
)
cls_data_preprocessor = dict(
    type='mmpretrain.ClsDataPreprocessor',
    num_classes=196,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)
data_preprocessor = dict(
    type='mmdet.MultiTaskDataPreprocessor',
    det = det_data_preprocessor,
    seg = seg_data_preprocessor,
    cls = cls_data_preprocessor,
)
#data_preprocessor = seg_data_preprocessor

model = dict(
    type='DINOMultiTask',
    num_queries=900,
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='InternImage',
        core_op='DCNv3',
        channels=192,
        depths=[5, 5, 24, 5],
        groups=[12, 24, 48, 96],
        mlp_ratio=4.,
        drop_path_rate=0.4,
        norm_layer='LN',
        layer_scale=1.0,
        offset_scale=2.0,
        post_norm=True,
        with_cp=False,
        out_indices=(0, 1, 2, 3),
        init_cfg=dict(
            type='Pretrained', 
            checkpoint='../params/internimage_xl_22kto1k_384_.pth',
            map_location = 'cpu'
        )
    ),
    neck=dict( # 默认是检测的neck
        type='ChannelMapper',
        in_channels=[192, 384, 768, 1536],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=5),
    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=5, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0))),
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            cross_attn_cfg=dict(embed_dims=256, num_levels=5, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0)),
        post_norm_cfg=None),
    positional_encoding=dict(
        num_feats=128, normalize=True, offset=0.0, temperature=20),
    bbox_head=dict(
        type='DINOHead',
        embed_dims = 256,
        num_classes=45,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    dn_cfg=dict(
        label_noise_scale=0.5,
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=100)),
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])
        ),
    test_cfg=dict(max_per_img=210),
    num_feature_levels=5,
    # 以上都是原始 det 的设置, 实例化时这些 key 不能冲突
    # 分割头设置
    seg_head=dict(
        type='mmseg.Mask2FormerHead',
        in_channels=[192, 384, 768, 1536],
        strides=[4, 8, 16, 32],
        feat_channels=256,
        out_channels=256,
        num_classes=19,
        num_queries=100,
        num_transformer_feat_level=3,
        align_corners=False,
        pixel_decoder=dict(
            type='mmdet.MSDeformAttnPixelDecoder',
            num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(  # DeformableDetrTransformerEncoder
                num_layers=6,
                layer_cfg=dict(  # DeformableDetrTransformerEncoderLayer
                    self_attn_cfg=dict(  # MultiScaleDeformableAttention
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        im2col_step=64,
                        dropout=0.0,
                        batch_first=True,
                        norm_cfg=None,
                        init_cfg=None),
                    ffn_cfg=dict(
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True))),
                init_cfg=None),
            positional_encoding=dict(  # SinePositionalEncoding
                num_feats=128, normalize=True),
            init_cfg=None),
        enforce_decoder_input_project=False,
        positional_encoding=dict(  # SinePositionalEncoding
            num_feats=128, normalize=True),
        transformer_decoder=dict(  # Mask2FormerTransformerDecoder
            return_intermediate=True,
            num_layers=9,
            layer_cfg=dict(  # Mask2FormerTransformerDecoderLayer
                self_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=True),
                cross_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=True),
                ffn_cfg=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True)),
            init_cfg=None),
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0] * 19 + [0.1]),
        loss_mask=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='mmdet.DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0),
        train_cfg=dict(
            num_points=12544,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            assigner=dict(
                type='mmdet.HungarianAssigner',
                match_costs=[
                    dict(type='mmdet.ClassificationCost', weight=2.0),
                    dict(
                        type='mmdet.CrossEntropyLossCost',
                        weight=5.0,
                        use_sigmoid=True),
                    dict(
                        type='mmdet.DiceCost',
                        weight=5.0,
                        pred_act=True,
                        eps=1.0)
                ]),
            sampler=dict(type='mmdet.MaskPseudoSampler'))),
    seg_test_cfg=dict(
        mode = 'slide', 
        crop_size = (1024, 1024),
        stride = (256, 256),
    ),
    seg_align_corners = False, 
    # 分类头设置 192, 384, 768, 1536
    cls_neck=dict(
        type='mmpretrain.MultiLinearNeck', # 输出 bs x 896
        in_channels=1536,
        latent_channels=[('bn', 1536), ('fc', 1024), ('bn', 1024), ('relu', 1024), ('fc', 1536), ('bn', 1536)], # 仅仅使用一个fc效果很差
        use_shortcut = True,
        init_cfg=dict(type='TruncNormal', layer=['Conv2d', 'Linear'], std=.02, bias=0.),
    ),
    cls_head=dict(
        type='mmpretrain.ArcFaceClsHead', # 双层linear, 896 -> 512 -> 196, 可以实验多层
        num_classes=196,
        in_channels=1536,
        num_subcenters = 1,
        loss=dict(type='mmpretrain.CrossEntropyLoss', loss_weight=1.0),
    ),
)

# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.

# 训练逻辑 loop 设置
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=total_iter_num, val_interval=iter_num_per_epoch) # 440
# 测试 loop 设置
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# optimizer
base_lr = 0.0002
optim_wrapper = dict(
    type='OptimWrapper', # 全精度
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.005),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1), # 对backbone的差分学习率
            'norm': dict(decay_mult=0.),
            'pos_block': dict(decay_mult=0.),
            'query_embed': dict(lr_mult=1.0, decay_mult=0.0),
            'query_feat': dict(lr_mult=1.0, decay_mult=0.0),
            'level_embed': dict(lr_mult=1.0, decay_mult=0.0),
            'level_embedding': dict(lr_mult=1.0, decay_mult=0.0),
            'cls_token':  dict(lr_mult=1.0, decay_mult=0.0),
            'posi_embedding': dict(lr_mult=1.0, decay_mult=0.0),
            }
        )
    )


param_scheduler = [
    dict(
        type='LinearLR', 
        start_factor = 1e-3, 
        end_factor = 1.0,
        begin=0,  
        end=lr_warmup_iter_num,
        by_epoch=False,  
        #convert_to_iter_based=True,
    ), 
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 1e-3,
        begin=lr_warmup_iter_num,
        end=total_iter_num,
        by_epoch=False,
        #convert_to_iter_based=True
    ),
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (16 GPUs) x (2 samples per GPU)
#auto_scale_lr = dict(enable=True, base_batch_size=16) # 需要加上enable=True才自动打开

# runtime设置
default_scope = 'mmdet'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False), # 50
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=iter_num_per_epoch), # 450
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook')
    )

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=10, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False

fp16 = dict(loss_scale=512.)
work_dir = '../work_dirs/submit-trainV2'
sync_bn = 'torch'
find_unused_parameters=True
custom_hooks = [dict(type='mmpretrain.SetAdaptiveMarginsHook', margin_min = 0.5, margin_max=0.55)]