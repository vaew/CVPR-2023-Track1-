default_scope = 'mmdet'
custom_imports = dict(
    imports=['mmpretrain.datasets.transforms'], allow_failed_imports=False)
backend_args = None
det_train_mosaic_pipeline = [
    dict(type='Mosaic', img_scale=(1024, 1024), pad_val=114.0),
    dict(
        type='RandomAffine',
        max_rotate_degree=0,
        max_shear_degree=0,
        max_translate_ratio=0,
        scaling_ratio_range=(0.2, 1.5),
        border=(-512, -512)),
    dict(
        type='MixUp',
        img_scale=(1024, 1024),
        ratio_range=(0.5, 1.5),
        prob=1.0,
        pad_val=114.0),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='mmpretrain.Albu',
        transforms=[
            dict(
                type='OneOf',
                transforms=[
                    dict(type='GaussNoise', var_limit=(0.0, 50.0), p=1.0),
                    dict(type='Blur', blur_limit=3, p=1.0),
                    dict(
                        type='Emboss',
                        alpha=(0.2, 0.5),
                        strength=(0.2, 0.7),
                        p=0.5),
                    dict(
                        type='GaussianBlur',
                        blur_limit=(3, 5),
                        sigma_limit=0,
                        p=1),
                    dict(
                        type='MultiplicativeNoise',
                        multiplier=(0.8, 1.2),
                        elementwise=True,
                        p=1.0)
                ],
                p=0.6)
        ]),
    dict(type='PackDetInputs', _scope_='mmdet')
]
det_train_multiscale_pipeline = [
    dict(
        type='RandomChoice',
        transforms=[[{
            'type':
            'RandomChoiceResize',
            'scales': [(800, 1024), (832, 1024), (864, 1024), (896, 1024),
                       (928, 1024), (960, 1024), (992, 1024), (1024, 1024)],
            'keep_ratio':
            True
        }],
                    [{
                        'type': 'RandomChoiceResize',
                        'scales': [(832, 1024), (928, 1024), (1024, 1024)],
                        'keep_ratio': True
                    }, {
                        'type': 'RandomCrop',
                        'crop_type': 'absolute_range',
                        'crop_size': (512, 1024),
                        'allow_negative_crop': True,
                        '_scope_': 'mmdet'
                    }, {
                        'type':
                        'RandomChoiceResize',
                        'scales': [(800, 1024), (832, 1024), (864, 1024),
                                   (896, 1024), (928, 1024), (960, 1024),
                                   (992, 1024), (1024, 1024)],
                        'keep_ratio':
                        True
                    }]]),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomApply',
        transforms=dict(type='mmdet.AutoAugment'),
        prob=0.6),
    dict(
        type='mmpretrain.Albu',
        transforms=[
            dict(
                type='OneOf',
                transforms=[
                    dict(type='GaussNoise', var_limit=(0.0, 50.0), p=1.0),
                    dict(type='Blur', blur_limit=3, p=1.0),
                    dict(
                        type='Emboss',
                        alpha=(0.2, 0.5),
                        strength=(0.2, 0.7),
                        p=0.5),
                    dict(
                        type='GaussianBlur',
                        blur_limit=(3, 5),
                        sigma_limit=0,
                        p=1),
                    dict(
                        type='MultiplicativeNoise',
                        multiplier=(0.8, 1.2),
                        elementwise=True,
                        p=1.0)
                ],
                p=0.6)
        ]),
    dict(type='PackDetInputs', _scope_='mmdet')
]
det_train_pipeline = [
    dict(
        type='RandomChoice',
        transforms=[[{
            'type': 'Mosaic',
            'img_scale': (1024, 1024),
            'pad_val': 114.0
        }, {
            'type': 'RandomAffine',
            'max_rotate_degree': 0,
            'max_shear_degree': 0,
            'max_translate_ratio': 0,
            'scaling_ratio_range': (0.2, 1.5),
            'border': (-512, -512)
        }, {
            'type': 'MixUp',
            'img_scale': (1024, 1024),
            'ratio_range': (0.5, 1.5),
            'prob': 1.0,
            'pad_val': 114.0
        }, {
            'type': 'YOLOXHSVRandomAug'
        }, {
            'type': 'Resize',
            'scale': (1024, 1024),
            'keep_ratio': True
        }, {
            'type': 'RandomFlip',
            'prob': 0.5
        }, {
            'type':
            'mmpretrain.Albu',
            'transforms': [{
                'type':
                'OneOf',
                'transforms': [{
                    'type': 'GaussNoise',
                    'var_limit': (0.0, 50.0),
                    'p': 1.0
                }, {
                    'type': 'Blur',
                    'blur_limit': 3,
                    'p': 1.0
                }, {
                    'type': 'Emboss',
                    'alpha': (0.2, 0.5),
                    'strength': (0.2, 0.7),
                    'p': 0.5
                }, {
                    'type': 'GaussianBlur',
                    'blur_limit': (3, 5),
                    'sigma_limit': 0,
                    'p': 1
                }, {
                    'type': 'MultiplicativeNoise',
                    'multiplier': (0.8, 1.2),
                    'elementwise': True,
                    'p': 1.0
                }],
                'p':
                0.6
            }]
        }, {
            'type': 'PackDetInputs',
            '_scope_': 'mmdet'
        }],
                    [{
                        'type':
                        'RandomChoice',
                        'transforms': [[{
                            'type':
                            'RandomChoiceResize',
                            'scales': [(800, 1024), (832, 1024), (864, 1024),
                                       (896, 1024), (928, 1024), (960, 1024),
                                       (992, 1024), (1024, 1024)],
                            'keep_ratio':
                            True
                        }],
                                       [{
                                           'type':
                                           'RandomChoiceResize',
                                           'scales': [(832, 1024), (928, 1024),
                                                      (1024, 1024)],
                                           'keep_ratio':
                                           True
                                       }, {
                                           'type': 'RandomCrop',
                                           'crop_type': 'absolute_range',
                                           'crop_size': (512, 1024),
                                           'allow_negative_crop': True,
                                           '_scope_': 'mmdet'
                                       }, {
                                           'type':
                                           'RandomChoiceResize',
                                           'scales': [(800, 1024), (832, 1024),
                                                      (864, 1024), (896, 1024),
                                                      (928, 1024), (960, 1024),
                                                      (992, 1024),
                                                      (1024, 1024)],
                                           'keep_ratio':
                                           True
                                       }]]
                    }, {
                        'type': 'RandomFlip',
                        'prob': 0.5
                    }, {
                        'type': 'RandomApply',
                        'transforms': {
                            'type': 'mmdet.AutoAugment'
                        },
                        'prob': 0.6
                    }, {
                        'type':
                        'mmpretrain.Albu',
                        'transforms': [{
                            'type':
                            'OneOf',
                            'transforms': [{
                                'type': 'GaussNoise',
                                'var_limit': (0.0, 50.0),
                                'p': 1.0
                            }, {
                                'type': 'Blur',
                                'blur_limit': 3,
                                'p': 1.0
                            }, {
                                'type': 'Emboss',
                                'alpha': (0.2, 0.5),
                                'strength': (0.2, 0.7),
                                'p': 0.5
                            }, {
                                'type': 'GaussianBlur',
                                'blur_limit': (3, 5),
                                'sigma_limit': 0,
                                'p': 1
                            }, {
                                'type': 'MultiplicativeNoise',
                                'multiplier': (0.8, 1.2),
                                'elementwise': True,
                                'p': 1.0
                            }],
                            'p':
                            0.6
                        }]
                    }, {
                        'type': 'PackDetInputs',
                        '_scope_': 'mmdet'
                    }]],
        prob=[0.8, 0.2])
]
det_test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
det_train_dataloader = dict(
    batch_size=2,
    num_workers=6,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type='Track1DetDataset',
            data_root=
            '/home/zhangzelun/.workspaceZZL/.wdata/track1_train_data/dec/train_and_val',
            ann_file=
            '/home/zhangzelun/.workspaceZZL/.wdata/track1_train_data/dec/train_val.json',
            data_prefix=dict(img=''),
            filter_cfg=dict(filter_empty_gt=False),
            pipeline=[
                dict(type='LoadImageFromFile', backend_args=None),
                dict(type='LoadAnnotations', with_bbox=True)
            ],
            backend_args=None,
            serialize_data=False,
            _scope_='mmdet'),
        pipeline=[
            dict(
                type='RandomChoice',
                transforms=[[{
                    'type': 'Mosaic',
                    'img_scale': (1024, 1024),
                    'pad_val': 114.0
                }, {
                    'type': 'RandomAffine',
                    'max_rotate_degree': 0,
                    'max_shear_degree': 0,
                    'max_translate_ratio': 0,
                    'scaling_ratio_range': (0.2, 1.5),
                    'border': (-512, -512)
                }, {
                    'type': 'MixUp',
                    'img_scale': (1024, 1024),
                    'ratio_range': (0.5, 1.5),
                    'prob': 1.0,
                    'pad_val': 114.0
                }, {
                    'type': 'YOLOXHSVRandomAug'
                }, {
                    'type': 'Resize',
                    'scale': (1024, 1024),
                    'keep_ratio': True
                }, {
                    'type': 'RandomFlip',
                    'prob': 0.5
                }, {
                    'type':
                    'mmpretrain.Albu',
                    'transforms': [{
                        'type':
                        'OneOf',
                        'transforms': [{
                            'type': 'GaussNoise',
                            'var_limit': (0.0, 50.0),
                            'p': 1.0
                        }, {
                            'type': 'Blur',
                            'blur_limit': 3,
                            'p': 1.0
                        }, {
                            'type': 'Emboss',
                            'alpha': (0.2, 0.5),
                            'strength': (0.2, 0.7),
                            'p': 0.5
                        }, {
                            'type': 'GaussianBlur',
                            'blur_limit': (3, 5),
                            'sigma_limit': 0,
                            'p': 1
                        }, {
                            'type': 'MultiplicativeNoise',
                            'multiplier': (0.8, 1.2),
                            'elementwise': True,
                            'p': 1.0
                        }],
                        'p':
                        0.6
                    }]
                }, {
                    'type': 'PackDetInputs',
                    '_scope_': 'mmdet'
                }],
                            [{
                                'type':
                                'RandomChoice',
                                'transforms': [[{
                                    'type':
                                    'RandomChoiceResize',
                                    'scales': [(800, 1024), (832, 1024),
                                               (864, 1024), (896, 1024),
                                               (928, 1024), (960, 1024),
                                               (992, 1024), (1024, 1024)],
                                    'keep_ratio':
                                    True
                                }],
                                               [{
                                                   'type':
                                                   'RandomChoiceResize',
                                                   'scales': [(832, 1024),
                                                              (928, 1024),
                                                              (1024, 1024)],
                                                   'keep_ratio':
                                                   True
                                               }, {
                                                   'type': 'RandomCrop',
                                                   'crop_type':
                                                   'absolute_range',
                                                   'crop_size': (512, 1024),
                                                   'allow_negative_crop': True,
                                                   '_scope_': 'mmdet'
                                               }, {
                                                   'type':
                                                   'RandomChoiceResize',
                                                   'scales': [(800, 1024),
                                                              (832, 1024),
                                                              (864, 1024),
                                                              (896, 1024),
                                                              (928, 1024),
                                                              (960, 1024),
                                                              (992, 1024),
                                                              (1024, 1024)],
                                                   'keep_ratio':
                                                   True
                                               }]]
                            }, {
                                'type': 'RandomFlip',
                                'prob': 0.5
                            }, {
                                'type': 'RandomApply',
                                'transforms': {
                                    'type': 'mmdet.AutoAugment'
                                },
                                'prob': 0.6
                            }, {
                                'type':
                                'mmpretrain.Albu',
                                'transforms': [{
                                    'type':
                                    'OneOf',
                                    'transforms': [{
                                        'type': 'GaussNoise',
                                        'var_limit': (0.0, 50.0),
                                        'p': 1.0
                                    }, {
                                        'type': 'Blur',
                                        'blur_limit': 3,
                                        'p': 1.0
                                    }, {
                                        'type': 'Emboss',
                                        'alpha': (0.2, 0.5),
                                        'strength': (0.2, 0.7),
                                        'p': 0.5
                                    }, {
                                        'type': 'GaussianBlur',
                                        'blur_limit': (3, 5),
                                        'sigma_limit': 0,
                                        'p': 1
                                    }, {
                                        'type': 'MultiplicativeNoise',
                                        'multiplier': (0.8, 1.2),
                                        'elementwise': True,
                                        'p': 1.0
                                    }],
                                    'p':
                                    0.6
                                }]
                            }, {
                                'type': 'PackDetInputs',
                                '_scope_': 'mmdet'
                            }]],
                prob=[0.8, 0.2])
        ]))
det_val_dataloader = dict(
    batch_size=24,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='Track1DetDataset',
        data_root=
        '/home/zhangzelun/.workspaceZZL/.wdata/track1_test_data/dec/test',
        ann_file=
        '/home/zhangzelun/.workspaceZZL/.wdata/track1_test_data/dec/test.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ],
        backend_args=None,
        serialize_data=False,
        _scope_='mmdet'))
det_test_dataloader = dict(
    batch_size=24,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='Track1DetDataset',
        data_root=
        '/home/zhangzelun/.workspaceZZL/.wdata/track1_test_data/dec/test',
        ann_file=
        '/home/zhangzelun/.workspaceZZL/.wdata/track1_test_data/dec/test.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ],
        backend_args=None,
        serialize_data=False,
        _scope_='mmdet'))
seg_weather_aug = dict(
    type='RandomChoice',
    transforms=[[{
        'type': 'mmpretrain.Spatter',
        'severity_range': (1, 3)
    }, {
        'type': 'mmpretrain.Rain',
        'severity_range': (1, 3)
    }], [{
        'type': 'mmpretrain.Frost',
        'severity_range': (2, 4)
    }], [{
        'type': 'mmpretrain.Fog',
        'severity_range': (2, 4)
    }],
                [{
                    'type': 'mmpretrain.Fog',
                    'severity_range': (1, 3)
                }, {
                    'type': 'mmpretrain.Rain',
                    'severity_range': (1, 3)
                }],
                [{
                    'type': 'mmpretrain.Fog',
                    'severity_range': (1, 3)
                }, {
                    'type': 'mmpretrain.Spatter',
                    'severity_range': (1, 3)
                }], [{
                    'type': 'mmpretrain.ColorJitter',
                    'brightness': 0.8
                }]])
seg_train_multiscale_pipeline = [
    dict(
        type='RandomResize',
        scale=(1820, 1024),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='mmseg.RandomCrop', crop_size=(1024, 1024), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='mmseg.PhotoMetricDistortion'),
    dict(type='RandomApply', transforms=dict(type='mmseg.RGB2Gray'), prob=0.1),
    dict(type='mmseg.RandomRotate', prob=0.2, degree=15),
    dict(
        type='mmpretrain.Albu',
        transforms=[
            dict(
                type='OneOf',
                transforms=[
                    dict(type='GaussNoise', var_limit=(0.0, 50.0), p=1.0),
                    dict(type='Blur', blur_limit=3, p=1.0),
                    dict(
                        type='Emboss',
                        alpha=(0.2, 0.5),
                        strength=(0.2, 0.7),
                        p=0.5),
                    dict(
                        type='GaussianBlur',
                        blur_limit=(3, 5),
                        sigma_limit=0,
                        p=1),
                    dict(
                        type='MultiplicativeNoise',
                        multiplier=(0.8, 1.2),
                        elementwise=True,
                        p=1.0)
                ],
                p=0.6)
        ]),
    dict(type='mmseg.PackSegInputs')
]
seg_train_mosaic_pipeline = [
    dict(
        type='mmseg.RandomMosaic',
        prob=1.0,
        img_scale=(720, 1280),
        center_ratio_range=(1.0, 1.0),
        pad_val=0,
        seg_pad_val=255),
    dict(
        type='RandomResize',
        scale=(1024, 1820),
        ratio_range=(1.0, 2.0),
        keep_ratio=True),
    dict(type='mmseg.RandomCrop', crop_size=(1024, 1024), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='mmseg.PhotoMetricDistortion'),
    dict(
        type='mmpretrain.Albu',
        transforms=[
            dict(
                type='OneOf',
                transforms=[
                    dict(type='GaussNoise', var_limit=(0.0, 50.0), p=1.0),
                    dict(type='Blur', blur_limit=3, p=1.0),
                    dict(
                        type='Emboss',
                        alpha=(0.2, 0.5),
                        strength=(0.2, 0.7),
                        p=0.5),
                    dict(
                        type='GaussianBlur',
                        blur_limit=(3, 5),
                        sigma_limit=0,
                        p=1),
                    dict(
                        type='MultiplicativeNoise',
                        multiplier=(0.8, 1.2),
                        elementwise=True,
                        p=1.0)
                ],
                p=0.6)
        ]),
    dict(type='mmseg.PackSegInputs')
]
seg_train_pipeline = [
    dict(
        type='mmseg.RandomChoiceSegTrack1',
        transforms=[[{
            'type': 'mmseg.RandomMosaic',
            'prob': 1.0,
            'img_scale': (720, 1280),
            'center_ratio_range': (1.0, 1.0),
            'pad_val': 0,
            'seg_pad_val': 255
        }, {
            'type': 'RandomResize',
            'scale': (1024, 1820),
            'ratio_range': (1.0, 2.0),
            'keep_ratio': True
        }, {
            'type': 'mmseg.RandomCrop',
            'crop_size': (1024, 1024),
            'cat_max_ratio': 0.75
        }, {
            'type': 'RandomFlip',
            'prob': 0.5
        }, {
            'type': 'mmseg.PhotoMetricDistortion'
        }, {
            'type':
            'mmpretrain.Albu',
            'transforms': [{
                'type':
                'OneOf',
                'transforms': [{
                    'type': 'GaussNoise',
                    'var_limit': (0.0, 50.0),
                    'p': 1.0
                }, {
                    'type': 'Blur',
                    'blur_limit': 3,
                    'p': 1.0
                }, {
                    'type': 'Emboss',
                    'alpha': (0.2, 0.5),
                    'strength': (0.2, 0.7),
                    'p': 0.5
                }, {
                    'type': 'GaussianBlur',
                    'blur_limit': (3, 5),
                    'sigma_limit': 0,
                    'p': 1
                }, {
                    'type': 'MultiplicativeNoise',
                    'multiplier': (0.8, 1.2),
                    'elementwise': True,
                    'p': 1.0
                }],
                'p':
                0.6
            }]
        }, {
            'type': 'mmseg.PackSegInputs'
        }],
                    [{
                        'type': 'RandomResize',
                        'scale': (1820, 1024),
                        'ratio_range': (0.5, 2.0),
                        'keep_ratio': True
                    }, {
                        'type': 'mmseg.RandomCrop',
                        'crop_size': (1024, 1024),
                        'cat_max_ratio': 0.75
                    }, {
                        'type': 'RandomFlip',
                        'prob': 0.5
                    }, {
                        'type': 'mmseg.PhotoMetricDistortion'
                    }, {
                        'type': 'RandomApply',
                        'transforms': {
                            'type': 'mmseg.RGB2Gray'
                        },
                        'prob': 0.1
                    }, {
                        'type': 'mmseg.RandomRotate',
                        'prob': 0.2,
                        'degree': 15
                    }, {
                        'type':
                        'mmpretrain.Albu',
                        'transforms': [{
                            'type':
                            'OneOf',
                            'transforms': [{
                                'type': 'GaussNoise',
                                'var_limit': (0.0, 50.0),
                                'p': 1.0
                            }, {
                                'type': 'Blur',
                                'blur_limit': 3,
                                'p': 1.0
                            }, {
                                'type': 'Emboss',
                                'alpha': (0.2, 0.5),
                                'strength': (0.2, 0.7),
                                'p': 0.5
                            }, {
                                'type': 'GaussianBlur',
                                'blur_limit': (3, 5),
                                'sigma_limit': 0,
                                'p': 1
                            }, {
                                'type': 'MultiplicativeNoise',
                                'multiplier': (0.8, 1.2),
                                'elementwise': True,
                                'p': 1.0
                            }],
                            'p':
                            0.6
                        }]
                    }, {
                        'type': 'mmseg.PackSegInputs'
                    }]],
        prob=[0.4, 0.6],
        total_iter_num=44000,
        mosaic_shutdown_iter_ratio=[0.4],
        mosaic_use_prob=[0.0])
]
seg_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1820, 1024), keep_ratio=True),
    dict(type='mmseg.LoadAnnotations'),
    dict(type='mmseg.PackSegInputs')
]
seg_train_dataloader = dict(
    batch_size=2,
    num_workers=6,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='mmseg.MultiImageMixDataset',
        dataset=dict(
            type='mmseg.Track1SegDataset',
            data_prefix=dict(
                img_path=
                '/home/zhangzelun/.workspaceZZL/.wdata/track1_train_data/seg/images/train_and_val',
                seg_map_path=
                '/home/zhangzelun/.workspaceZZL/.wdata/track1_train_data/seg/label/train_and_val'
            ),
            img_suffix='.jpg',
            seg_map_suffix='.png',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='mmseg.LoadAnnotations')
            ],
            serialize_data=False),
        pipeline=[
            dict(
                type='mmseg.RandomChoiceSegTrack1',
                transforms=[[{
                    'type': 'mmseg.RandomMosaic',
                    'prob': 1.0,
                    'img_scale': (720, 1280),
                    'center_ratio_range': (1.0, 1.0),
                    'pad_val': 0,
                    'seg_pad_val': 255
                }, {
                    'type': 'RandomResize',
                    'scale': (1024, 1820),
                    'ratio_range': (1.0, 2.0),
                    'keep_ratio': True
                }, {
                    'type': 'mmseg.RandomCrop',
                    'crop_size': (1024, 1024),
                    'cat_max_ratio': 0.75
                }, {
                    'type': 'RandomFlip',
                    'prob': 0.5
                }, {
                    'type': 'mmseg.PhotoMetricDistortion'
                }, {
                    'type':
                    'mmpretrain.Albu',
                    'transforms': [{
                        'type':
                        'OneOf',
                        'transforms': [{
                            'type': 'GaussNoise',
                            'var_limit': (0.0, 50.0),
                            'p': 1.0
                        }, {
                            'type': 'Blur',
                            'blur_limit': 3,
                            'p': 1.0
                        }, {
                            'type': 'Emboss',
                            'alpha': (0.2, 0.5),
                            'strength': (0.2, 0.7),
                            'p': 0.5
                        }, {
                            'type': 'GaussianBlur',
                            'blur_limit': (3, 5),
                            'sigma_limit': 0,
                            'p': 1
                        }, {
                            'type': 'MultiplicativeNoise',
                            'multiplier': (0.8, 1.2),
                            'elementwise': True,
                            'p': 1.0
                        }],
                        'p':
                        0.6
                    }]
                }, {
                    'type': 'mmseg.PackSegInputs'
                }],
                            [{
                                'type': 'RandomResize',
                                'scale': (1820, 1024),
                                'ratio_range': (0.5, 2.0),
                                'keep_ratio': True
                            }, {
                                'type': 'mmseg.RandomCrop',
                                'crop_size': (1024, 1024),
                                'cat_max_ratio': 0.75
                            }, {
                                'type': 'RandomFlip',
                                'prob': 0.5
                            }, {
                                'type': 'mmseg.PhotoMetricDistortion'
                            }, {
                                'type': 'RandomApply',
                                'transforms': {
                                    'type': 'mmseg.RGB2Gray'
                                },
                                'prob': 0.1
                            }, {
                                'type': 'mmseg.RandomRotate',
                                'prob': 0.2,
                                'degree': 15
                            }, {
                                'type':
                                'mmpretrain.Albu',
                                'transforms': [{
                                    'type':
                                    'OneOf',
                                    'transforms': [{
                                        'type': 'GaussNoise',
                                        'var_limit': (0.0, 50.0),
                                        'p': 1.0
                                    }, {
                                        'type': 'Blur',
                                        'blur_limit': 3,
                                        'p': 1.0
                                    }, {
                                        'type': 'Emboss',
                                        'alpha': (0.2, 0.5),
                                        'strength': (0.2, 0.7),
                                        'p': 0.5
                                    }, {
                                        'type': 'GaussianBlur',
                                        'blur_limit': (3, 5),
                                        'sigma_limit': 0,
                                        'p': 1
                                    }, {
                                        'type': 'MultiplicativeNoise',
                                        'multiplier': (0.8, 1.2),
                                        'elementwise': True,
                                        'p': 1.0
                                    }],
                                    'p':
                                    0.6
                                }]
                            }, {
                                'type': 'mmseg.PackSegInputs'
                            }]],
                prob=[0.4, 0.6],
                total_iter_num=44000,
                mosaic_shutdown_iter_ratio=[0.4],
                mosaic_use_prob=[0.0])
        ]))
seg_val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='mmseg.Track1SegDataset',
        data_prefix=dict(
            img_path=
            '/home/zhangzelun/.workspaceZZL/.wdata/track1_test_data/seg/images/test',
            seg_map_path=
            '/home/zhangzelun/.workspaceZZL/.wdata/track1_test_data/seg/images/test_label'
        ),
        img_suffix='.jpg',
        seg_map_suffix='.png',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(1820, 1024), keep_ratio=True),
            dict(type='mmseg.LoadAnnotations'),
            dict(type='mmseg.PackSegInputs')
        ],
        serialize_data=False))
seg_test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='mmseg.Track1SegDataset',
        data_prefix=dict(
            img_path=
            '/home/zhangzelun/.workspaceZZL/.wdata/track1_test_data/seg/images/test',
            seg_map_path=
            '/home/zhangzelun/.workspaceZZL/.wdata/track1_test_data/seg/images/test_label'
        ),
        img_suffix='.jpg',
        seg_map_suffix='.png',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(1820, 1024), keep_ratio=True),
            dict(type='mmseg.LoadAnnotations'),
            dict(type='mmseg.PackSegInputs')
        ],
        serialize_data=False))
bgr_mean = [103.53, 116.28, 123.675]
bgr_std = [57.375, 57.12, 58.395]
cls_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512)),
    dict(
        type='RandomApply',
        transforms=dict(
            type='mmpretrain.RandomResizedCrop',
            scale=(512, 512),
            crop_ratio_range=(0.98, 1.0)),
        prob=0.2),
    dict(
        type='mmpretrain.RandomErasing',
        erase_prob=0.5,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=0.3333333333333333,
        fill_color=[103.53, 116.28, 123.675],
        fill_std=[57.375, 57.12, 58.395]),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandomApply',
        transforms=dict(
            type='mmpretrain.AutoAugment',
            policies='imagenet',
            hparams=dict(pad_val=[104, 116, 124], interpolation='bilinear')),
        prob=0.8),
    dict(
        type='mmpretrain.Albu',
        transforms=[
            dict(
                type='ShiftScaleRotate',
                shift_limit=0.0,
                scale_limit=[-0.1, 0.2],
                rotate_limit=30,
                interpolation=1,
                border_mode=1,
                p=0.5),
            dict(type='Perspective', pad_mode=1, p=0.5),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='GaussNoise', var_limit=(0.0, 50.0), p=1.0),
                    dict(type='Blur', blur_limit=3, p=1.0),
                    dict(
                        type='Emboss',
                        alpha=(0.2, 0.5),
                        strength=(0.2, 0.7),
                        p=0.5),
                    dict(
                        type='GaussianBlur',
                        blur_limit=(3, 5),
                        sigma_limit=0,
                        p=1),
                    dict(
                        type='MultiplicativeNoise',
                        multiplier=(0.8, 1.2),
                        elementwise=True,
                        p=1.0),
                    dict(type='Superpixels', p_replace=0.2, p=0.5)
                ],
                p=0.6)
        ]),
    dict(type='mmpretrain.PackInputs')
]
cls_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512)),
    dict(type='mmpretrain.PackInputs')
]
cls_train_dataloader = dict(
    batch_size=8,
    num_workers=6,
    dataset=dict(
        type='mmpretrain.CustomDataset',
        data_root=
        '/home/zhangzelun/.workspaceZZL/.wdata/track1_train_data/cls/train_and_val',
        ann_file=
        '/home/zhangzelun/.workspaceZZL/.wdata/track1_train_data/cls/train_and_val-amend.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(512, 512)),
            dict(
                type='RandomApply',
                transforms=dict(
                    type='mmpretrain.RandomResizedCrop',
                    scale=(512, 512),
                    crop_ratio_range=(0.98, 1.0)),
                prob=0.2),
            dict(
                type='mmpretrain.RandomErasing',
                erase_prob=0.5,
                mode='rand',
                min_area_ratio=0.02,
                max_area_ratio=0.3333333333333333,
                fill_color=[103.53, 116.28, 123.675],
                fill_std=[57.375, 57.12, 58.395]),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(
                type='RandomApply',
                transforms=dict(
                    type='mmpretrain.AutoAugment',
                    policies='imagenet',
                    hparams=dict(
                        pad_val=[104, 116, 124], interpolation='bilinear')),
                prob=0.8),
            dict(
                type='mmpretrain.Albu',
                transforms=[
                    dict(
                        type='ShiftScaleRotate',
                        shift_limit=0.0,
                        scale_limit=[-0.1, 0.2],
                        rotate_limit=30,
                        interpolation=1,
                        border_mode=1,
                        p=0.5),
                    dict(type='Perspective', pad_mode=1, p=0.5),
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(
                                type='GaussNoise',
                                var_limit=(0.0, 50.0),
                                p=1.0),
                            dict(type='Blur', blur_limit=3, p=1.0),
                            dict(
                                type='Emboss',
                                alpha=(0.2, 0.5),
                                strength=(0.2, 0.7),
                                p=0.5),
                            dict(
                                type='GaussianBlur',
                                blur_limit=(3, 5),
                                sigma_limit=0,
                                p=1),
                            dict(
                                type='MultiplicativeNoise',
                                multiplier=(0.8, 1.2),
                                elementwise=True,
                                p=1.0),
                            dict(type='Superpixels', p_replace=0.2, p=0.5)
                        ],
                        p=0.6)
                ]),
            dict(type='mmpretrain.PackInputs')
        ],
        serialize_data=False),
    sampler=dict(type='InfiniteSampler', shuffle=True))
cls_val_dataloader = dict(
    batch_size=36,
    num_workers=6,
    dataset=dict(
        type='mmpretrain.CustomDataset',
        data_root=
        '/home/zhangzelun/.workspaceZZL/.wdata/track1_test_data/cls/test',
        ann_file=
        '/home/zhangzelun/.workspaceZZL/.wdata/track1_train_data/cls/test/dataset-orginal-test-label.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(512, 512)),
            dict(type='mmpretrain.PackInputs')
        ],
        serialize_data=False),
    sampler=dict(type='DefaultSampler', shuffle=False))
cls_test_dataloader = dict(
    batch_size=36,
    num_workers=6,
    dataset=dict(
        type='mmpretrain.CustomDataset',
        data_root=
        '/home/zhangzelun/.workspaceZZL/.wdata/track1_test_data/cls/test',
        ann_file=
        '/home/zhangzelun/.workspaceZZL/.wdata/track1_train_data/cls/test/dataset-orginal-test-label.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(512, 512)),
            dict(type='mmpretrain.PackInputs')
        ],
        serialize_data=False),
    sampler=dict(type='DefaultSampler', shuffle=False))
train_dataloader = dict(
    mode='train',
    det=dict(
        batch_size=2,
        num_workers=6,
        persistent_workers=True,
        sampler=dict(type='InfiniteSampler', shuffle=True),
        dataset=dict(
            type='MultiImageMixDataset',
            dataset=dict(
                type='Track1DetDataset',
                data_root=
                '/home/zhangzelun/.workspaceZZL/.wdata/track1_train_data/dec/train_and_val',
                ann_file=
                '/home/zhangzelun/.workspaceZZL/.wdata/track1_train_data/dec/train_val.json',
                data_prefix=dict(img=''),
                filter_cfg=dict(filter_empty_gt=False),
                pipeline=[
                    dict(type='LoadImageFromFile', backend_args=None),
                    dict(type='LoadAnnotations', with_bbox=True)
                ],
                backend_args=None,
                serialize_data=False,
                _scope_='mmdet'),
            pipeline=[
                dict(
                    type='RandomChoice',
                    transforms=[[{
                        'type': 'Mosaic',
                        'img_scale': (1024, 1024),
                        'pad_val': 114.0
                    }, {
                        'type': 'RandomAffine',
                        'max_rotate_degree': 0,
                        'max_shear_degree': 0,
                        'max_translate_ratio': 0,
                        'scaling_ratio_range': (0.2, 1.5),
                        'border': (-512, -512)
                    }, {
                        'type': 'MixUp',
                        'img_scale': (1024, 1024),
                        'ratio_range': (0.5, 1.5),
                        'prob': 1.0,
                        'pad_val': 114.0
                    }, {
                        'type': 'YOLOXHSVRandomAug'
                    }, {
                        'type': 'Resize',
                        'scale': (1024, 1024),
                        'keep_ratio': True
                    }, {
                        'type': 'RandomFlip',
                        'prob': 0.5
                    }, {
                        'type':
                        'mmpretrain.Albu',
                        'transforms': [{
                            'type':
                            'OneOf',
                            'transforms': [{
                                'type': 'GaussNoise',
                                'var_limit': (0.0, 50.0),
                                'p': 1.0
                            }, {
                                'type': 'Blur',
                                'blur_limit': 3,
                                'p': 1.0
                            }, {
                                'type': 'Emboss',
                                'alpha': (0.2, 0.5),
                                'strength': (0.2, 0.7),
                                'p': 0.5
                            }, {
                                'type': 'GaussianBlur',
                                'blur_limit': (3, 5),
                                'sigma_limit': 0,
                                'p': 1
                            }, {
                                'type': 'MultiplicativeNoise',
                                'multiplier': (0.8, 1.2),
                                'elementwise': True,
                                'p': 1.0
                            }],
                            'p':
                            0.6
                        }]
                    }, {
                        'type': 'PackDetInputs',
                        '_scope_': 'mmdet'
                    }],
                                [{
                                    'type':
                                    'RandomChoice',
                                    'transforms': [[{
                                        'type':
                                        'RandomChoiceResize',
                                        'scales': [(800, 1024), (832, 1024),
                                                   (864, 1024), (896, 1024),
                                                   (928, 1024), (960, 1024),
                                                   (992, 1024), (1024, 1024)],
                                        'keep_ratio':
                                        True
                                    }],
                                                   [{
                                                       'type':
                                                       'RandomChoiceResize',
                                                       'scales':
                                                       [(832, 1024),
                                                        (928, 1024),
                                                        (1024, 1024)],
                                                       'keep_ratio':
                                                       True
                                                   }, {
                                                       'type': 'RandomCrop',
                                                       'crop_type':
                                                       'absolute_range',
                                                       'crop_size':
                                                       (512, 1024),
                                                       'allow_negative_crop':
                                                       True,
                                                       '_scope_': 'mmdet'
                                                   },
                                                    {
                                                        'type':
                                                        'RandomChoiceResize',
                                                        'scales':
                                                        [(800, 1024),
                                                         (832, 1024),
                                                         (864, 1024),
                                                         (896, 1024),
                                                         (928, 1024),
                                                         (960, 1024),
                                                         (992, 1024),
                                                         (1024, 1024)],
                                                        'keep_ratio':
                                                        True
                                                    }]]
                                }, {
                                    'type': 'RandomFlip',
                                    'prob': 0.5
                                }, {
                                    'type': 'RandomApply',
                                    'transforms': {
                                        'type': 'mmdet.AutoAugment'
                                    },
                                    'prob': 0.6
                                }, {
                                    'type':
                                    'mmpretrain.Albu',
                                    'transforms': [{
                                        'type':
                                        'OneOf',
                                        'transforms': [{
                                            'type': 'GaussNoise',
                                            'var_limit': (0.0, 50.0),
                                            'p': 1.0
                                        }, {
                                            'type': 'Blur',
                                            'blur_limit': 3,
                                            'p': 1.0
                                        }, {
                                            'type': 'Emboss',
                                            'alpha': (0.2, 0.5),
                                            'strength': (0.2, 0.7),
                                            'p': 0.5
                                        }, {
                                            'type': 'GaussianBlur',
                                            'blur_limit': (3, 5),
                                            'sigma_limit': 0,
                                            'p': 1
                                        }, {
                                            'type': 'MultiplicativeNoise',
                                            'multiplier': (0.8, 1.2),
                                            'elementwise': True,
                                            'p': 1.0
                                        }],
                                        'p':
                                        0.6
                                    }]
                                }, {
                                    'type': 'PackDetInputs',
                                    '_scope_': 'mmdet'
                                }]],
                    prob=[0.8, 0.2])
            ])),
    seg=dict(
        batch_size=2,
        num_workers=6,
        persistent_workers=True,
        sampler=dict(type='InfiniteSampler', shuffle=True),
        dataset=dict(
            type='mmseg.MultiImageMixDataset',
            dataset=dict(
                type='mmseg.Track1SegDataset',
                data_prefix=dict(
                    img_path=
                    '/home/zhangzelun/.workspaceZZL/.wdata/track1_train_data/seg/images/train_and_val',
                    seg_map_path=
                    '/home/zhangzelun/.workspaceZZL/.wdata/track1_train_data/seg/label/train_and_val'
                ),
                img_suffix='.jpg',
                seg_map_suffix='.png',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='mmseg.LoadAnnotations')
                ],
                serialize_data=False),
            pipeline=[
                dict(
                    type='mmseg.RandomChoiceSegTrack1',
                    transforms=[[{
                        'type': 'mmseg.RandomMosaic',
                        'prob': 1.0,
                        'img_scale': (720, 1280),
                        'center_ratio_range': (1.0, 1.0),
                        'pad_val': 0,
                        'seg_pad_val': 255
                    }, {
                        'type': 'RandomResize',
                        'scale': (1024, 1820),
                        'ratio_range': (1.0, 2.0),
                        'keep_ratio': True
                    }, {
                        'type': 'mmseg.RandomCrop',
                        'crop_size': (1024, 1024),
                        'cat_max_ratio': 0.75
                    }, {
                        'type': 'RandomFlip',
                        'prob': 0.5
                    }, {
                        'type': 'mmseg.PhotoMetricDistortion'
                    }, {
                        'type':
                        'mmpretrain.Albu',
                        'transforms': [{
                            'type':
                            'OneOf',
                            'transforms': [{
                                'type': 'GaussNoise',
                                'var_limit': (0.0, 50.0),
                                'p': 1.0
                            }, {
                                'type': 'Blur',
                                'blur_limit': 3,
                                'p': 1.0
                            }, {
                                'type': 'Emboss',
                                'alpha': (0.2, 0.5),
                                'strength': (0.2, 0.7),
                                'p': 0.5
                            }, {
                                'type': 'GaussianBlur',
                                'blur_limit': (3, 5),
                                'sigma_limit': 0,
                                'p': 1
                            }, {
                                'type': 'MultiplicativeNoise',
                                'multiplier': (0.8, 1.2),
                                'elementwise': True,
                                'p': 1.0
                            }],
                            'p':
                            0.6
                        }]
                    }, {
                        'type': 'mmseg.PackSegInputs'
                    }],
                                [{
                                    'type': 'RandomResize',
                                    'scale': (1820, 1024),
                                    'ratio_range': (0.5, 2.0),
                                    'keep_ratio': True
                                }, {
                                    'type': 'mmseg.RandomCrop',
                                    'crop_size': (1024, 1024),
                                    'cat_max_ratio': 0.75
                                }, {
                                    'type': 'RandomFlip',
                                    'prob': 0.5
                                }, {
                                    'type': 'mmseg.PhotoMetricDistortion'
                                }, {
                                    'type': 'RandomApply',
                                    'transforms': {
                                        'type': 'mmseg.RGB2Gray'
                                    },
                                    'prob': 0.1
                                }, {
                                    'type': 'mmseg.RandomRotate',
                                    'prob': 0.2,
                                    'degree': 15
                                }, {
                                    'type':
                                    'mmpretrain.Albu',
                                    'transforms': [{
                                        'type':
                                        'OneOf',
                                        'transforms': [{
                                            'type': 'GaussNoise',
                                            'var_limit': (0.0, 50.0),
                                            'p': 1.0
                                        }, {
                                            'type': 'Blur',
                                            'blur_limit': 3,
                                            'p': 1.0
                                        }, {
                                            'type': 'Emboss',
                                            'alpha': (0.2, 0.5),
                                            'strength': (0.2, 0.7),
                                            'p': 0.5
                                        }, {
                                            'type': 'GaussianBlur',
                                            'blur_limit': (3, 5),
                                            'sigma_limit': 0,
                                            'p': 1
                                        }, {
                                            'type': 'MultiplicativeNoise',
                                            'multiplier': (0.8, 1.2),
                                            'elementwise': True,
                                            'p': 1.0
                                        }],
                                        'p':
                                        0.6
                                    }]
                                }, {
                                    'type': 'mmseg.PackSegInputs'
                                }]],
                    prob=[0.4, 0.6],
                    total_iter_num=44000,
                    mosaic_shutdown_iter_ratio=[0.4],
                    mosaic_use_prob=[0.0])
            ])),
    cls=dict(
        batch_size=8,
        num_workers=6,
        dataset=dict(
            type='mmpretrain.CustomDataset',
            data_root=
            '/home/zhangzelun/.workspaceZZL/.wdata/track1_train_data/cls/train_and_val',
            ann_file=
            '/home/zhangzelun/.workspaceZZL/.wdata/track1_train_data/cls/train_and_val-amend.txt',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='Resize', scale=(512, 512)),
                dict(
                    type='RandomApply',
                    transforms=dict(
                        type='mmpretrain.RandomResizedCrop',
                        scale=(512, 512),
                        crop_ratio_range=(0.98, 1.0)),
                    prob=0.2),
                dict(
                    type='mmpretrain.RandomErasing',
                    erase_prob=0.5,
                    mode='rand',
                    min_area_ratio=0.02,
                    max_area_ratio=0.3333333333333333,
                    fill_color=[103.53, 116.28, 123.675],
                    fill_std=[57.375, 57.12, 58.395]),
                dict(type='RandomFlip', prob=0.5, direction='horizontal'),
                dict(
                    type='RandomApply',
                    transforms=dict(
                        type='mmpretrain.AutoAugment',
                        policies='imagenet',
                        hparams=dict(
                            pad_val=[104, 116, 124],
                            interpolation='bilinear')),
                    prob=0.8),
                dict(
                    type='mmpretrain.Albu',
                    transforms=[
                        dict(
                            type='ShiftScaleRotate',
                            shift_limit=0.0,
                            scale_limit=[-0.1, 0.2],
                            rotate_limit=30,
                            interpolation=1,
                            border_mode=1,
                            p=0.5),
                        dict(type='Perspective', pad_mode=1, p=0.5),
                        dict(
                            type='OneOf',
                            transforms=[
                                dict(
                                    type='GaussNoise',
                                    var_limit=(0.0, 50.0),
                                    p=1.0),
                                dict(type='Blur', blur_limit=3, p=1.0),
                                dict(
                                    type='Emboss',
                                    alpha=(0.2, 0.5),
                                    strength=(0.2, 0.7),
                                    p=0.5),
                                dict(
                                    type='GaussianBlur',
                                    blur_limit=(3, 5),
                                    sigma_limit=0,
                                    p=1),
                                dict(
                                    type='MultiplicativeNoise',
                                    multiplier=(0.8, 1.2),
                                    elementwise=True,
                                    p=1.0),
                                dict(type='Superpixels', p_replace=0.2, p=0.5)
                            ],
                            p=0.6)
                    ]),
                dict(type='mmpretrain.PackInputs')
            ],
            serialize_data=False),
        sampler=dict(type='InfiniteSampler', shuffle=True)))
val_dataloader = dict(
    mode='val',
    det=dict(
        batch_size=24,
        num_workers=8,
        persistent_workers=True,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(
            type='Track1DetDataset',
            data_root=
            '/home/zhangzelun/.workspaceZZL/.wdata/track1_test_data/dec/test',
            ann_file=
            '/home/zhangzelun/.workspaceZZL/.wdata/track1_test_data/dec/test.json',
            data_prefix=dict(img=''),
            test_mode=True,
            pipeline=[
                dict(type='LoadImageFromFile', backend_args=None),
                dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(
                    type='PackDetInputs',
                    meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                               'scale_factor'))
            ],
            backend_args=None,
            serialize_data=False,
            _scope_='mmdet')),
    seg=dict(
        batch_size=1,
        num_workers=2,
        persistent_workers=True,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(
            type='mmseg.Track1SegDataset',
            data_prefix=dict(
                img_path=
                '/home/zhangzelun/.workspaceZZL/.wdata/track1_test_data/seg/images/test',
                seg_map_path=
                '/home/zhangzelun/.workspaceZZL/.wdata/track1_test_data/seg/images/test_label'
            ),
            img_suffix='.jpg',
            seg_map_suffix='.png',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='Resize', scale=(1820, 1024), keep_ratio=True),
                dict(type='mmseg.LoadAnnotations'),
                dict(type='mmseg.PackSegInputs')
            ],
            serialize_data=False)),
    cls=dict(
        batch_size=36,
        num_workers=6,
        dataset=dict(
            type='mmpretrain.CustomDataset',
            data_root=
            '/home/zhangzelun/.workspaceZZL/.wdata/track1_test_data/cls/test',
            ann_file=
            '/home/zhangzelun/.workspaceZZL/.wdata/track1_train_data/cls/test/dataset-orginal-test-label.txt',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='Resize', scale=(512, 512)),
                dict(type='mmpretrain.PackInputs')
            ],
            serialize_data=False),
        sampler=dict(type='DefaultSampler', shuffle=False)))
test_dataloader = dict(
    mode='val',
    det=dict(
        batch_size=24,
        num_workers=8,
        persistent_workers=True,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(
            type='Track1DetDataset',
            data_root=
            '/home/zhangzelun/.workspaceZZL/.wdata/track1_test_data/dec/test',
            ann_file=
            '/home/zhangzelun/.workspaceZZL/.wdata/track1_test_data/dec/test.json',
            data_prefix=dict(img=''),
            test_mode=True,
            pipeline=[
                dict(type='LoadImageFromFile', backend_args=None),
                dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(
                    type='PackDetInputs',
                    meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                               'scale_factor'))
            ],
            backend_args=None,
            serialize_data=False,
            _scope_='mmdet')),
    seg=dict(
        batch_size=1,
        num_workers=2,
        persistent_workers=True,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(
            type='mmseg.Track1SegDataset',
            data_prefix=dict(
                img_path=
                '/home/zhangzelun/.workspaceZZL/.wdata/track1_test_data/seg/images/test',
                seg_map_path=
                '/home/zhangzelun/.workspaceZZL/.wdata/track1_test_data/seg/images/test_label'
            ),
            img_suffix='.jpg',
            seg_map_suffix='.png',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='Resize', scale=(1820, 1024), keep_ratio=True),
                dict(type='mmseg.LoadAnnotations'),
                dict(type='mmseg.PackSegInputs')
            ],
            serialize_data=False)),
    cls=dict(
        batch_size=36,
        num_workers=6,
        dataset=dict(
            type='mmpretrain.CustomDataset',
            data_root=
            '/home/zhangzelun/.workspaceZZL/.wdata/track1_test_data/cls/test',
            ann_file=
            '/home/zhangzelun/.workspaceZZL/.wdata/track1_train_data/cls/test/dataset-orginal-test-label.txt',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='Resize', scale=(512, 512)),
                dict(type='mmpretrain.PackInputs')
            ],
            serialize_data=False),
        sampler=dict(type='DefaultSampler', shuffle=False)))
det_val_evaluator = dict(
    type='mmdet.CocoMetric',
    ann_file=
    '/home/zhangzelun/.workspaceZZL/.wdata/track1_test_data/dec/test.json',
    classwise=True,
    metric='bbox',
    format_only=False,
    backend_args=None)
det_test_evaluator = dict(
    type='mmdet.CocoMetric',
    ann_file=
    '/home/zhangzelun/.workspaceZZL/.wdata/track1_test_data/dec/test.json',
    classwise=True,
    metric='bbox',
    format_only=False,
    backend_args=None)
seg_val_evaluator = dict(type='mmseg.IoUMetric', iou_metrics=['mIoU'])
seg_test_evaluator = dict(type='mmseg.IoUMetric', iou_metrics=['mIoU'])
cls_val_evaluator = dict(type='mmpretrain.Accuracy', topk=(1, 5))
cls_test_evaluator = dict(type='mmpretrain.Accuracy', topk=(1, 5))
val_evaluator = dict(
    det=dict(
        type='mmdet.CocoMetric',
        ann_file=
        '/home/zhangzelun/.workspaceZZL/.wdata/track1_test_data/dec/test.json',
        classwise=True,
        metric='bbox',
        format_only=False,
        backend_args=None),
    seg=dict(type='mmseg.IoUMetric', iou_metrics=['mIoU']),
    cls=dict(type='mmpretrain.Accuracy', topk=(1, 5)))
test_evaluator = dict(
    det=dict(
        type='mmdet.CocoMetric',
        ann_file=
        '/home/zhangzelun/.workspaceZZL/.wdata/track1_test_data/dec/test.json',
        classwise=True,
        metric='bbox',
        format_only=False,
        backend_args=None),
    seg=dict(type='mmseg.IoUMetric', iou_metrics=['mIoU']),
    cls=dict(type='mmpretrain.Accuracy', topk=(1, 5)))
det_data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=1,
    _scope_='mmdet')
seg_data_preprocessor = dict(
    type='mmseg.SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=(1024, 1024))
cls_data_preprocessor = dict(
    type='mmpretrain.ClsDataPreprocessor',
    num_classes=196,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)
data_preprocessor = dict(
    type='mmdet.MultiTaskDataPreprocessor',
    det=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1,
        _scope_='mmdet'),
    seg=dict(
        type='mmseg.SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=(1024, 1024)),
    cls=dict(
        type='mmpretrain.ClsDataPreprocessor',
        num_classes=196,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True))
model = dict(
    type='DINOMultiTask',
    num_queries=900,
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor=dict(
        type='mmdet.MultiTaskDataPreprocessor',
        det=dict(
            type='DetDataPreprocessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            bgr_to_rgb=True,
            pad_size_divisor=1,
            _scope_='mmdet'),
        seg=dict(
            type='mmseg.SegDataPreProcessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            bgr_to_rgb=True,
            pad_val=0,
            seg_pad_val=255,
            size=(1024, 1024)),
        cls=dict(
            type='mmpretrain.ClsDataPreprocessor',
            num_classes=196,
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True)),
    backbone=dict(
        type='InternImage',
        core_op='DCNv3',
        channels=192,
        depths=[5, 5, 24, 5],
        groups=[12, 24, 48, 96],
        mlp_ratio=4.0,
        drop_path_rate=0.4,
        norm_layer='LN',
        layer_scale=1.0,
        offset_scale=2.0,
        post_norm=True,
        with_cp=False,
        out_indices=(0, 1, 2, 3),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            '/home/zhangzelun/.workspaceZZL/pretrain/internimage_xl_22kto1k_384_.pth',
            map_location='cpu')),
    neck=dict(
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
        embed_dims=256,
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
            ])),
    test_cfg=dict(max_per_img=210),
    num_feature_levels=5,
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
            encoder=dict(
                num_layers=6,
                layer_cfg=dict(
                    self_attn_cfg=dict(
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
            positional_encoding=dict(num_feats=128, normalize=True),
            init_cfg=None),
        enforce_decoder_input_project=False,
        positional_encoding=dict(num_feats=128, normalize=True),
        transformer_decoder=dict(
            return_intermediate=True,
            num_layers=9,
            layer_cfg=dict(
                self_attn_cfg=dict(
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=True),
                cross_attn_cfg=dict(
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
            class_weight=[
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1
            ]),
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
    seg_test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(512, 512)),
    seg_align_corners=False,
    cls_neck=dict(
        type='mmpretrain.MultiLinearNeck',
        in_channels=1536,
        latent_channels=[('bn', 1536), ('fc', 1024), ('bn', 1024),
                         ('relu', 1024), ('fc', 1536), ('bn', 1536)],
        use_shortcut=True,
        init_cfg=dict(
            type='TruncNormal', layer=['Conv2d', 'Linear'], std=0.02,
            bias=0.0)),
    cls_head=dict(
        type='mmpretrain.ArcFaceClsHead',
        num_classes=196,
        in_channels=1536,
        num_subcenters=1,
        loss=dict(type='mmpretrain.CrossEntropyLoss', loss_weight=1.0)))
train_cfg = dict(type='IterBasedTrainLoop', max_iters=44000, val_interval=440)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
base_lr = 0.0002
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.005),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys=dict(
            backbone=dict(lr_mult=0.1),
            norm=dict(decay_mult=0.0),
            pos_block=dict(decay_mult=0.0),
            query_embed=dict(lr_mult=1.0, decay_mult=0.0),
            query_feat=dict(lr_mult=1.0, decay_mult=0.0),
            level_embed=dict(lr_mult=1.0, decay_mult=0.0),
            level_embedding=dict(lr_mult=1.0, decay_mult=0.0),
            cls_token=dict(lr_mult=1.0, decay_mult=0.0),
            posi_embedding=dict(lr_mult=1.0, decay_mult=0.0))))
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        end_factor=1.0,
        begin=0,
        end=2200,
        by_epoch=False),
    dict(
        type='CosineAnnealingLR',
        eta_min=2.0000000000000002e-07,
        begin=2200,
        end=44000,
        by_epoch=False)
]
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=440),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')
log_processor = dict(type='LogProcessor', window_size=10, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False
fp16 = dict(loss_scale=512.0)
work_dir = '/mnt/sfs_turbo_jiaob/RetailVision/softwares/panxue_custom/.software_workdirs/cudnn/multi-task_dino_mask2former_internimage-b_st18'
sync_bn = 'torch'
find_unused_parameters = True
custom_hooks = [
    dict(
        type='mmpretrain.SetAdaptiveMarginsHook',
        margin_min=0.5,
        margin_max=0.55)
]
launcher = 'pytorch'
