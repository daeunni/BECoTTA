# dataset settings
dataset_type = 'ACDCDataset'
data_root = 'your data path'   

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (1024, 1024)     
train_pipeline = [
    dict(type='LoadImageFromFile_wDomain'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1920, 1080), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]


test_pipeline = [
    dict(type='LoadImageFromFile_wDomain'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1920//2, 1080//2),
        img_ratios=[0.5, 1.0, 1.5, 2.0],  
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]


data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=500,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            img_dir='leftImg8bit/train',
            ann_dir='gtFine/train',
            pipeline=train_pipeline)),
    
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='rgb_anon/fog/train',
        ann_dir='gt/fog/train',
        pipeline=test_pipeline),
    
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='rgb_anon/fog/train',
        ann_dir='gt/fog/train',
        pipeline=test_pipeline),
    
    test1=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='rgb_anon/night/train',
            ann_dir='gt/night/train',
        pipeline=test_pipeline),
    
    test2=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='rgb_anon/rain/train',
        ann_dir='gt/rain/train',
        pipeline=test_pipeline),
    
    test3=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='rgb_anon/snow/train',
        ann_dir='gt/snow/train',
        pipeline=test_pipeline),
    
    )
