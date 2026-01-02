# mmpretrain config for TinyNet with Mixup on ImageWoof dataset

# Model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='TIMMBackbone',
        model_name='tinynet_e.in1k',
        pretrained=True,
        features_only=False,
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=10,
        in_channels=1280,  # TinyNet-E outputs 1280 channels
        loss=dict(
            type='CrossEntropyLoss',
            loss_weight=1.0,
        ),
        topk=(1, 5),
    ),
    train_cfg=dict(
        augments=[
            dict(type='Mixup', alpha=0.2),
        ]
    )
)

# Dataset settings
dataset_type = 'CustomDataset'
data_preprocessor = dict(
    num_classes=10,
    mean=[123.675, 116.28, 103.53],  # ImageNet mean (in BGR order)
    std=[58.395, 57.12, 57.375],     # ImageNet std (in BGR order)
    to_rgb=True,
)

# Training pipeline with Mixup
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224, backend='pillow', interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies='timm_increasing',
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5,
        hparams=dict(pad_val=[104, 116, 124], interpolation='bicubic')
    ),
    dict(
        type='RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=1/3,
        fill_color=[103.53, 116.28, 123.675],
        fill_std=[57.375, 57.12, 58.395]
    ),
    dict(type='PackInputs'),
]

# Validation pipeline
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=256, edge='short', backend='pillow', interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root='/Users/jeremyong/Desktop/research_agent/dataset/imagewoof-160/train',
        pipeline=train_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=True),
    persistent_workers=True,
)

val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root='/Users/jeremyong/Desktop/research_agent/dataset/imagewoof-160/val',
        pipeline=test_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True,
)

val_evaluator = dict(type='Accuracy', topk=(1, 5))

test_dataloader = val_dataloader
test_evaluator = val_evaluator

# Schedule settings (override _base_)
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=1e-3,
        weight_decay=0.01,
    ),
    clip_grad=dict(max_norm=1.0),
)

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=10,
        eta_min=1e-5,
        by_epoch=True,
        begin=0,
        end=10,
    )
]

# Training settings
train_cfg = dict(
    by_epoch=True,
    max_epochs=10,
    val_interval=1,
)

val_cfg = dict()
test_cfg = dict()

# Runtime settings
default_scope = 'mmpretrain'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=3,
        save_best='auto',
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='VisualizationHook', enable=False),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='imagewoof-160',
            name='tinynet_mixup_mmpretrain',
        )
    )
]

visualizer = dict(
    type='Visualizer',
    vis_backends=vis_backends
)

log_level = 'INFO'
load_from = None
resume = False

# Random seed
randomness = dict(seed=42, deterministic=False)

