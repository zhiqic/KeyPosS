_base_ = [
    '../../../../_base_/default_runtime.py',
    '../../../../_base_/datasets/cofw.py'
]
checkpoint_config = dict(interval=50)
evaluation = dict(interval=5, metric=['NME'], save_best='NME')

optimizer = dict(
    type='Adam',
    lr=2e-3,
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[40, 55])

#v3
# optimizer = dict(
#     type='Adam', #Adam AdaMod
#     lr=0.002,
# )
# optimizer_config = dict(grad_clip=None)
# lr_config = dict(
#     policy='CosineAnnealing',
#     min_lr=0.00001,
#     warmup='linear',
#     warmup_iters=10,
#     warmup_ratio=0.001,
#     warmup_by_epoch=True,
#     by_epoch=True)


total_epochs = 100
log_config = dict(
    interval=5,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

channel_cfg = dict(
    num_output_channels=29,
    dataset_joints=29,
    dataset_channel=[
        list(range(29)),
    ],
    inference_channel=list(range(29)))

data_cfg = dict(
    image_size=[256, 256],
    heatmap_size=[64, 64], ##--------------------------------
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'])

in_channels=[18, 36, 72, 144]
in_index=(0, 1, 2, 3)
if data_cfg['heatmap_size'][0] ==8:
    in_channels=[18, 36, 72]
    in_index=(0, 1, 2)
if data_cfg['heatmap_size'][0] ==4:
    in_channels=[18, 36]
    in_index=(0, 1)

# model settings
model = dict(
    type='TopDown',
    # pretrained='open-mmlab://msra/hrnetv2_w18',
    pretrained='/home/shouzhou.bx/workspace/face_landmark/mmpose_pretrained/hrnetv2_w18/hrnetv2_w18-00eb2006.pth',
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(18, 36)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(18, 36, 72)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(18, 36, 72, 144),
                multiscale_output=True),
            upsample=dict(mode='bilinear', align_corners=False),
            data_cfg=data_cfg )),
    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=in_channels,  # # #-------------2023-04-23------------
        in_index=in_index,
        input_transform='resize_concat',
        out_channels=channel_cfg['num_output_channels'],
        num_deconv_layers=0,
        extra=dict(
            final_conv_kernel=1, num_conv_layers=1, num_conv_kernels=(1, ), data_cfg=data_cfg),
        loss_keypoint=dict(type='HuberLoss', use_target_weight=True, delta=1.0)),

        # loss_keypoint=dict(type='WingLoss_tril', omega=5.0, epsilon=1.0, loss_weight=1., use_target_weight=True)),
        # loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='default',
        shift_heatmap=True,
        modulate_kernel=11))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownGetBboxCenterScale', padding=1.25),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=30,
        scale_factor=0.25),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTarget', sigma=1.5),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight', 'joints_3d', 'center', 'scale'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'flip_pairs'
        ]),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownGetBboxCenterScale', padding=1.25),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=['image_file', 'center', 'scale', 'rotation', 'flip_pairs']),
]

test_pipeline = val_pipeline

# data_root = 'data/cofw'
data_root = '/home/shouzhou.bx/workspace/face_landmark/face_landmark_data/cofw/'

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=32),
    test_dataloader=dict(samples_per_gpu=32),
    train=dict(
        type='FaceCOFWDataset',
        ann_file=f'{data_root}/annotations/cofw_train.json',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
    val=dict(
        type='FaceCOFWDataset',
        ann_file=f'{data_root}/annotations/cofw_test_sub.json',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='FaceCOFWDataset',
        ann_file=f'{data_root}/annotations/cofw_test_sub.json',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}}),
)
