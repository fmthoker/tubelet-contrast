dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
syncbn = True

work_dir = './output/moco/r3d_18_mini_kinetics/pretraining/'


model = dict(
    type='MoCo',
    backbone=dict(
        type='R3D',
        depth=18,
        num_stages=4,
        stem=dict(
            temporal_kernel_size=3,
            temporal_stride=1,
            in_channels=3,
            with_pool=False,
        ),
        down_sampling=[False, True, True, True],
        channel_multiplier=1.0,
        bottleneck_multiplier=1.0,
        with_bn=True,
        pretrained=None,
    ),
    in_channels=512,
    out_channels=128,
    queue_size=16384,
    momentum=0.999,
    temperature=0.20,
    mlp=True,
    network='resnet'
)



data = dict(
    videos_per_gpu=8,  # total batch size is 8Gpus*4 == 32
    workers_per_gpu=4,
    train=dict(
        type='MoCoDataset',
        data_source=dict(
            type='JsonClsDataSource',
            ann_file='kinetics/annotations_mini_kinetics/train_split_1.json',

        ),
        backend=dict(
            type='ZipBackend',
            zip_fmt='kinetics/zips/{}.zip',
            frame_fmt='img_{:05d}.jpg',
        ),
        frame_sampler=dict(
            type='RandomFrameSampler',
            num_clips=1,
            clip_len=16,
            strides=[1, 2, 3, 4],
            temporal_jitter=True
        ),
        transform_cfg=[
            dict(type='GroupScale', scales=[(149, 112), (171, 128), (192, 144)]),
            dict(type='GroupFlip', flip_prob=0.5),
            dict(type='GroupRandomCrop', out_size=112),
        ],
        transform_cfg_2=[
            dict(
                type='Tubelets',
                region_sampler=dict(
                    scales=[16, 24, 28, 32, 48, 64],
                    ratios=[0.5, 0.67, 0.75, 1.0, 1.33, 1.50, 2.0],
                    scale_jitter=0.18,
                    num_rois=2,
                ),
                key_frame_probs=[0.5, 0.3, 0.2],
                loc_velocity=5,
                rot_velocity=6,
                shear_velocity=0.066,
                size_velocity=0.0001,
                label_prob=1.0,
                motion_type='gaussian',
                patch_transformation='rotation',
            ),
            dict(type='RandomBrightness', prob=0.20, delta=32),
            dict(type='RandomContrast', prob=0.20, delta=0.20),
            dict(type='RandomHueSaturation', prob=0.20, hue_delta=12, saturation_delta=0.1),
            dict(
                type='GroupToTensor',
                switch_rgb_channels=True,
                div255=True,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        ],
        test_mode=False
    )
)

# optimizer
total_epochs = 200
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0.)

checkpoint_config = dict(interval=10, max_keep_ckpts=30, create_symlink=False)
workflow = [('train', 1)]
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ]
)
