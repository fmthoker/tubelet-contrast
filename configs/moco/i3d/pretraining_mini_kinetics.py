dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
syncbn = True

work_dir = './output/moco/i3d_mini_kinetics/pretraining/'

model = dict(
    type='MoCo',
    backbone=dict(
        type='I3D',
    ),
    in_channels=1024,
    out_channels=128,
    #queue_size=65536,
    queue_size=16384,
    momentum=0.999,
    temperature=0.20,
    mlp=False,
    network='i3d'
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
            dict(type='GroupScale', scales=[(298, 224), (342, 256), (384, 288)]),
            dict(type='GroupFlip', flip_prob=0.5),
            dict(type='GroupRandomCrop', out_size=224),
        ],
        transform_cfg_2=[
            dict(
                type='Tubelets',
                region_sampler=dict(
                    scales=[32, 48, 56, 64, 96, 128],
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
