# model settings
_base_ = [
    '../_base_/models/hv_pointpillars_secfpn_carla.py',
    '../_base_/datasets/carla-3d-car-bench6.py',
    '../_base_/schedules/cyclic_40e.py', '../_base_/default_runtime.py'
]

point_cloud_range = [-101.84, -59.52, -0.1, 1.84, 59.52, 3.9]
model = dict(
    bbox_head=dict(
        type="Anchor3DHead",
        num_classes=1,
        anchor_generator=dict(
            _delete_=True,
            type="AlignedAnchor3DRangeGenerator",
            ranges=[[-101.84, -59.52, 0.72, 1.84, 59.52, 0.72]],
            sizes=[[3.9, 1.6, 1.56]],
            rotations=[0, 1.57],
            reshape_out=True,
        )
    ),
    # model training and testing settings
    train_cfg=dict(
        _delete_=True,
        assigner=dict(
            type="MaxIoUAssigner",
            iou_calculator=dict(type="BboxOverlapsNearest3D"),
            pos_iou_thr=0.6,
            neg_iou_thr=0.45,
            min_pos_iou=0.45,
            ignore_iof_thr=-1,
        ),
        allowed_border=0,
        pos_weight=-1,
        debug=False,
    ),
)

# dataset settings
dataset_type = "CarlaDataset"
data_root = "data/carla_bench/c32h80_bench"
class_names = ["Car"]
db_sampler = dict(
    data_root=data_root,
    info_path=data_root + "/carla_dbinfos_train.pkl",
    rate=1.0,
    prepare=dict(filter_by_difficulty=[-1], filter_by_min_points=dict(Car=5)),
    sample_groups=dict(Car=15),
    classes=class_names,
)

train_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=4,
        use_dim=4,
    ),
    dict(
        type="LoadAnnotations3D",
        with_bbox_3d=True,
        with_label_3d=True,
    ),
    dict(
        type="ObjectSample",
        db_sampler=db_sampler,
        use_ground_plane=True,
    ),
    dict(
        type="RandomFlip3D",
        flip_ratio_bev_horizontal=0.5,
    ),
    dict(
        type="GlobalRotScaleTrans",
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05],
    ),
    dict(
        type="PointsRangeFilter",
        point_cloud_range=point_cloud_range,
    ),
    dict(
        type="ObjectRangeFilter",
        point_cloud_range=point_cloud_range,
    ),
    dict(type="PointShuffle"),
    dict(
        type="DefaultFormatBundle3D",
        class_names=class_names,
    ),
    dict(
        type="Collect3D",
        keys=["points", "gt_bboxes_3d", "gt_labels_3d"],
    ),
]
test_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=4,
        use_dim=4,
    ),
    dict(
        type="MultiScaleFlipAug3D",
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type="GlobalRotScaleTrans",
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0],
            ),
            dict(type="RandomFlip3D"),
            dict(
                type="PointsRangeFilter",
                point_cloud_range=point_cloud_range,
            ),
            dict(
                type="DefaultFormatBundle3D",
                class_names=class_names,
                with_label=False,
            ),
            dict(type="Collect3D", keys=["points"]),
        ],
    ),
]

data = dict(
    samples_per_gpu=6,
    workers_per_gpu=4,
    train=dict(
        type="RepeatDataset",
        times=2,
        dataset=dict(pipeline=train_pipeline, classes=class_names)
    ),
    val=dict(pipeline=test_pipeline, classes=class_names),
    test=dict(pipeline=test_pipeline, classes=class_names),
)

# optimizer
lr = 0.01  # max learning rate
optimizer = dict(lr=lr, betas=(0.95, 0.85))
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=150)
evaluation = dict(interval=1)
