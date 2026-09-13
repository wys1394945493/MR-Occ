import os

_base_ = "mmdet3d::_base_/default_runtime.py"

dataset_type = "NuScenesDatasetSurroundOcc"
data_root    = '/c20250502/wangyushen/Datasets/NuScenes/v1.0-trainval/'
ann_root     = '/c20250502/wangyushen/Datasets/NuScenes/method/superocc/'
occ_gt       = '/c20250502/wangyushen/Datasets/'

backbone_repo = os.getenv("DINOV3_REPO", "dinov3")
backbone_checkpoint = "/c20250502/wangyushen/Weights/dvgt/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"


max_epochs = 1
val_interval = 1

batch_size = 8
num_workers = 8
train_ann = ann_root + "nuscenes_infos_train_sweep.pkl"
val_ann = ann_root + "nuscenes_infos_val_sweep.pkl"
load_interval = 1
logger_interval = 50


seed = 0

point_cloud_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
voxel_size = [0.5, 0.5, 0.5]
scale_range = [0.01, 3.2]
u_range = [0.1, 2]
v_range = [0.1, 2]

embed_dims = 256
num_layers = 6
num_query = 1200

num_frames = 1

num_levels = 4
num_points = 2
num_refines = [2, 2, 4, 4, 8, 8]


noise_scale = 0.5
alpha = 1.0

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)

occ_names = [
    "noise",
    "barrier",
    "bicycle",
    "bus",
    "car",
    "construction_vehicle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "trailer",
    "truck",
    "driveable_surface",
    "other_flat",
    "sidewalk",
    "terrain",
    "manmade",
    "vegetation",
]

use_ego = False
ignore_label = 255


manual_class_weight = [
    1.01552756,
    1.06897009,
    1.30013094,
    1.07253735,
    0.94637502,
    1.10087012,
    1.26960524,
    1.06258364,
    1.189019,
    1.06217292,
    1.00595144,
    0.85706115,
    1.03923299,
    0.90867526,
    0.8936431,
    0.85486129,
    0.8527829,
    0.5,
]

class_counts = [
    12029675,
    4992563,
    301900,
    4749134,
    40784501,
    3270178,
    402736,
    5792502,
    1040733,
    5995197,
    13754976,
    258417482,
    7340707,
    82964921,
    113713395,
    271945241,
    284762437,
]


model = dict(
    type="MROcc",
    backbone_repo=backbone_repo,
    calibration_stage=True,
    data_aug=dict(
        img_color_aug=False,
        img_norm_cfg=img_norm_cfg,
        img_pad_cfg=dict(size_divisor=32),
    ),
    num_levels=num_levels,
    out_dim=embed_dims,
    dim_in=1024,
    patch_embed="dinov3_vitl16",
    model_url=backbone_checkpoint,
    pts_bbox_head=dict(
        type="MROccHead",
        calibration_stage=True,
        num_classes=len(occ_names),
        in_channels=embed_dims,
        num_query=num_query,
        scale_range=scale_range,
        u_range=u_range,
        v_range=v_range,
        pc_range=point_cloud_range,
        voxel_size=voxel_size,
        manual_class_weight=manual_class_weight,
        ignore_label=ignore_label,
        noise_scale=noise_scale,
        alpha=alpha,
        transformer=dict(
            type="MROccTransformer",
            calibration_stage=True,
            embed_dims=embed_dims,
            num_frames=num_frames,
            num_points=num_points,
            num_layers=num_layers,
            num_levels=num_levels,
            num_classes=len(occ_names),
            num_refines=num_refines,
            pc_range=point_cloud_range,
            use_ego=use_ego,
        ),
        loss_occ=dict(
            type="LabelAwareCELoss",
            activated=True,
            loss_weight=10.0,
            label_aware_smoothing=True,
            class_counts=class_counts,
            alpha_head=0.05,
            alpha_tail=0.0,
            las_form="linear",
            empty_label=17,
            smooth_empty=False,
        ),
        loss_pts=dict(type="mmdet.SmoothL1Loss", beta=0.2, loss_weight=0.5),
    ),
)


object_names = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]
collect_keys = [
    "lidar2img",
    "intrinsics",
    "extrinsics",
    "timestamp",
    "img_timestamp",
    "ego_pose",
    "ego_pose_inv",
]

input_modality = dict(
    use_lidar=False, use_camera=True, use_radar=False, use_map=False, use_external=True
)

ida_aug_conf = {
    "resize_lim": (0.38, 0.55),
    "final_dim": (256, 704),
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 900,
    "W": 1600,
    "rand_flip": True,
}

train_pipeline = [
    dict(
        type="CustomLoadMultiViewImageFromFiles", to_float32=False, color_type="color"
    ),
    dict(type="LoadOccupancySurroundOcc"),
    dict(type="RandomTransformImage", ida_aug_conf=ida_aug_conf, training=True),
    dict(
        type="CustomFormatBundle3D", class_names=object_names, collect_keys=collect_keys
    ),
    dict(
        type="Collect3D",
        keys=["img", "voxel_semantics", "mask_camera"] + collect_keys,
        meta_keys=(
            "filename",
            "occ_gt_path",
            "ori_shape",
            "img_shape",
            "scale_factor",
            "flip",
            "scene_token",
            "token",
        ),
    ),
]

test_pipeline = [
    dict(
        type="CustomLoadMultiViewImageFromFiles", to_float32=False, color_type="color"
    ),
    dict(type="LoadOccupancySurroundOcc"),
    dict(type="RandomTransformImage", ida_aug_conf=ida_aug_conf, training=False),
    dict(
        type="CustomFormatBundle3D", class_names=object_names, collect_keys=collect_keys
    ),
    dict(
        type="Collect3D",
        keys=["img", "voxel_semantics", "mask_camera"] + collect_keys,
        meta_keys=(
            "filename",
            "occ_gt_path",
            "ori_shape",
            "img_shape",
            "scale_factor",
            "flip",
            "scene_token",
            "token",
        ),
    ),
]

train_dataset_config = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file=train_ann,
    occ_gt=occ_gt,
    pipeline=train_pipeline,
    modality=input_modality,
    test_mode=False,
    load_interval=load_interval,
)

val_dataset_config = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file=val_ann,
    occ_gt=occ_gt,
    pipeline=test_pipeline,
    modality=input_modality,
    test_mode=True,
    load_interval=load_interval,
)


train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True if num_workers > 0 else False,
    pin_memory=True,
    sampler=dict(type="DefaultSampler", shuffle=True, seed=seed),
    collate_fn=dict(type="custom_collate_fn"),
    dataset=train_dataset_config,
)

val_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True if num_workers > 0 else False,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False, seed=seed),
    collate_fn=dict(type="custom_collate_fn"),
    dataset=val_dataset_config,
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type="ECEMetricDual",
    class_indices=list(range(1, 17)),
    empty_label=17,
    label_str=[
        "barrier",
        "bicycle",
        "bus",
        "car",
        "cons.veh",
        "motorcycle",
        "pedestrian",
        "traffic_cone",
        "trailer",
        "truck",
        "drive.surf",
        "other_flat",
        "sidewalk",
        "terrain",
        "manmade",
        "vegetation",
    ],
    dataset_empty_label=17,
    filter_minmax=False,
    num_ece_bins=10,
    use_softmax_for_ece=False,
    calibration_save_dir=None,
)
test_evaluator = val_evaluator

randomness = dict(
    seed=seed,
    deterministic=False,
)

train_cfg = dict(
    type="EpochBasedTrainLoop",
    max_epochs=max_epochs,
    val_interval=val_interval,
)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=1e-4, weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            "cls_scale": dict(lr_mult=1.0, decay_mult=0.0),
            "cls_delta_weight": dict(lr_mult=1.0, decay_mult=1.0),
            "cls_delta_bias": dict(lr_mult=1.0, decay_mult=0.0),
        }
    ),
    clip_grad=dict(max_norm=35, norm_type=2),
)


param_scheduler = [
    dict(
        type="MultiStepLR",
        by_epoch=True,
        milestones=[5],
        gamma=0.1,
    )
]

default_hooks = dict(
    logger=dict(
        type="LoggerHook",
        interval=logger_interval,
    ),
    checkpoint=dict(
        type="CheckpointHook",
        interval=1,
        max_keep_ckpts=1,
    ),
)

log_processor = dict(
    type="LogProcessor",
    window_size=50,
    by_epoch=True,
    mean_pattern=r".*(time|data_time).*",
)
model_wrapper_cfg = dict(
    type="MMDistributedDataParallel",
    find_unused_parameters=True,
)
