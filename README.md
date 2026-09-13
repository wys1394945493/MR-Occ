# MR-Occ

Official implementation of **MR-Occ: Multi-Path Representation Learning for
3D Semantic Occupancy Prediction**.

## Installation

The reference environment uses Python 3.10, PyTorch 2.1.1, torchvision 0.16.1,
and CUDA 12.1.

```bash
conda create -n mrocc python=3.10 -y
conda activate mrocc

pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Compile the two custom CUDA operators from the repository root:

```bash
cd ops/msmv_sampling && python setup.py build_ext --inplace
cd ../..
cd ops/superquadric_splatting && python setup.py build_ext --inplace
```

The CUDA toolkit used to compile the operators should be compatible with the
installed PyTorch build. Recompile both operators after changing PyTorch or
CUDA versions.

## Data preparation

Download the [nuScenes dataset](https://www.nuscenes.org/nuscenes) and the
semantic occupancy labels provided by
[SurroundOcc](https://github.com/weiyithu/SurroundOcc).

Download the pre-generated annotation files:

- [nuscenes_infos_train_sweep.pkl]()
- [nuscenes_infos_val_sweep.pkl]()

Place both files in the annotation directory and set `ann_root` in the configuration files to that directory.

```text
<NUSCENES_ROOT>/
├── samples/
├── sweeps/
├── maps/
└── v1.0-trainval/

<ANNOTATION_ROOT>/
├── nuscenes_infos_train_sweep.pkl
└── nuscenes_infos_val_sweep.pkl

<OCCUPANCY_ROOT>/
└── surroundocc/
    └── samples/
        └── *.npy
```

## Training

All commands below are run from the repository root.

### Stage 1: Representation Learning

```bash
python train.py configs/mr_occ_stage1.py \
    --work-dir work_dirs/mr_occ_stage1
```

For distributed training:

```bash
torchrun --nproc_per_node=4 train.py configs/mr_occ_stage1.py \
    --launcher pytorch \
    --work-dir work_dirs/mr_occ_stage1
```

### Stage 2: Classifier Calibration

Initialize Stage 2 from a Stage 1 checkpoint:

```bash
python train.py configs/mr_occ_stage2.py \
    --load-from /path/to/stage1_checkpoint.pth \
    --work-dir work_dirs/mr_occ_stage2
```

## Evaluation

```bash
python test.py configs/mr_occ_stage2.py \
    --checkpoint /path/to/stage2_checkpoint.pth \
    --work-dir work_dirs/mr_occ_stage2_eval
```

To evaluate the Stage 1 model directly, use `configs/mr_occ_stage1.py` and its
corresponding checkpoint instead.

## Acknowledgements

This repository mainly builds upon ideas and code from
[GaussTR](https://github.com/YkiWu/GaussTR) and
[SuperOcc](https://github.com/Daniel-xsy/SuperOcc), and uses
[DINOv3](https://github.com/facebookresearch/dinov3) as the image backbone.
We thank their authors for making their work publicly available.

## License

The original MR-Occ code is released under the [MIT License](LICENSE).
Third-party code retains its original copyright notices and is governed by
its respective license. In particular, the bundled DINOv3 source is subject
to the [DINOv3 License Agreement](dinov3/LICENSE.md),
while `ops/superquadric_splatting` is limited to non-commercial research and
evaluation use under its bundled
[license](ops/superquadric_splatting/LICENSE.md).
