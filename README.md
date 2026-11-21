# Weed-YOLOv12: UAV-based Weed Detection with YOLOv12

Weed-YOLOv12 brings the DEAL-YOLO modifications, originally built on YOLOv8, to the
[YOLOv12](https://github.com/sunsmarterjie/yolov12) (turbo) baseline for small-weed detection in UAV images
(USU-Corn-WeedDB). The code base is the official YOLOv12 repository with four additions taken from
[DEAL-YOLO](https://github.com/IshaanGakhar/DEAL-YOLO).

## Key Features

1. **Additional P2 detection head, P5 head removed.** Detection runs at P2/P3/P4, which is a 160×160 / 80×80 / 40×40
   grid at 640 input. The P3 neck feature is upsampled, concatenated with the early backbone P2 feature, and refined by
   a `C3k2` block before the P2 head. The P5 head and its bottom-up neck layers are removed.
2. **Scale Sequence Feature Fusion (SSFF).** Before each head, the three neck outputs (P2, P3, P4) are resized to
   that head's resolution, stacked along a scale axis, fused with a 3D convolution, and added to that head's input.
   Each head therefore sees all three scales.
3. **Linear Deformable Convolution (LDConv).** Every stride-2 `Conv` in the backbone and neck is replaced by `LDConv`.
   It keeps the same input and output size but learns where to sample.
4. **NWD + Wise-IoU box loss.** The CIoU box loss is replaced by `iou_ratio * WIoU-v3 + (1 - iou_ratio) * (1 - NWD)`.
   The balance (`iou_ratio`) is a training argument, so it can be chosen in the ablation study.

| Component | Where | Origin |
| --- | --- | --- |
| P2 head / no P5 head | `ultralytics/cfg/models/weed-yolov12/*.yaml` | DEAL-YOLO `yolov8-ASF-P2P2.yaml` |
| SSFF (`ScalSeq`, `Add`) | `ultralytics/nn/extra_modules/block.py` | DEAL-YOLO `nn/extra_modules/block.py` |
| `LDConv` | `ultralytics/nn/modules/conv.py` | DEAL-YOLO `nn/modules/conv.py` |
| `wasserstein_loss`, `WiseIouLoss` | `ultralytics/utils/metrics.py` | DEAL-YOLO `utils/metrics.py` |
| Configurable `BboxLoss` | `ultralytics/utils/loss.py` | DEAL-YOLO `utils/loss.py` |
| Loss arguments `iou_type`, `nwd`, `iou_ratio`, `nwd_constant` | `ultralytics/cfg/default.yaml` | new (DEAL-YOLO hard-codes them) |
| Module registration | `ultralytics/nn/tasks.py` (`parse_model`) | DEAL-YOLO `nn/tasks.py` |

## Model Architecture

```
Backbone (YOLOv12-turbo, stride-2 Conv -> LDConv)
  0 LDConv  P1/2   1 LDConv  P2/4   2 C3k2 (P2 feature)
  3 LDConv  P3/8   4 C3k2           5 LDConv P4/16   6 A2C2f (area attention)
  7 LDConv  P5/32  8 A2C2f (area attention)
Neck
  top-down : P5 -> up + P4 -> A2C2f(11) -> up + P3 -> A2C2f(14) -> up + P2 -> C3k2(17)            [P2 out]
  bottom-up: 17 -> LDConv s2 + 14 -> A2C2f(20) [P3 out] -> LDConv s2 + 11 -> A2C2f(23)            [P4 out]
SSFF + Add (one per head)
  ScalSeq[17,20,23]@P2 + 17 -> 25    ScalSeq[20,17,23]@P3 + 20 -> 27    ScalSeq[23,17,20]@P4 + 23 -> 29
Detect(25, 27, 29)  -> strides 4 / 8 / 16
```

## Model Configs

All configs live in `ultralytics/cfg/models/weed-yolov12/` (the baseline is `ultralytics/cfg/models/v12/yolov12.yaml`).
Put the scale letter right after `yolov12`, e.g. `weed-yolov12s.yaml` or `yolov12m-SSFF-P234.yaml`.

Params and GFLOPs below are for scale **n**, 640×640 input, and nc = 3, as printed by `python ablation.py --info --scale n`:

| Config | C1 P2/−P5 | C2 SSFF | C3 LDConv | Heads | Params (M) | GFLOPs |
| --- | :-: | :-: | :-: | --- | --: | --: |
| `yolov12.yaml` (baseline) | | | | P3–P5 | 2.520 | 6.0 |
| `yolov12-P2.yaml` | P2 only | | | P2–P5 | 2.594 | 9.9 |
| `yolov12-P234.yaml` | ✓ | | | P2–P4 | 1.866 | 9.3 |
| `yolov12-SSFF.yaml` | | ✓ | | P3–P5 | 2.724 | 6.6 |
| `yolov12-LD.yaml` | | | ✓ | P3–P5 | 2.133 | 5.5 |
| `yolov12-SSFF-P234.yaml` | ✓ | ✓ | | P2–P4 | 1.918 | 10.0 |
| `yolov12-LD-P234.yaml` | ✓ | | ✓ | P2–P4 | 1.566 | 8.8 |
| `yolov12-SSFF-LD.yaml` | | ✓ | ✓ | P3–P5 | 2.337 | 6.1 |
| **`weed-yolov12.yaml`** | ✓ | ✓ | ✓ | P2–P4 | **1.618** | 9.5 |
| `weed-yolov12-lite.yaml` (backbone P5 also removed, DEAL style) | ✓ | ✓ | ✓ | P2–P4 | 0.748 | 8.5 |

Component 4 (the box loss) is set with training arguments and works with any config:

| Argument | Default (= YOLOv12) | Weed-YOLOv12 | Meaning |
| --- | --- | --- | --- |
| `iou_type` | `CIoU` | `WIoU` | IoU term: `CIoU` or `WIoU` (Wise-IoU v3) |
| `nwd` | `False` | `True` | add the Normalized Wasserstein Distance term |
| `iou_ratio` | `0.5` | `0.5` (tune) | `box = iou_ratio * IoU term + (1 - iou_ratio) * NWD term` |
| `nwd_constant` | `12.8` | `12.8` | NWD constant C, in feature-grid units (see notes) |

## Installation

Same as YOLOv12. Use a separate environment: this package is also named `ultralytics` and would overwrite the
original one.

```bash
conda create -n weed-yolov12 python=3.11
conda activate weed-yolov12
pip install -r requirements.txt
pip install -e .
pip install flash-attn --no-build-isolation   # optional (Linux + CUDA GPU): faster area attention for training
```

Without flash-attn, YOLOv12 falls back to PyTorch attention, which is slower but numerically equivalent.
On a Jetson, `docker/Dockerfile-jetson-jetpack6` (or `-jetpack5`) builds a JetPack container with this repository
installed: `sudo docker build -f docker/Dockerfile-jetson-jetpack6 -t weed-yolov12 .`

Running the scripts from the repository root also works without `pip install -e .`, as in DEAL-YOLO.

## Repository Layout

```
Weed-YOLOv12/
├── train.py  val.py  detect.py  get_FPS.py  ablation.py   # DEAL-YOLO style entry scripts + ablation runner
├── docker/Dockerfile-jetson-jetpack{5,6}                  # Jetson (Orin Nano) containers
└── ultralytics/
    ├── cfg/models/weed-yolov12/   # Weed-YOLOv12 + ablation configs
    ├── cfg/models/v12/            # YOLOv12 baseline
    ├── cfg/datasets/              # USU-CornWeedDB.yaml template (+ coco8 / coco8-seg for quick sanity runs)
    ├── nn/extra_modules/block.py  # SSFF (ScalSeq, Add)
    ├── nn/modules/conv.py         # LDConv
    ├── utils/loss.py, metrics.py  # NWD + Wise-IoU box loss
    └── models/yolo/{detect,segment}
```

The folder is the YOLOv12 code base reduced to what Weed-YOLOv12 uses: the **detect** task, and the **segment** task
kept for the planned segmentation variant. Removed from the YOLOv12 copy: the Gradio demo (`app.py`), `examples/`,
`tests/`, the docs config, the non-Jetson Dockerfiles, the other model families (YOLOv3–v11, RT-DETR), the
classify/pose/OBB/World tasks, SAM/FastSAM/NAS, tracking, `solutions`, and dataset configs for other datasets. The
original `yolov12/` repository still has all of them.

## Dataset

The dataset config is `Data/data.yaml`, for [USU-Corn-WeedDB](https://zenodo.org/records/20044178): 3 weed species,
640×640 tiles, with `Data/images/{train,val,test}` and `Data/labels/{train,val,test}`. It has no `path` key, so the
dataset root is the `Data/` folder itself wherever the repository is cloned. `train.py`, `val.py`, `detect.py` and
`ablation.py` all use it by default. **Check that the class order matches the released labels.**
(`ultralytics/cfg/datasets/USU-CornWeedDB.yaml` is the original template and is not used by the scripts.)

## Training / Validation / Prediction

The scripts follow DEAL-YOLO's style. Edit the paths at the top of each file.

```bash
python train.py     # Weed-YOLOv12n with WIoU + NWD (iou_ratio=0.5)
python val.py       # evaluate best.pt on the test split
python detect.py    # predict on a folder of images
python get_FPS.py --weights runs/train/weed-yolov12n/weights/best.pt --device 0 --half   # latency / FPS
```

Python API / CLI:

```python
from ultralytics import YOLO

model = YOLO('weed-yolov12n.yaml')
model.train(data='Data/data.yaml', epochs=300, imgsz=640, iou_type='WIoU', nwd=True, iou_ratio=0.5)
model.export(format='onnx')  # or format='engine', half=True for TensorRT (Jetson)
```

```bash
yolo train model=weed-yolov12n.yaml data=Data/data.yaml iou_type=WIoU nwd=True iou_ratio=0.5
```

## Ablation Study

`ablation.py` keeps a list of named experiments. They all use identical training settings (SGD, seed 0, same epochs,
batch size and image size) and differ only in the model yaml and the box loss. Each finished run trains, re-evaluates
`best.pt` on `--split`, and appends a row to `runs/ablation/summary.csv` (params, GFLOPs, P, R, mAP50, mAP50-95).

```bash
python ablation.py --list                       # experiments and groups
python ablation.py --info --scale n             # params / GFLOPs table, no training
python ablation.py --group components --device 0 --split test   # --data defaults to Data/data.yaml
python ablation.py --exp baseline weed --data Data/data.yaml --epochs 300 --batch 16
```

| Group | Experiments | Question |
| --- | --- | --- |
| `components` | baseline, p234, ssff, ld, loss, weed | effect of each component alone, and all together |
| `cumulative` | baseline → p234 → p234_ssff → p234_ssff_ld → weed | adding components one by one |
| `factorial` | all 2³ combinations of C1–C3 (CIoU loss) | interactions between the architecture changes |
| `head` | baseline, p2, p234 | adding P2 vs. adding P2 and removing P5 |
| `loss` | CIoU, WIoU, NWD, CIoU+NWD, WIoU+NWD with iou_ratio ∈ {0.25, 0.5, 0.75} | choosing the loss and its balance |

## Implementation Notes

- **LDConv arguments.** The YAML arguments are `[c2, num_param, stride]`. The 3rd value is the number of sampling
  points, not a kernel size. DEAL-YOLO keeps `3` (it swapped `Conv` → `LDConv` without changing the arguments), and so
  does Weed-YOLOv12. Larger values such as 5 or 9 can be tried in the ablation.
- **LDConv port.** The port is behaviour-preserving. DEAL-YOLO's `p_conv` backward hook never changed the gradients,
  so it is dropped. The sampling gather uses one shared index instead of one copy per channel, which saves a lot of
  memory on the P1/P2 layers, and its forward pass and gradients are identical to the original. `einops` is no longer
  needed.
- **GFLOPs for LDConv.** `thop` counts LDConv's convolutions but not its bilinear sampling, as in DEAL-YOLO. Report
  measured latency (`get_FPS.py`) next to GFLOPs.
- **SSFF anchoring.** `ScalSeq` produces the size and channels of its first input. Inputs that are coarser than it
  are upsampled exactly as in DEAL-YOLO. Inputs that are finer (needed for the P3/P4 heads) are reduced with adaptive
  max + avg pooling, as in ASF-YOLO's `Zoom_cat`.
- **Backbone P5.** `weed-yolov12.yaml` removes the P5 head and its neck layers but keeps the backbone P5 stage, which
  still feeds the top-down path. `weed-yolov12-lite.yaml` also removes the backbone P5 stage, like DEAL-YOLO's
  P2P2 model.
- **P2 block.** In YOLOv12-turbo the neck `A2C2f` layers are already built with `a2=False` (C3k blocks, no
  attention); area attention is only used in backbone layers 6 and 8. `A2C2f` also requires its hidden width to be a
  multiple of 32, which a 128-channel P2 block does not meet at scale n. The P2 block therefore uses `C3k2`.
- **NWD units.** The YOLO box loss receives boxes in feature-grid units (pixels / stride). With DEAL-YOLO's
  C = 12.8, the effective constant is 12.8 × stride pixels (51 px at P2, 102 px at P3, 205 px at P4).
- **Default loss.** `default.yaml` keeps YOLOv12's CIoU, so baseline runs in this repository are unchanged. The
  Weed-YOLOv12 loss is switched on explicitly (`train.py`, `ablation.py`, or the CLI), and every run's `args.yaml`
  records which loss was used.

## Citation

Weed-YOLOv12 builds on:

```bibtex
@article{tian2026yolov12,
  title={Yolov12: Attention-centric real-time object detectors},
  author={Tian, Yunjie and Ye, Qixiang and Doermann, David},
  journal={Advances in neural information processing systems},
  volume={38},
  pages={78433--78457},
  year={2026}
}

@misc{naidu2025dealyolodronebasedefficientanimal,
  title={DEAL-YOLO: Drone-based Efficient Animal Localization using YOLO},
  author={Aditya Prashant Naidu and Hem Gosalia and Ishaan Gakhar and Shaurya Singh Rathore and Krish Didwania and Ujjwal Verma},
  year={2025},
  eprint={2503.04698},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2503.04698},
}
```

Component papers: SSFF / ASF-YOLO ([arXiv:2312.06458](https://arxiv.org/abs/2312.06458)), LDConv
([arXiv:2311.11587](https://arxiv.org/abs/2311.11587)), NWD ([arXiv:2110.13389](https://arxiv.org/abs/2110.13389)),
Wise-IoU ([arXiv:2301.10051](https://arxiv.org/abs/2301.10051)).

## Acknowledgement

The code is based on [YOLOv12](https://github.com/sunsmarterjie/yolov12), [DEAL-YOLO](https://github.com/IshaanGakhar/DEAL-YOLO)
and [ultralytics](https://github.com/ultralytics/ultralytics). AGPL-3.0 license.
