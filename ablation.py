"""
Weed-YOLOv12 ablation runner. Every experiment uses the same training settings; only the model yaml and box loss change.

    python ablation.py --list                                         # show experiments and groups
    python ablation.py --info --scale n                               # params / GFLOPs of every experiment (no training)
    python ablation.py --group components                              # train + evaluate a group (Data/data.yaml)
    python ablation.py --exp baseline weed --data Data/data.yaml --split test

Results of each run are appended to <project>/summary.csv.
Components: C1 = P2 head / no P5 head, C2 = SSFF, C3 = LDConv, C4 = NWD + Wise-IoU box loss.
"""

import warnings
warnings.filterwarnings('ignore')
import argparse
import csv
from pathlib import Path

from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import get_flops, get_num_params

ROOT = Path(__file__).resolve().parent  # repo root
DATA = ROOT / 'Data' / 'data.yaml'  # USU-Corn-WeedDB dataset yaml

CIOU = dict(iou_type='CIoU', nwd=False)  # YOLOv12 default box loss
WEED = dict(iou_type='WIoU', nwd=True, iou_ratio=0.5)  # Weed-YOLOv12 box loss (C4)

# name: (model yaml, box-loss args, description)
EXPERIMENTS = {
    # one component at a time on top of YOLOv12
    'baseline':      ('yolov12.yaml', CIOU, 'YOLOv12'),
    'p2':            ('yolov12-P2.yaml', CIOU, '+ P2 head, P5 head kept (P2-P5)'),
    'p234':          ('yolov12-P234.yaml', CIOU, '+ P2 head, - P5 head          [C1]'),
    'ssff':          ('yolov12-SSFF.yaml', CIOU, '+ SSFF                        [C2]'),
    'ld':            ('yolov12-LD.yaml', CIOU, '+ LDConv                      [C3]'),
    'loss':          ('yolov12.yaml', WEED, '+ NWD + Wise-IoU              [C4]'),
    # combinations
    'p234_ssff':     ('yolov12-SSFF-P234.yaml', CIOU, 'C1 + C2'),
    'p234_ld':       ('yolov12-LD-P234.yaml', CIOU, 'C1 + C3'),
    'ssff_ld':       ('yolov12-SSFF-LD.yaml', CIOU, 'C2 + C3'),
    'p234_ssff_ld':  ('weed-yolov12.yaml', CIOU, 'C1 + C2 + C3'),
    'weed':          ('weed-yolov12.yaml', WEED, 'C1 + C2 + C3 + C4 = Weed-YOLOv12'),
    'weed_lite':     ('weed-yolov12-lite.yaml', WEED, 'Weed-YOLOv12, backbone P5 stage removed too'),
    # box-loss study on the full architecture (iou_ratio = weight of the IoU term, 1 - iou_ratio = weight of NWD)
    'weed_wiou':     ('weed-yolov12.yaml', dict(iou_type='WIoU', nwd=False), 'Wise-IoU only'),
    'weed_nwd':      ('weed-yolov12.yaml', dict(iou_type='WIoU', nwd=True, iou_ratio=0.0), 'NWD only'),
    'weed_ciou_nwd': ('weed-yolov12.yaml', dict(iou_type='CIoU', nwd=True, iou_ratio=0.5), 'CIoU + NWD (0.5)'),
    'weed_r025':     ('weed-yolov12.yaml', dict(iou_type='WIoU', nwd=True, iou_ratio=0.25), 'Wise-IoU + NWD (0.25)'),
    'weed_r075':     ('weed-yolov12.yaml', dict(iou_type='WIoU', nwd=True, iou_ratio=0.75), 'Wise-IoU + NWD (0.75)'),
}

GROUPS = {
    'components': ['baseline', 'p234', 'ssff', 'ld', 'loss', 'weed'],  # each component alone, then all together
    'cumulative': ['baseline', 'p234', 'p234_ssff', 'p234_ssff_ld', 'weed'],  # add components one by one
    'factorial':  ['baseline', 'p234', 'ssff', 'ld', 'p234_ssff', 'p234_ld', 'ssff_ld', 'p234_ssff_ld'],  # C1-C3, 2^3
    'head':       ['baseline', 'p2', 'p234'],  # add P2 vs. add P2 + remove P5
    'loss':       ['p234_ssff_ld', 'weed_wiou', 'weed_nwd', 'weed_ciou_nwd', 'weed_r025', 'weed', 'weed_r075'],
    'all':        list(EXPERIMENTS),
}


def scaled(cfg, scale):
    """'weed-yolov12.yaml' -> 'weed-yolov12n.yaml' (Ultralytics reads the scale from the letter after 'yolov12')."""
    return cfg.replace('yolov12', f'yolov12{scale}', 1)


def complexity(model, imgsz=640):
    """Params (M) and GFLOPs of a DetectionModel (thop counts LDConv's convolutions, not its bilinear sampling)."""
    return get_num_params(model) / 1e6, get_flops(model, imgsz)


def loss_str(loss):
    return loss['iou_type'] + (f" + NWD (iou_ratio={loss['iou_ratio']})" if loss.get('nwd') else '')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', nargs='+', default=[], help=f'experiments: {", ".join(EXPERIMENTS)}')
    parser.add_argument('--group', nargs='+', default=[], help=f'experiment groups: {", ".join(GROUPS)}')
    parser.add_argument('--list', action='store_true', help='list experiments and groups')
    parser.add_argument('--info', action='store_true', help='print params / GFLOPs only, no training')
    parser.add_argument('--data', type=str, default=str(DATA), help='dataset yaml')
    parser.add_argument('--nc', type=int, default=3, help='number of classes used by --info (USU-CornWeedDB: 3)')
    parser.add_argument('--scale', type=str, default='n', choices=list('nsmlx'), help='model scale')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--pretrained', type=str, default='', help='optional weights to load first, e.g. yolov12n.pt')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'], help='split for the final evaluation')
    parser.add_argument('--project', type=str, default='runs/ablation')
    opt = parser.parse_args()

    if opt.list:
        for name, (cfg, loss, desc) in EXPERIMENTS.items():
            print(f'{name:15s} {cfg:24s} {loss_str(loss):34s} {desc}')
        print()
        for name, exps in GROUPS.items():
            print(f'{name:12s} {" ".join(exps)}')
        raise SystemExit

    names = list(dict.fromkeys(opt.exp + [e for g in opt.group for e in GROUPS[g]]))  # keep order, drop duplicates
    if not names and not opt.info:
        parser.error('select experiments with --exp and/or --group')
    names = names or list(EXPERIMENTS)
    unknown = [n for n in names if n not in EXPERIMENTS]
    assert not unknown, f'unknown experiments {unknown}, choose from {list(EXPERIMENTS)}'

    if opt.info:
        print(f'\n{"experiment":15s} {"model":26s} {"params(M)":>10s} {"GFLOPs":>8s}  box loss')
        for name in names:
            cfg, loss, desc = EXPERIMENTS[name]
            n_p, flops = complexity(DetectionModel(scaled(cfg, opt.scale), nc=opt.nc, verbose=False), opt.imgsz)
            print(f'{name:15s} {scaled(cfg, opt.scale):26s} {n_p:10.3f} {flops:8.1f}  {loss_str(loss)}')
        raise SystemExit

    summary = Path(opt.project) / 'summary.csv'
    for name in names:
        cfg, loss, desc = EXPERIMENTS[name]
        model = YOLO(scaled(cfg, opt.scale))
        if opt.pretrained:
            model.load(opt.pretrained)  # transfers only layers with matching shapes
        model.train(data=opt.data,
                    cache=False,
                    imgsz=opt.imgsz,
                    epochs=opt.epochs,
                    batch=opt.batch,
                    close_mosaic=10,
                    workers=opt.workers,
                    device=opt.device,
                    optimizer='SGD',
                    seed=opt.seed,
                    deterministic=True,
                    project=opt.project,
                    name=f'{name}-{opt.scale}',
                    **loss,
                    )
        save_dir = model.trainer.save_dir
        best = YOLO(save_dir / 'weights' / 'best.pt')
        n_p, flops = complexity(best.model, opt.imgsz)
        m = best.val(data=opt.data, split=opt.split, imgsz=opt.imgsz, batch=opt.batch, device=opt.device,
                     project=opt.project, name=f'{name}-{opt.scale}-{opt.split}')

        row = {'experiment': name, 'description': desc, 'model': scaled(cfg, opt.scale), 'box_loss': loss_str(loss),
               'params(M)': f'{n_p:.3f}', 'GFLOPs': f'{flops:.1f}', 'split': opt.split,
               'P': f'{m.box.mp:.4f}', 'R': f'{m.box.mr:.4f}', 'mAP50': f'{m.box.map50:.4f}',
               'mAP50-95': f'{m.box.map:.4f}', 'save_dir': str(save_dir)}
        new = not summary.exists()
        summary.parent.mkdir(parents=True, exist_ok=True)
        with open(summary, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if new:
                writer.writeheader()
            writer.writerow(row)
        print(f'\n{name}: mAP50={row["mAP50"]} mAP50-95={row["mAP50-95"]} -> {summary}\n')
