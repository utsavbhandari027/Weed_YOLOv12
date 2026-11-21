import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent  # repo root, so paths work from any working directory
DATA = ROOT / 'Data' / 'data.yaml'

if __name__ == '__main__':
    model = YOLO('weed-yolov12n.yaml')  # Weed-YOLOv12: P2-P4 heads + SSFF + LDConv
    model.train(data=str(DATA),
                cache=False,
                imgsz=640,
                epochs=200,
                patience=30,
                batch=16,
                close_mosaic=10,
                workers=8,
                device='0',
                optimizer='SGD', # using SGD
                seed=0,
                # Weed-YOLOv12 box loss: iou_ratio * Wise-IoU(v3) + (1 - iou_ratio) * NWD
                iou_type='WIoU', # CIoU (YOLOv12 default) or WIoU (Wise-IoU v3)
                nwd=True,        # add Normalized Wasserstein Distance
                iou_ratio=0.5,   # balance between the IoU term and NWD (tune in the ablation study)
                # resume=str(ROOT / 'runs/train/weed-yolov12n/weights/last.pt'), # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                project=str(ROOT / 'runs' / 'train'),
                name='weed-yolov12n',
                )
