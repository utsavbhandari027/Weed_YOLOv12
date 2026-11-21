import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent  # repo root, so paths work from any working directory
DATA = ROOT / 'Data' / 'data.yaml'
WEIGHTS = ROOT / 'runs' / 'train' / 'weed-yolov12n' / 'weights' / 'best.pt'

if __name__ == '__main__':
    model = YOLO(str(WEIGHTS))
    model.val(data=str(DATA),
              split='test',
              imgsz=640,
              batch=16,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              project=str(ROOT / 'runs' / 'val'),
              name='weed-yolov12n',
              )
