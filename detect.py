import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent  # repo root, so paths work from any working directory

if __name__ == '__main__':
    model = YOLO(str(ROOT / 'runs' / 'train' / 'weed-yolov12n' / 'weights' / 'best.pt')) # select your model.pt path
    model.predict(source=str(ROOT / 'Data' / 'images' / 'test'),
                  imgsz=640,
                  project=str(ROOT / 'runs' / 'detect'),
                  name='exp',
                  verbose=True,
                  save=True,
                  # conf=0.2,
                  # visualize=False # visualize model features maps
                  )
