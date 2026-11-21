#!/usr/bin/env bash
# Weed-YOLOv12 on a RunPod GPU pod. Run from the repository folder (e.g. /workspace/Weed-YOLOv12):
#   bash runpod.sh setup   # install train/val dependencies, check the GPU            (once per pod)
#   bash runpod.sh test    # 1-epoch smoke test into runs/smoke                        (~1-2 min)
#   bash runpod.sh train   # python train.py in the background, log in train.log
#   bash runpod.sh val     # python val.py on the test split, log in val.log
#   bash runpod.sh pack    # results.tar.gz with runs/ + logs, to download
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python3}

case "${1:-}" in
  setup)
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    "$PY" -c 'import sys; v = sys.version_info; print("python", sys.version.split()[0]); sys.exit(0 if (3, 8) <= v[:2] <= (3, 12) else "torch==2.2.2 needs Python 3.8-3.12: pick a RunPod template with py3.10/py3.11")'
    "$PY" -m pip install -q --upgrade pip
    # train/val only: leave out the ONNX export packages (onnx, onnxruntime, onnxruntime-gpu, onnxslim) and
    # pycocotools (only needed for save_json=True); they are not used by train.py / val.py.
    grep -v -i -E '^(onnx|onnxruntime|onnxruntime-gpu|onnxslim|pycocotools)==' requirements.txt > /tmp/req-train.txt
    "$PY" -m pip install -r /tmp/req-train.txt
    "$PY" -m pip install -e .  # this repo's modified ultralytics (+ matplotlib, pandas, seaborn, thop, ...)
    "$PY" -c 'import torch, ultralytics; print("torch", torch.__version__, "| CUDA", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0)); print("ultralytics from", ultralytics.__file__)'
    ;;
  test)  # separate project folder, so runs/train/weed-yolov12n stays free for the real run
    "$PY" -c "from train import DATA; from ultralytics import YOLO; YOLO('weed-yolov12n.yaml').train(data=str(DATA), epochs=1, batch=16, device='0', optimizer='SGD', iou_type='WIoU', nwd=True, iou_ratio=0.5, project='runs/smoke', name='t', exist_ok=True)"
    ;;
  train)  # nohup: keeps running after the browser tab / terminal is closed
    nohup "$PY" train.py > train.log 2>&1 &
    echo "training started (PID $!). Follow it with:  tail -f train.log   (Ctrl+C stops following, not training)"
    ;;
  val)
    "$PY" val.py 2>&1 | tee val.log
    ;;
  pack)
    tar -czf results.tar.gz runs $(ls train.log val.log 2>/dev/null)
    echo "created $(pwd)/results.tar.gz ($(du -h results.tar.gz | cut -f1)). Download it from the Jupyter file browser."
    ;;
  *)
    echo "usage: bash runpod.sh setup|test|train|val|pack"; exit 1
    ;;
esac
