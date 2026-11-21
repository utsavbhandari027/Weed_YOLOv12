# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.models.yolo import detect, segment

from .model import YOLO

__all__ = "detect", "segment", "YOLO"
