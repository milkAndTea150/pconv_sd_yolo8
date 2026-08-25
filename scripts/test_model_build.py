#!/usr/bin/env python3
"""Build baseline and PConv model YAMLs to verify Ultralytics parser registration."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from ultralytics import YOLO


MODELS = (
    "models/yolov8n-p2-baseline.yaml",
    "models/yolov8n-p2p-pconv.yaml",
)


def main() -> None:
    for model_yaml in MODELS:
        model = YOLO(model_yaml)
        layers = len(model.model.model)
        params = sum(p.numel() for p in model.model.parameters())
        model.model.eval()
        with torch.inference_mode():
            _ = model.model(torch.zeros(1, 3, 640, 640))
        print(f"OK {model_yaml}: layers={layers}, params={params}, forward_ch=3")


if __name__ == "__main__":
    main()
