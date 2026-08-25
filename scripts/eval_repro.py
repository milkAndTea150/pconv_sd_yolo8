#!/usr/bin/env python3
"""Evaluate a trained reproduction checkpoint on val or test and emit JSON metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.data.utils import IMG_FORMATS, check_det_dataset
from ultralytics.nn.modules.APConv import PConv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "datasets/my_ir.yaml")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/repro_compare/eval")
    parser.add_argument("--name", default=None)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_split_images(data: Path, split: str) -> int:
    """Count images from the same resolved split definition used by Ultralytics."""
    dataset = check_det_dataset(str(data), autodownload=False)
    sources = dataset[split]
    sources = sources if isinstance(sources, list) else [sources]
    count = 0
    for source in map(Path, sources):
        if source.is_dir():
            count += sum(path.suffix[1:].lower() in IMG_FORMATS for path in source.rglob("*.*"))
        elif source.is_file():
            count += sum(bool(line.strip()) for line in source.read_text(encoding="utf-8").splitlines())
        else:
            raise FileNotFoundError(f"Dataset split source not found: {source}")
    return count


def main() -> None:
    args = parse_args()
    weights = args.weights.expanduser().resolve()
    data = args.data.expanduser().resolve()
    if not weights.is_file():
        raise SystemExit(f"Weights file not found: {weights}")
    if not data.is_file():
        raise SystemExit(f"Dataset YAML not found: {data}")

    model = YOLO(str(weights))
    pconv_count = sum(isinstance(module, PConv) for module in model.model.modules())
    name = args.name or f"{weights.parents[1].name}_{args.split}"
    metrics = model.val(
        data=str(data),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(args.project.resolve()),
        name=name,
        exist_ok=True,
        conf=0.001,
        iou=0.7,
        plots=False,
    )
    output = {
        "weights": str(weights),
        "weights_sha256": sha256(weights),
        "data": str(data),
        "split": args.split,
        "images": count_split_images(data, args.split),
        "pconv_layers": pconv_count,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "speed_ms": {key: float(value) for key, value in metrics.speed.items()},
        "save_dir": str(metrics.save_dir),
    }
    output_path = Path(metrics.save_dir) / "metrics.json"
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"metrics_output: {output_path}")


if __name__ == "__main__":
    main()
