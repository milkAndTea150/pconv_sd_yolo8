#!/usr/bin/env python3
"""Train the YOLOv8n-P2 baseline or PConv+SDIoU variant with matched settings."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from ultralytics import YOLO, __version__ as ultralytics_version
from ultralytics.nn.modules.APConv import PConv


VARIANTS = {
    "baseline": {
        "model": ROOT / "models/yolov8n-p2-baseline.yaml",
        "bbox_loss": "ciou",
        "expected_pconv": 0,
    },
    "pconv_sdiou": {
        "model": ROOT / "models/yolov8n-p2p-pconv.yaml",
        "bbox_loss": "sdiou",
        "expected_pconv": 2,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--data", type=Path, default=ROOT / "datasets/my_ir.yaml")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", type=Path, default=ROOT / "runs/repro_compare")
    parser.add_argument("--name", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--exist-ok", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fraction", type=float, default=1.0)
    return parser.parse_args()


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() or "unknown"


def main() -> None:
    args = parse_args()
    data = args.data.expanduser().resolve()
    if not data.is_file():
        raise SystemExit(f"Dataset YAML not found: {data}")
    if args.epochs < 1 or args.batch < 1 or args.imgsz < 32:
        raise SystemExit("epochs and batch must be positive, and imgsz must be at least 32")
    if not 0.0 < args.fraction <= 1.0:
        raise SystemExit("fraction must be in (0, 1]")

    spec = VARIANTS[args.variant]
    model = YOLO(str(spec["model"]))
    pconv_count = sum(isinstance(module, PConv) for module in model.model.modules())
    if pconv_count != spec["expected_pconv"]:
        raise RuntimeError(
            f"{args.variant} resolved to {pconv_count} PConv layers; expected {spec['expected_pconv']}"
        )

    name = args.name or args.variant
    resolved = {
        "variant": args.variant,
        "model": str(spec["model"]),
        "bbox_loss": spec["bbox_loss"],
        "sdiou_delta": 0.5,
        "pconv_layers": pconv_count,
        "data": str(data),
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "workers": args.workers,
        "seed": args.seed,
        "amp": args.amp,
        "optimizer": "SGD",
        "lr0": 0.01,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "close_mosaic": 10,
        "patience": 0,
        "pretrained": False,
        "fraction": args.fraction,
        "git_commit": git_commit(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "ultralytics": ultralytics_version,
    }
    print(json.dumps(resolved, indent=2, ensure_ascii=False))

    result = model.train(
        model=str(spec["model"]),
        data=str(data),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=str(args.project.resolve()),
        name=name,
        exist_ok=args.exist_ok,
        amp=args.amp,
        optimizer="SGD",
        lr0=0.01,
        momentum=0.9,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        close_mosaic=10,
        patience=0,
        seed=args.seed,
        deterministic=True,
        pretrained=False,
        fraction=args.fraction,
        bbox_loss=spec["bbox_loss"],
        sdiou_delta=0.5,
    )
    save_dir = Path(result.save_dir)
    (save_dir / "repro_metadata.json").write_text(
        json.dumps(resolved, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"training_output: {save_dir}")


if __name__ == "__main__":
    main()
