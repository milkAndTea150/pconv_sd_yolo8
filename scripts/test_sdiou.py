#!/usr/bin/env python3
"""Smoke-test CIoU and SDIoU bbox_iou paths with dummy boxes."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from ultralytics.utils.metrics import bbox_iou


def main() -> None:
    box1 = torch.tensor([[0.0, 0.0, 4.0, 4.0], [0.0, 0.0, 20.0, 20.0]], requires_grad=True)
    box2 = torch.tensor([[0.2, 0.1, 4.1, 4.2], [0.2, 0.1, 20.1, 20.2]])

    ciou = bbox_iou(box1, box2, xywh=False, CIoU=True)
    sdiou = bbox_iou(box1, box2, xywh=False, SDIoU=True, delta=0.5)

    assert ciou.shape == sdiou.shape == (2, 1)
    assert torch.isfinite(ciou).all()
    assert torch.isfinite(sdiou).all()
    assert not torch.isclose(ciou[0], sdiou[0]), "small-object SDIoU should differ from CIoU"
    assert torch.isclose(ciou[1], sdiou[1]), "large-object SDIoU should reach its CIoU limit"
    (1.0 - sdiou).sum().backward()
    assert box1.grad is not None and torch.isfinite(box1.grad).all()

    print(f"CIoU: {ciou.flatten().tolist()}")
    print(f"SDIoU: {sdiou.flatten().tolist()}")


if __name__ == "__main__":
    main()
