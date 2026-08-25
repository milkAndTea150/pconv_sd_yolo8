#!/usr/bin/env python3
"""Two-stage PConv+SDIoU experiment on clean and occluded IR_Mix data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import numpy as np
import torch
import yaml
from ultralytics import YOLO
from ultralytics.nn.modules.APConv import PConv


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CLASS_NAMES = {
    0: "Cars",
    1: "People",
    2: "helicopter",
    3: "jeep",
    4: "truck",
    5: "speedboat",
    6: "fighter",
    7: "tank",
    8: "pickup",
    9: "freighter light",
}


@dataclass(frozen=True)
class DatasetSpec:
    yaml_path: Path
    root: Path
    names: dict[int, str]
    splits: dict[str, Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=("prepare", "smoke", "train-base", "eval-base", "finetune", "eval-final", "all"),
    )
    parser.add_argument(
        "--config",
        default="configs/occlusion_compare.yaml",
        help="Experiment configuration YAML.",
    )
    parser.add_argument("--models", nargs="+", default=None, help="Model keys from config.")
    parser.add_argument("--train-devices", nargs="+", type=int, default=None)
    parser.add_argument("--eval-device", type=int, default=None)
    parser.add_argument("--no-parallel", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required = {"project_root", "output_root", "datasets", "models", "train", "evaluation"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Config is missing keys: {sorted(missing)}")
    return config


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def normalize_names(raw: Any) -> dict[int, str]:
    if isinstance(raw, list):
        return {index: str(name) for index, name in enumerate(raw)}
    if isinstance(raw, dict):
        return {int(index): str(name) for index, name in raw.items()}
    raise ValueError("Dataset names must be a list or mapping")


def resolve_dataset(yaml_path: Path) -> DatasetSpec:
    with yaml_path.open("r", encoding="utf-8-sig") as handle:
        raw = yaml.safe_load(handle)
    root = Path(raw.get("path", yaml_path.parent))
    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()
    splits: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        value = raw.get(split)
        if value is None:
            continue
        candidate = Path(value)
        splits[split] = candidate if candidate.is_absolute() else root / candidate
    return DatasetSpec(yaml_path=yaml_path, root=root, names=normalize_names(raw["names"]), splits=splits)


def list_images(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(item.resolve() for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)
    if path.is_file() and path.suffix.lower() == ".txt":
        images = []
        for line in path.read_text(encoding="utf-8").splitlines():
            value = line.strip()
            if value:
                images.append(Path(value).resolve())
        return images
    raise FileNotFoundError(f"Image split does not exist or is unsupported: {path}")


def label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    try:
        reverse_index = parts[::-1].index("images")
    except ValueError as exc:
        raise ValueError(f"Image path has no 'images' directory: {image_path}") from exc
    index = len(parts) - reverse_index - 1
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def validate_labels(images: Iterable[Path], class_count: int) -> dict[str, int]:
    image_count = 0
    instance_count = 0
    empty_labels = 0
    for image_path in images:
        image_count += 1
        label_path = label_path_for_image(image_path)
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing label for {image_path}: {label_path}")
        lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            empty_labels += 1
        for line_number, line in enumerate(lines, start=1):
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"Invalid YOLO row {label_path}:{line_number}: {line}")
            class_id = int(float(fields[0]))
            coords = [float(value) for value in fields[1:]]
            if not 0 <= class_id < class_count:
                raise ValueError(f"Class id {class_id} out of range in {label_path}:{line_number}")
            if any(value < 0.0 or value > 1.0 for value in coords):
                raise ValueError(f"Non-normalized bbox in {label_path}:{line_number}: {line}")
            instance_count += 1
    return {"images": image_count, "instances": instance_count, "empty_labels": empty_labels}


def environment_record(config_path: Path) -> dict[str, Any]:
    import ultralytics

    gpu_records = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            gpu_records.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": properties.total_memory,
                }
            )
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "ultralytics": ultralytics.__version__,
        "config": str(config_path.resolve()),
        "config_sha256": sha256(config_path),
        "gpus": gpu_records,
    }


def prepare(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    dataset_cfg = config["datasets"]
    clean = resolve_dataset(Path(dataset_cfg["clean_yaml"]))
    occluded = resolve_dataset(Path(dataset_cfg["occluded_yaml"]))
    if clean.names != CLASS_NAMES or occluded.names != CLASS_NAMES:
        raise ValueError(f"Dataset class definitions do not match expected IR_Mix classes: {clean.names}, {occluded.names}")

    expected = dataset_cfg["expected_counts"]
    report: dict[str, Any] = {"environment": environment_record(config_path), "datasets": {}}
    for label, dataset in (("clean", clean), ("occluded", occluded)):
        split_report = {}
        for split in ("train", "val", "test"):
            images = list_images(dataset.splits[split])
            if len(images) != int(expected[split]):
                raise ValueError(f"{label}/{split}: expected {expected[split]} images, found {len(images)}")
            split_report[split] = validate_labels(images, len(dataset.names))
        report["datasets"][label] = {
            "yaml": str(dataset.yaml_path),
            "yaml_sha256": sha256(dataset.yaml_path),
            "root": str(dataset.root),
            "splits": split_report,
        }

    source_dir = Path(dataset_cfg["occluded_only_images"])
    selected_names = sorted(item.name for item in source_dir.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)
    if len(selected_names) != int(expected["occluded_only_test"]):
        raise ValueError(f"Expected {expected['occluded_only_test']} occluded-only images, found {len(selected_names)}")

    full_test_by_name = {path.name: path for path in list_images(occluded.splits["test"])}
    missing = [name for name in selected_names if name not in full_test_by_name]
    if missing:
        raise FileNotFoundError(f"Occluded-only filenames missing from full test split: {missing[:10]}")
    selected_images = [full_test_by_name[name] for name in selected_names]
    subset_stats = validate_labels(selected_images, len(occluded.names))

    generated_list = Path(dataset_cfg["generated_list"])
    atomic_write_text(generated_list, "".join(f"{path}\n" for path in selected_images))
    generated_yaml = Path(dataset_cfg["generated_yaml"])
    yaml_payload = {
        "path": str(occluded.root),
        "train": str(occluded.splits["train"]),
        "val": str(occluded.splits["val"]),
        "test": str(generated_list),
        "names": CLASS_NAMES,
    }
    atomic_write_text(generated_yaml, yaml.safe_dump(yaml_payload, allow_unicode=True, sort_keys=False))
    report["datasets"]["occluded_only_test"] = {
        "source_dir": str(source_dir),
        "generated_list": str(generated_list),
        "generated_list_sha256": sha256(generated_list),
        "generated_yaml": str(generated_yaml),
        **subset_stats,
    }

    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_root / "prepare_manifest.json", report)
    print(json.dumps(report["datasets"], ensure_ascii=False, indent=2))
    return report


def last_epoch(results_csv: Path) -> int:
    if not results_csv.is_file():
        return 0
    with results_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return 0
    return int(float(rows[-1]["epoch"]))


def train_run(
    config: dict[str, Any], model_key: str, device: int, stage: str, smoke: bool = False
) -> Path:
    train_cfg = config["train"]
    output_root = Path(config["output_root"])
    if smoke:
        run_name = f"_smoke/{model_key}"
        weights = Path(config["models"][model_key]["model"])
        data = Path(config["datasets"]["clean_yaml"])
        epochs = 1
        lr0 = float(train_cfg["base_lr0"])
    elif stage == "base":
        run_name = f"{model_key}_clean_base"
        weights = Path(config["models"][model_key]["model"])
        data = Path(config["datasets"]["clean_yaml"])
        epochs = int(train_cfg["base_epochs"])
        lr0 = float(train_cfg["base_lr0"])
    elif stage == "finetune":
        run_name = f"{model_key}_occluded_finetune"
        weights = output_root / f"{model_key}_clean_base" / "weights" / "best.pt"
        data = Path(config["datasets"]["occluded_yaml"])
        epochs = int(train_cfg["finetune_epochs"])
        lr0 = float(train_cfg["finetune_lr0"])
    else:
        raise ValueError(f"Unknown training stage: {stage}")

    if not weights.is_file():
        raise FileNotFoundError(f"Training weights do not exist: {weights}")
    run_dir = output_root / run_name
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    expected_epochs = epochs
    if best.is_file() and last_epoch(run_dir / "results.csv") >= expected_epochs:
        print(f"[skip] completed run: {run_dir}")
        return best

    common = dict(
        data=str(data),
        epochs=epochs,
        imgsz=int(train_cfg["imgsz"]),
        batch=8 if smoke else int(train_cfg["batch"]),
        device=str(device),
        workers=2 if smoke else int(train_cfg["workers"]),
        amp=bool(train_cfg["amp"]),
        cache=bool(train_cfg["cache"]),
        patience=int(train_cfg["patience"]),
        optimizer=str(train_cfg["optimizer"]),
        lr0=lr0,
        lrf=float(train_cfg["lrf"]),
        weight_decay=float(train_cfg["weight_decay"]),
        momentum=float(train_cfg["momentum"]),
        warmup_epochs=float(train_cfg["warmup_epochs"]),
        close_mosaic=int(train_cfg["close_mosaic"]),
        seed=int(train_cfg["seed"]),
        deterministic=bool(train_cfg["deterministic"]),
        plots=True,
        save=True,
        save_period=-1,
        verbose=True,
        project=str(output_root),
        name=run_name,
        exist_ok=True,
        fraction=0.01 if smoke else 1.0,
    )

    if last.is_file():
        print(f"[resume] {last}")
        model = YOLO(str(last))
        model.train(resume=True)
    else:
        print(f"[train] {model_key} stage={stage} device={device} weights={weights}")
        model = YOLO(str(weights))
        pconv_count = sum(isinstance(module, PConv) for module in model.model.modules())
        expected_pconv = int(config["models"][model_key]["expected_pconv"])
        if pconv_count != expected_pconv:
            raise RuntimeError(f"Expected {expected_pconv} PConv layers, found {pconv_count}: {weights}")
        model.train(
            **common,
            pretrained=False,
            bbox_loss="sdiou",
            sdiou_delta=float(train_cfg["sdiou_delta"]),
        )
    if not best.is_file():
        raise RuntimeError(f"Training did not produce best.pt: {run_dir}")
    return best


def box_iou(one: np.ndarray, many: np.ndarray) -> np.ndarray:
    if many.size == 0:
        return np.empty((0,), dtype=np.float32)
    intersection_x1 = np.maximum(one[0], many[:, 0])
    intersection_y1 = np.maximum(one[1], many[:, 1])
    intersection_x2 = np.minimum(one[2], many[:, 2])
    intersection_y2 = np.minimum(one[3], many[:, 3])
    intersection = np.maximum(0.0, intersection_x2 - intersection_x1) * np.maximum(0.0, intersection_y2 - intersection_y1)
    area_one = max(0.0, one[2] - one[0]) * max(0.0, one[3] - one[1])
    area_many = np.maximum(0.0, many[:, 2] - many[:, 0]) * np.maximum(0.0, many[:, 3] - many[:, 1])
    return intersection / np.maximum(area_one + area_many - intersection, 1e-9)


def read_ground_truth(image_path: Path, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    boxes = []
    classes = []
    for line in label_path_for_image(image_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        class_id, xc, yc, box_width, box_height = map(float, line.split())
        boxes.append(
            [
                (xc - box_width / 2) * width,
                (yc - box_height / 2) * height,
                (xc + box_width / 2) * width,
                (yc + box_height / 2) * height,
            ]
        )
        classes.append(int(class_id))
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4), np.asarray(classes, dtype=np.int64)


def false_alarms_per_image(
    model: YOLO,
    images: list[Path],
    evaluation: dict[str, Any],
    device: int,
) -> float:
    unmatched = 0
    count = 0
    predictions = model.predict(
        source=[str(path) for path in images],
        stream=True,
        imgsz=int(evaluation["imgsz"]),
        batch=int(evaluation["batch"]),
        device=str(device),
        half=bool(evaluation["half"]),
        augment=False,
        conf=float(evaluation["confidence"]),
        iou=float(evaluation["iou"]),
        max_det=int(evaluation["max_det"]),
        verbose=False,
    )
    for image_path, result in zip(images, predictions):
        count += 1
        height, width = result.orig_shape
        # Ultralytics may expose synthetic names such as ``image0.jpg`` when a
        # Python list is used as the source. Pair by stream order instead of
        # deriving labels from result.path.
        gt_boxes, gt_classes = read_ground_truth(image_path, height, width)
        used = np.zeros(len(gt_boxes), dtype=bool)
        if result.boxes is None or len(result.boxes) == 0:
            continue
        pred_boxes = result.boxes.xyxy.detach().cpu().numpy()
        pred_classes = result.boxes.cls.detach().cpu().numpy().astype(np.int64)
        confidences = result.boxes.conf.detach().cpu().numpy()
        for pred_index in np.argsort(-confidences):
            eligible = np.where((gt_classes == pred_classes[pred_index]) & (~used))[0]
            if eligible.size == 0:
                unmatched += 1
                continue
            ious = box_iou(pred_boxes[pred_index], gt_boxes[eligible])
            best_local = int(np.argmax(ious))
            if float(ious[best_local]) >= float(evaluation["iou"]):
                used[eligible[best_local]] = True
            else:
                unmatched += 1
    if count != len(images):
        raise RuntimeError(f"Prediction count mismatch: expected {len(images)}, got {count}")
    return unmatched / max(count, 1)


def model_flops(model: YOLO, imgsz: int) -> float | None:
    try:
        from ultralytics.utils.torch_utils import get_flops

        return float(get_flops(model.model, imgsz=imgsz))
    except Exception as exc:
        print(f"[warning] FLOPs unavailable: {exc}")
        return None


def evaluation_weight(config: dict[str, Any], model_key: str, stage: str) -> Path:
    suffix = "clean_base" if stage == "base" else "occluded_finetune"
    return Path(config["output_root"]) / f"{model_key}_{suffix}" / "weights" / "best.pt"


def evaluate_scope(
    config: dict[str, Any], model_key: str, stage: str, scope: str, device: int
) -> dict[str, Any]:
    evaluation = config["evaluation"]
    weights = evaluation_weight(config, model_key, stage)
    if not weights.is_file():
        raise FileNotFoundError(f"Evaluation weights do not exist: {weights}")
    scope_to_yaml = {
        "clean_test": "clean_yaml",
        "full_mixed_test": "occluded_yaml",
        "occluded_only_test": "generated_yaml",
    }
    data_yaml = Path(config["datasets"][scope_to_yaml[scope]])
    dataset = resolve_dataset(data_yaml)
    images = list_images(dataset.splits["test"])
    model = YOLO(str(weights))
    pconv_count = sum(isinstance(module, PConv) for module in model.model.modules())
    expected_pconv = int(config["models"][model_key]["expected_pconv"])
    if pconv_count != expected_pconv:
        raise RuntimeError(f"Evaluation checkpoint has {pconv_count} PConv layers, expected {expected_pconv}")

    blank = np.zeros((int(evaluation["imgsz"]), int(evaluation["imgsz"]), 3), dtype=np.uint8)
    for _ in range(int(evaluation["warmup_iterations"])):
        model.predict(
            source=blank,
            imgsz=int(evaluation["imgsz"]),
            batch=1,
            device=str(device),
            half=bool(evaluation["half"]),
            verbose=False,
        )

    repetitions = []
    for repeat in range(int(evaluation["repeats"])):
        metrics = model.val(
            data=str(data_yaml),
            split="test",
            imgsz=int(evaluation["imgsz"]),
            batch=int(evaluation["batch"]),
            device=str(device),
            workers=int(evaluation["workers"]),
            half=bool(evaluation["half"]),
            augment=False,
            max_det=int(evaluation["max_det"]),
            plots=False,
            save_json=False,
            verbose=False,
            project=str(Path(config["output_root"]) / "evaluations"),
            name=f"{model_key}_{stage}_{scope}_repeat{repeat + 1}",
            exist_ok=True,
        )
        speed = metrics.speed
        repetitions.append(
            {
                "precision": float(metrics.results_dict["metrics/precision(B)"]),
                "recall": float(metrics.results_dict["metrics/recall(B)"]),
                "map50": float(metrics.results_dict["metrics/mAP50(B)"]),
                "map50_95": float(metrics.results_dict["metrics/mAP50-95(B)"]),
                "preprocess_ms": float(speed.get("preprocess", 0.0)),
                "inference_ms": float(speed.get("inference", 0.0)),
                "postprocess_ms": float(speed.get("postprocess", 0.0)),
            }
        )
    aggregate = {key: median(record[key] for record in repetitions) for key in repetitions[0]}
    end_to_end_ms = aggregate["preprocess_ms"] + aggregate["inference_ms"] + aggregate["postprocess_ms"]
    aggregate.update(
        {
            "model": model_key,
            "stage": stage,
            "scope": scope,
            "weights": str(weights),
            "weights_sha256": sha256(weights),
            "data_yaml": str(data_yaml),
            "images": len(images),
            "fa_per_image": false_alarms_per_image(model, images, evaluation, device),
            "fps_inference": 1000.0 / aggregate["inference_ms"] if aggregate["inference_ms"] else None,
            "fps_end_to_end": 1000.0 / end_to_end_ms if end_to_end_ms else None,
            "parameters": sum(parameter.numel() for parameter in model.model.parameters()),
            "pconv_layers": pconv_count,
            "gflops": model_flops(model, int(evaluation["imgsz"])),
            "device": device,
            "repetitions": repetitions,
        }
    )
    return aggregate


def load_existing_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def save_results(config: dict[str, Any], new_rows: list[dict[str, Any]]) -> None:
    output_root = Path(config["output_root"])
    json_path = output_root / "summary.json"
    existing = load_existing_rows(json_path)
    merged = {(row["model"], row["stage"], row["scope"]): row for row in existing}
    for row in new_rows:
        merged[(row["model"], row["stage"], row["scope"])] = row
    rows = [merged[key] for key in sorted(merged)]
    atomic_write_json(json_path, rows)

    flat_fields = [
        "model", "stage", "scope", "images", "precision", "recall", "map50", "map50_95",
        "fa_per_image", "preprocess_ms", "inference_ms", "postprocess_ms", "fps_inference",
        "fps_end_to_end", "parameters", "pconv_layers", "gflops", "device", "weights", "weights_sha256", "data_yaml",
    ]
    csv_path = output_root / "summary.csv"
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=flat_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in flat_fields})
    temporary.replace(csv_path)
    write_markdown_summary(output_root / "summary.md", rows)


def write_markdown_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    lookup = {(row["model"], row["stage"], row["scope"]): row for row in rows}
    lines = [
        "# PConv + SDIoU 两阶段遮挡实验结果",
        "",
        "| 模型 | 阶段 | 测试范围 | 图像数 | mAP50 | FA/image | 推理 FPS | 端到端 FPS |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    model_keys = sorted({row["model"] for row in rows})
    scopes = ("clean_test", "full_mixed_test", "occluded_only_test")
    for model_key in model_keys:
        for stage in ("base", "finetune"):
            for scope in scopes:
                result = lookup.get((model_key, stage, scope))
                if not result:
                    continue
                lines.append(
                    f"| {model_key} | {stage} | {scope} | {result['images']} | "
                    f"{result['map50']:.4f} | {result['fa_per_image']:.4f} | "
                    f"{result['fps_inference']:.2f} | {result['fps_end_to_end']:.2f} |"
                )
    lines.extend(
        [
            "",
            "| 模型 | 测试范围 | 微调后 ΔmAP50 | 微调后 ΔFA/image |",
            "|---|---|---:|---:|",
        ]
    )
    for model_key in model_keys:
        for scope in scopes:
            base = lookup.get((model_key, "base", scope))
            final = lookup.get((model_key, "finetune", scope))
            if base and final:
                lines.append(
                    f"| {model_key} | {scope} | {final['map50'] - base['map50']:+.4f} | "
                    f"{final['fa_per_image'] - base['fa_per_image']:+.4f} |"
                )
    atomic_write_text(path, "\n".join(lines) + "\n")


def evaluate_stage(config: dict[str, Any], models: list[str], stage: str, device: int) -> None:
    rows = []
    for model_key in models:
        for scope in ("clean_test", "full_mixed_test", "occluded_only_test"):
            print(f"[evaluate] model={model_key} stage={stage} scope={scope} device={device}")
            row = evaluate_scope(config, model_key, stage, scope, device)
            rows.append(row)
            save_results(config, rows)


def run_parallel_training(
    args: argparse.Namespace,
    config: dict[str, Any],
    config_path: Path,
    phase: str,
    models: list[str],
    devices: list[int],
) -> None:
    if len(models) == 1 or args.no_parallel or args.child:
        stage = "base" if phase == "train-base" else "finetune"
        for model_key, device in zip(models, devices):
            train_run(config, model_key, device, stage=stage)
        return

    output_root = Path(config["output_root"])
    log_dir = output_root / "orchestrator_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    processes = []
    for model_key, device in zip(models, devices):
        log_path = log_dir / f"{phase}_{model_key}.log"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            phase,
            "--config",
            str(config_path),
            "--models",
            model_key,
            "--train-devices",
            str(device),
            "--child",
        ]
        handle = log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, cwd=config["project_root"])
        processes.append((model_key, process, handle, log_path))
        print(f"[spawn] {model_key} pid={process.pid} device={device} log={log_path}")
    failures = []
    for model_key, process, handle, log_path in processes:
        return_code = process.wait()
        handle.close()
        if return_code:
            failures.append(f"{model_key}: exit={return_code}, log={log_path}")
    if failures:
        raise RuntimeError("Parallel training failed: " + "; ".join(failures))


def smoke(config: dict[str, Any], models: list[str], devices: list[int], eval_device: int) -> None:
    for model_key, device in zip(models, devices):
        weights = train_run(config, model_key, device, stage="base", smoke=True)
        model = YOLO(str(weights))
        pconv_count = sum(isinstance(module, PConv) for module in model.model.modules())
        expected_pconv = int(config["models"][model_key]["expected_pconv"])
        if pconv_count != expected_pconv:
            raise RuntimeError(f"Smoke checkpoint has {pconv_count} PConv layers, expected {expected_pconv}")
        metrics = model.val(
            data=config["datasets"]["generated_yaml"],
            split="test",
            imgsz=int(config["evaluation"]["imgsz"]),
            batch=1,
            device=str(eval_device),
            workers=0,
            half=False,
            plots=False,
            verbose=False,
            project=str(Path(config["output_root"]) / "_smoke_eval"),
            name=model_key,
            exist_ok=True,
        )
        if len(model.names) != len(CLASS_NAMES):
            raise RuntimeError(f"{model_key} smoke model has {len(model.names)} classes, expected {len(CLASS_NAMES)}")
        smoke_images = list_images(resolve_dataset(Path(config["datasets"]["generated_yaml"])).splits["test"])[
            :16
        ]
        fa_per_image = false_alarms_per_image(model, smoke_images, config["evaluation"], eval_device)
        flops = model_flops(model, int(config["evaluation"]["imgsz"]))
        print(
            f"[smoke-ok] {model_key} classes={len(model.names)} "
            f"mAP50={metrics.results_dict['metrics/mAP50(B)']:.6f} "
            f"inference_ms={metrics.speed.get('inference', 0.0):.4f} "
            f"FA/image(16)={fa_per_image:.4f} GFLOPs={flops}"
        )


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    os.chdir(config["project_root"])
    models = args.models or list(config["models"])
    unknown = [model for model in models if model not in config["models"]]
    if unknown:
        raise ValueError(f"Unknown model keys: {unknown}")
    default_devices = [int(config["models"][model]["train_device"]) for model in models]
    devices = args.train_devices or default_devices
    if len(devices) == 1 and len(models) > 1:
        devices = devices * len(models)
    if len(devices) != len(models):
        raise ValueError("--train-devices must provide one device per model, or one shared device")
    eval_device = int(args.eval_device if args.eval_device is not None else config["evaluation"]["device"])

    if args.phase in {"prepare", "all"}:
        prepare(config, config_path)
    elif not Path(config["datasets"]["generated_yaml"]).is_file():
        raise FileNotFoundError("Prepared occluded-only YAML is missing; run the prepare phase first")

    if args.phase in {"smoke", "all"}:
        smoke(config, models, devices, eval_device)
    if args.phase in {"train-base", "all"}:
        run_parallel_training(args, config, config_path, "train-base", models, devices)
    if args.phase in {"eval-base", "all"}:
        evaluate_stage(config, models, "base", eval_device)
    if args.phase in {"finetune", "all"}:
        run_parallel_training(args, config, config_path, "finetune", models, devices)
    if args.phase in {"eval-final", "all"}:
        evaluate_stage(config, models, "finetune", eval_device)


if __name__ == "__main__":
    main()
