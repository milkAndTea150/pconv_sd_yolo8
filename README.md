# YOLOv8n-P2 与 PConv + SDIoU 红外目标检测复现

本仓库提供两组训练配置，用于对比：

1. **Baseline**：YOLOv8n-P2 + CIoU；
2. **PConv + SDIoU**：在 YOLOv8n-P2 骨干网络前两层引入 PConv，并使用尺度动态 IoU 损失 SDIoU。

两组实验共用数据划分、训练入口和超参数，默认从头训练，不加载预训练权重。仓库不包含数据集、训练权重、缓存或运行日志。

## 1. 环境配置

已验证环境：

```text
Ubuntu 22.04.5 LTS
Python 3.10.20
PyTorch 2.5.1+cu121
torchvision 0.20.1+cu121
CUDA 12.1（PyTorch 构建版本）
NVIDIA A100 40GB
Ultralytics fork 8.1.0
```

建议使用 Conda 创建独立环境：

```bash
git clone git@github.com:milkAndTea150/pconv_sd_yolo8.git
cd pconv_sd_yolo8

conda create -n pconv_sd python=3.10 -y
conda activate pconv_sd
pip install -r requirements.txt
```

依赖版本集中在 `requirements.txt`。安装结束后可执行：

```bash
python -m pip check
python scripts/test_model_build.py
python scripts/test_sdiou.py
```

## 2. 数据集准备（训练前必须修改）

### 2.1 数据集不会随仓库上传

请自行准备 YOLO 检测格式数据。推荐结构如下：

```text
IR_Mix/
└── infrared_true/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/
        ├── val/
        └── test/
```

每张图像应有一个同名标签文件，例如：

```text
images/train/example_001.jpg
labels/train/example_001.txt
```

标签每一行采用标准 YOLO 格式：

```text
class_id center_x center_y width height
```

其中坐标和宽高均应归一化到 `[0, 1]`。没有目标的图像可以没有标签文件，或使用空标签文件。

### 2.2 创建本地数据配置

仓库只追踪可移植模板 `datasets/my_ir.example.yaml`。首次训练前执行：

```bash
cp datasets/my_ir.example.yaml datasets/my_ir.yaml
```

然后编辑 `datasets/my_ir.yaml`：

```yaml
path: /absolute/path/to/IR_Mix/infrared_true
train: images/train
val: images/val
test: images/test

names:
  0: Cars
  1: People
  2: helicopter
  3: jeep
  4: truck
  5: speedboat
  6: fighter
  7: tank
  8: pickup
  9: freighter light
```

需要特别注意：

- `path` 必须指向同时包含 `images/` 和 `labels/` 的数据根目录，而不是仓库目录。
- `train`、`val`、`test` 相对于 `path` 解析；也可以改为各自的绝对路径或图像清单文件。
- `names` 的编号、顺序和数量必须与标签中的 `class_id` 完全一致。
- 当前模型会根据数据配置重建 10 类检测头；不要直接使用类别定义不同的数据配置。
- 若训练遮挡数据集，只需把 `path` 改为遮挡数据根目录，但其目录结构、划分和类别映射必须保持一致。
- `datasets/my_ir.yaml` 含本机绝对路径，已被 `.gitignore` 排除，不应提交。
- 训练前应确认所有类别编号位于 `0` 到 `len(names)-1`，并检查同名图像和标签能正确匹配。

本次验证使用的 IR_Mix 划分为：

| 划分 | 图像数 |
|---|---:|
| train | 12,267 |
| val | 1,364 |
| test | 1,515 |

## 3. 训练流程

所有命令从仓库根目录启动。两个 Shell 脚本最终都调用统一入口 `scripts/train_repro.py`，用于固定公共参数并显式选择模型和损失函数。

### 3.1 Baseline：YOLOv8n-P2 + CIoU

```bash
bash scripts/train_baseline.sh \
  --data datasets/my_ir.yaml \
  --device 0
```

### 3.2 PConv + SDIoU

```bash
bash scripts/train_pconv_sd.sh \
  --data datasets/my_ir.yaml \
  --device 1
```

默认公共参数：

```text
imgsz=640
batch=16
epochs=300
optimizer=SGD
lr0=0.01
momentum=0.9
weight_decay=0.0005
warmup_epochs=3
close_mosaic=10
seed=0
deterministic=True
patience=0
amp=False
pretrained=False
```

可在命令后覆盖训练轮数、batch size、GPU 和输出名称，例如执行 1 epoch 链路检查：

```bash
bash scripts/train_baseline.sh \
  --data datasets/my_ir.yaml --epochs 1 --device 0 \
  --name baseline_1ep --exist-ok

bash scripts/train_pconv_sd.sh \
  --data datasets/my_ir.yaml --epochs 1 --device 1 \
  --name pconv_sdiou_1ep --exist-ok
```

训练结果保存在：

```text
runs/repro_compare/<run_name>/
```

其中包括 `args.yaml`、`best.pt`、`last.pt`、训练曲线和 `repro_metadata.json`。`runs/` 不由 Git 追踪。

## 4. 测试集评测

两种模型必须使用同一评测脚本和同一 test 划分：

```bash
python scripts/eval_repro.py \
  --weights runs/repro_compare/baseline/weights/best.pt \
  --data datasets/my_ir.yaml --split test --device 0 \
  --name baseline_test

python scripts/eval_repro.py \
  --weights runs/repro_compare/pconv_sdiou/weights/best.pt \
  --data datasets/my_ir.yaml --split test --device 0 \
  --name pconv_sdiou_test
```

评测结果写入 `runs/repro_compare/eval/<name>/metrics.json`，包括 Precision、Recall、mAP50、mAP50-95、PConv 层数、耗时和权重 SHA256。

## 5. 代码修改位置

| 功能 | 文件与位置 |
|---|---|
| PConv 实现 | `ultralytics/nn/modules/APConv.py::PConv` |
| PConv 导出 | `ultralytics/nn/modules/__init__.py` |
| PConv 模型解析 | `ultralytics/nn/tasks.py::parse_model` |
| PConv 网络结构 | `models/yolov8n-p2p-pconv.yaml` |
| Baseline 网络结构 | `models/yolov8n-p2-baseline.yaml` |
| SDIoU 数学实现 | `ultralytics/utils/metrics.py::bbox_iou` |
| CIoU/SDIoU 损失选择 | `ultralytics/utils/loss.py::BboxLoss` |
| 损失配置字段 | `ultralytics/cfg/default.yaml` |
| 统一训练入口 | `scripts/train_repro.py` |
| 统一评测入口 | `scripts/eval_repro.py` |

训练入口会检查模型结构：baseline 必须包含 0 个 PConv 层，PConv+SDIoU 模型必须包含 2 个 PConv 层。`bbox_loss` 由入口分别设为 `ciou` 和 `sdiou`，避免只修改模型名称但没有真正启用对应模块。

## 6. 一轮训练验证结果

以下结果用于验证数据、模型、损失、反向传播和测试链路能够完整运行，不代表正式收敛性能：

| 模型 | PConv 层数 | 损失 | Test mAP50 | Test mAP50-95 |
|---|---:|---|---:|---:|
| YOLOv8n-P2 baseline | 0 | CIoU | 0.0790 | 0.0233 |
| YOLOv8n-P2 + PConv | 2 | SDIoU | 0.0981 | 0.0300 |

正式对比应完成相同轮数训练，并使用各自在验证集上选出的 `best.pt` 在同一 test 集上评测。当前两组实验比较的是 PConv 与 SDIoU 的联合效果；若要区分各模块贡献，应补充 `baseline + SDIoU` 和 `PConv + CIoU` 两组消融实验。

## 7. 参考

PConv 与尺度动态损失参考：Yang et al., *Pinwheel-shaped Convolution and Scale-based Dynamic Loss for Infrared Small Target Detection*, AAAI 2025。

- Paper: <https://arxiv.org/abs/2412.16986>
- Reference implementation: <https://github.com/JN-Yang/PConv-SDloss-Data>
