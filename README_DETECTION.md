# SCFusion-Det: Object Detection Extension

这是基于SCFusion模型的目标检测扩展版本。保持原有的编码器、融合模块和解码器架构不变，增加了YOLOv8风格的检测头，用于通用目标检测任务（边界框预测）。

## 新增文件说明

### 模型文件
- **`models/detection_head.py`**: YOLOv8风格的检测头实现
  - `DetectionHead`: 多尺度检测头，输出边界框和类别预测
  - `DFL`: Distribution Focal Loss用于边界框回归
  - `SCFusionDetectionHead`: 适配器，连接SCFusion特征和检测头

- **`models/scfusion_det.py`**: SCFusion检测模型
  - `SCFusionDet`: 主检测模型，复用FREFormer编码器和CFGIM融合模块
  - `SCFusionDetWithBackbone`: 单模态检测版本（用于消融实验）

### 损失函数
- **`utils/detection_losses.py`**: 检测损失函数
  - `DetectionLoss`: YOLOv8风格的组合损失（分类 + 边界框 + DFL）
  - `TaskAlignedAssigner`: 任务对齐的标签分配策略
  - `bbox_iou`: IoU计算（支持GIoU、DIoU、CIoU）

### 数据加载
- **`data/detection_dataset.py`**: 目标检测数据集加载器
  - `RGBTDetectionDataset`: 支持COCO和YOLO格式标注
  - `DetectionTransforms`: 检测任务的数据增强
  - 自动检测VT5000风格的目录结构

### 训练和评估
- **`train_detection.py`**: 检测模型训练脚本
- **`scripts/inference_detection.py`**: 检测推理脚本（可视化结果）
- **`scripts/evaluate_detection.py`**: 检测评估脚本（计算mAP）
- **`utils/detection_metrics.py`**: 检测指标实现（mAP、NMS等）

### 配置文件
- **`configs/detection_config.py`**: 检测任务配置模板

## 快速开始

### 1. 数据准备

支持两种标注格式：

#### COCO格式
```
dataset/
  Train/
    RGB/
    T/
    annotations_train.json
  Test/
    RGB/
    T/
    annotations_test.json
```

annotations.json格式：
```json
{
  "images": [{"id": 1, "file_name": "img001.jpg", "width": 640, "height": 480}],
  "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [x, y, w, h]}],
  "categories": [{"id": 1, "name": "person"}]
}
```

#### YOLO格式
```
dataset/
  Train/
    RGB/
    T/
    labels/  # .txt文件，每行: class x_center y_center width height (归一化)
  Test/
    RGB/
    T/
    labels/
```

### 2. 配置修改

编辑 `configs/detection_config.py`:

```python
cfg = {
    'dataset_root': 'path/to/your/dataset',  # 修改为你的数据集路径
    'annotation_format': 'coco',  # 或 'yolo'
    'input_size': (640, 640),

    'model': {
        'num_classes': 80,  # 修改为你的类别数
        'base_ch': 32,
        'stage_channels': (32, 64, 128, 256),
        'share_encoder': True,
        'use_fpn': False,
    },

    'epochs': 300,
    'batch_size': 8,
    'lr': 1e-3,
    'device': 'cuda:0',
}
```

### 3. 训练

```bash
# 从头开始训练
python train_detection.py --config configs/detection_config.py --save_dir checkpoints_det

# 从检查点恢复训练
python train_detection.py --config configs/detection_config.py --resume checkpoints_det/ckpt_epoch_100.pth --save_dir checkpoints_det
```

### 4. 推理

```bash
# 单张图像对推理
python scripts/inference_detection.py \
    --config configs/detection_config.py \
    --ckpt checkpoints_det/ckpt_epoch_300.pth \
    --ir path/to/ir.png \
    --vis path/to/rgb.png \
    --out_dir results

# 批量推理（目录包含RGB/和T/子文件夹）
python scripts/inference_detection.py \
    --config configs/detection_config.py \
    --ckpt checkpoints_det/ckpt_epoch_300.pth \
    --input_dir dataset/Test \
    --out_dir results \
    --conf_threshold 0.25 \
    --nms_threshold 0.45

# 使用类别名称文件
python scripts/inference_detection.py \
    --config configs/detection_config.py \
    --ckpt checkpoints_det/ckpt_epoch_300.pth \
    --input_dir dataset/Test \
    --out_dir results \
    --class_names coco_classes.txt
```

### 5. 评估

```bash
python scripts/evaluate_detection.py \
    --config configs/detection_config.py \
    --ckpt checkpoints_det/ckpt_epoch_300.pth \
    --split test \
    --batch_size 4 \
    --num_workers 4
```

输出指标：
- **mAP@0.5**: IoU阈值0.5的平均精度
- **mAP@0.75**: IoU阈值0.75的平均精度
- **mAP@0.5:0.95**: COCO风格的平均精度（IoU从0.5到0.95，步长0.05）

## 模型架构

```
输入: IR图像 [B, 1, H, W] + VIS图像 [B, 3, H, W]
  ↓
[编码器] FREFormer (复用原模型)
  ├─ IR分支: Stem → 4个Stage → 多尺度特征 [f0, f1, f2, f3]
  └─ VIS分支: Stem → 4个Stage → 多尺度特征 [f0, f1, f2, f3]
  ↓
[融合模块] CFGIM (复用原模型)
  ├─ CFGIM0: 融合f0_ir和f0_vis → fused_f0
  ├─ CFGIM1: 融合f1_ir和f1_vis → fused_f1
  ├─ CFGIM2: 融合f2_ir和f2_vis → fused_f2
  └─ CFGIM3: 融合f3_ir和f3_vis → fused_f3
  ↓
[检测头] YOLOv8-style (新增)
  ├─ 边界框回归分支: Conv → Conv → DFL → 4个坐标
  └─ 分类分支: Conv → Conv → num_classes个类别概率
  ↓
输出: 边界框 [B, N, 4] + 类别概率 [B, N, num_classes]
```

## 损失函数

总损失 = λ_box × Box_Loss + λ_cls × Cls_Loss + λ_dfl × DFL_Loss

- **Box_Loss**: CIoU损失，衡量预测框和真实框的重叠度
- **Cls_Loss**: 二元交叉熵损失，用于多类别分类DFL_Loss**: Distribution Focal Loss，用于精确的边界框回归

默认权重: λ_box=7.5, λ_cls=0.5, λ_dfl=1.5

## 关键特性

1. **复用原模型组件**: 编码器（FREFormer）和融合模块（CFGIM）完全不变
2. **YOLOv8检测头**: 现代化的检测头设计，支持DFL边界框回归
3. **任务对齐分配**: 使用Task-Aligned Assigner进行标签分配
4. **多格式支持**: 同时支持COCO和YOLO标注格式
5. **完整pipeline**: 包含训练、推理、评估和可视化

## 与原模型的对比

| 特性 | 原SCFusion | SCFusion-Det |
|------|-----------|--------------|
| 任务 | 显著性目标检测 | 通用目标检测 |
| 输出 | 显著性图 (H×W) | 边界框 + 类别 |
| 标注 | 像素级mask | 边界框 (xyxy) |
| 损失 | BCE + Dice + FF | BCE + CIoU + DFL |
| 评估指标 | MAE, F-measure, S-measure | mAP@0.5, mAP@0.75, mAP@0.5:0.95 |
| 编码器 | ✓ 相同 | ✓ 相同 |
| 融合模块 | ✓ 相同 | ✓ 相同 |
| 解码器 | FRGM频域重建 | YOLOv8检测头 |

## 注意事项

1. **GPU内存**: 检测任务通常需要更大的输入尺寸（640×640），建议使用至少8GB显存的GPU
2. **批量大小**: 根据GPU显存调整batch_size，推荐从4或8开始
3. **学习率**: 默认1e-3适用于AdamW优化器，使用SGD时可能需要更大的学习率
4. **数据增强**: 检测任务对数据增强敏感，可以尝试调整hflip_prob和scale_range
5. **类别数**: 确保配置文件中的num_classes与数据集匹配

## 示例：COCO数据集

如果使用COCO数据集（80类）：

```python
# configs/detection_config.py
cfg = {
    'dataset_root': 'path/to/coco',
    'annotation_format': 'coco',
    'model': {
        'num_classes': 80,
        # ... 其他配置
    },
    'loss': {
        'num_classes': 80,
        # ... 其他配置
    },
}
```

创建类别名称文件 `coco_classes.txt`:
```
person
bicycle
car
...
```

## 故障排除

### 问题1: 训练损失不下降
- 检查学习率是否合适
- 确认数据标注格式正确
- 尝试降低batch_size
- 检查数据增强是否过强

### 问题2: mAP很低
- 确认IoU阈值设置合理
- 检查NMS阈值（推理时）
- 验证数据集标注质量
- 尝试训练更多epoch

### 问题3: 显存不足
- 降低batch_size
- 减小input_size（如512×512）
- 减小base_ch（如24或16）
- 使用梯度累积

## 引用

如果使用本代码，请引用原始SCFusion论文：

```bibtex
@article{scfusion,
  title={SCFusion: Spatial-Channel Cross-Frequency Guided Fusion Network for Infrared-Visible Image Object Detection},
  author={...},
  journal={...},
  year={2024}
}
```

## 许可证

本扩展代码遵循与原SCFusion项目相同的许可证。
