# SCFusion-Det: 基于FRGM解码器的目标检测扩展

这是基于SCFusion模型的目标检测扩展版本。**完整保留**原有的编码器、融合模块和FRGM解码器架构，在FRGM解码器后接YOLOv8风格的检测头，用于通用目标检测任务（边界框预测）。

## 架构设计

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
[FRGM解码器] (复用原模型) ← **保留频域重建优势**
  ├─ FRGM频域引导模块 (4个尺度)
  ├─ 多尺度上采样和特征融合
  └─ 输出中间特征: [d0, d1, d2, d3]
  ↓
[YOLOv8检测头] (新增)
  ├─ 边界框回归分支 (DFL)
  └─ 分类分支
  ↓
输出: 边界框 [B, N, 4] + 类别概率 [B, N, num_classes]
```

## 核心优势

### 1. **完整保留SCFusion优势**
- ✅ **FREFormer编码器**: 频域选择性滤波
- ✅ **CFGIM融合模块**: 跨频域引导交互
- ✅ **FRGM解码器**: 频域重建引导模块
- ✅ **频域处理**: 贯穿编码、融合、解码全流程

### 2. **FRGM解码器的价值**
- **特征增强**: 通过频域重建提供更丰富的特征表示
- **空间细节**: 多尺度上采样保留空间细节信息
- **频域引导**: 4个FRGM模块分别处理不同尺度
- **多尺度融合**: d0-d3提供了4个尺度的精炼特征

### 3. **科学性论证**
这种"FRGM解码器 + 检测头"的架构是科学的：
- FRGM解码器原本用于显著性检测，其中间特征包含了丰富的语义和空间信息
- 检测任务同样需要多尺度特征和空间细节
- FRGM的频域重建能力可以增强特征表示，有利于检测

## 新增文件说明

### 模型文件
- **`models/detection_head.py`**: YOLOv8风格的检测头实现
  - `DetectionHead`: 多尺度检测头
  - `DFL`: Distribution Focal Loss用于边界框回归
  - `SCFusionDetectionHead`: 适配器

- **`models/scfusion_det_with_frgm.py`**: 主检测模型
  - `SCFusionDetWithFRGM`: 完整模型（编码器+融合+FRGM+检测头）
  - `DecoderWithIntermediateFeatures`: 修改的解码器，返回中间特征

### 损失函数
- **`utils/detection_losses.py`**: 检测损失函数
  - `DetectionLoss`: YOLOv8风格组合损失（分类 + 边界框 + DFL）
  - `TaskAlignedAssigner`: 任务对齐的标签分配策略
  - `bbox_iou`: IoU计算（支持GIoU、DIoU、CIoU）

### 数据加载
- **`data/detection_dataset.py`**: 目标检测数据集加载器
  - `RGBTDetectionDataset`: 支持COCO和YOLO格式标注
  - `DetectionTransforms`: 检测任务的数据增强

### 训练和评估
- **`train_detection_frgm.py`**: 检测模型训练脚本
- **`scripts/inference_detection_frgm.py`**: 检测推理脚本
- **`scripts/evaluate_detection_frgm.py`**: 检测评估脚本
- **`utils/detection_metrics.py`**: 检测指标（mAP、NMS等）

### 配置文件
- **`configs/detection_frgm_config.py`**: 检测任务配置

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

编辑 `configs/detection_frgm_config.py`:

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
        'frgm_band_thresholds': None,  # FRGM频段阈值
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
python train_detection_frgm.py \
    --config configs/detection_frgm_config.py \
    --save_dir checkpoints_det

# 从检查点恢复训练
python train_detection_frgm.py \
    --config configs/detection_frgm_config.py \
    --resume checkpoints_det/ckpt_epoch_100.pth \
    --save_dir checkpoints_det
```

### 4. 推理

```bash
# 单张图像对推理
python scripts/inference_detection_frgm.py \
    --config configs/detection_frgm_config.py \
    --ckpt checkpoints_det/ckpt_epoch_300.pth \
    --ir path/to/ir.png \
    --vis path/to/rgb.png \
    --out_dir results

# 批量推理
python scripts/inference_detection_frgm.py \
    --config configs/detection_frgm_config.py \
    --ckpt checkpoints_det/ckpt_epoch_300.pth \
    --input_dir dataset/Test \
    --out_dir results    --conf_threshold 0.25 \
    --nms_threshold 0.45

# 使用类别名称文件
python scripts/inference_detection_frgm.py \
    --config configs/detection_frgm_config.py \
    --ckpt checkpoints_det/ckpt_epoch_300.pth \
    --input_dir dataset/Test \
    --out_dir results \
    --class_names coco_classes.txt
```

### 5. 评估

```bash
python scripts/evaluate_detection_frgm.py \
    --config configs/detection_frgm_config.py \
    --ckpt checkpoints_det/ckpt_epoch_300.pth \
    --split test \
    --batch_size 4 \
    --num_workers 4
```

输出指标：
- **mAP@0.5**: IoU阈值0.5的平均精度
- **mAP@0.75**: IoU阈值0.75的平均精度
- **mAP@0.5:0.95**: COCO风格的平均精度

## 损失函数

总损失 = λ_box × Box_Loss + λ_cls × Cls_Loss + λ_dfl × DFL_Loss

- **Box_Loss**: CIoU损失，衡量预测框和真实框的重叠度
- **Cls_Loss**: 二元交叉熵损失，用于多类别分类
- **DFL_Loss**: Distribution Focal Loss，用于精确的边界框回归

默认权重: λ_box=7.5, λ_cls=0.5, λ_dfl=1.5

## 与原模型的对比

| 特性 | 原SCFusion | SCFusion-Det (FRGM版) |
|------|-----------|---------------------|
| 任务 | 显著性目标检测 | 通用目标检测 |
| 输出 | 显著性图 (H×W) | 边界框 + 类别 |
| 标注 | 像素级mask | 边界框 (xyxy) |
| 损失 | BCE + Dice + FF | BCE + CIoU + DFL |
| 评估指标 | MAE, F-measure, S-measure | mAP@0.5, mAP@0.75, mAP@0.5:0.95 |
| 编码器 | ✓ FREFormer | ✓ FREFormer（相同）|
| 融合模块 | ✓ CFGIM | ✓ CFGIM（相同）|
| 解码器 | ✓ FRGM | ✓ FRGM（相同）|
| 检测头 | ✗ 无 | ✓ YOLOv8风格 |

## 注意事项

1. **GPU内存**: 检测任务通常需要更大的输入尺寸（640×640），建议使用至少8GB显存的GPU
2. **批量大小**: 根据GPU显存调整batch_size，推荐从4或8开始
3. **学习率**: 默认1e-3适用于AdamW优化器
4. **数据增强**: 检测任务对数据增强敏感，可调整hflip_prob和scale_range
5. **类别数**: 确保配置文件中的num_classes与数据集匹配
6. **FRGM优势**: 相比直接接检测头，FRGM版本保留了频域处理优势，可能获得更好的性能

## 示例：COCO数据集

```python
# configs/detectinfig.py
cfg = {
    'dataset_root': 'path/to/coco',
    'annotation_format': 'coco',
    'model': {
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

## 技术细节

### FRGM解码器的作用
1. **频域重建**: 4个FRGM模块分别处理不同尺度的特征
2. **多尺度融合**: 通过上采样和特征拼接融合多尺度信息
3. **空间细节**: 恢复编码器下采样丢失的空间细节
4. **特征精炼**: 输出的d0-d3特征比融合模块输出更丰富

### 为什么使用中间特征而非最终输出
- FRGM的最终输出是单通道显著性图，信息不够丰富
- 中间特征d0-d3保留了多通道特征表示
- 这些特征经过FRGM频域引导，质量更高
- 多尺度特征适合YOLOv8风格的检测头

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
