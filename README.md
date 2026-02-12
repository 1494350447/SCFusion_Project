# SCFusion_Project

基于论文 **SCFusion: Spatial-Channel Cross-Frequency Guided Fusion Network for Infrared-Visible Image Object Detection** 的工程化实现与训练/测试脚本。

## 当前实现要点（已按论文结构对齐）

- 双模态并行编码器（支持共享主干），核心编码块为 `FREFormer`
- 融合模块 `CFGIM = HFRB + HSFB`
- 四阶段解码器，每阶段注入 `FRGM` 频域重建引导
- 联合损失：`Frequency Fidelity + Dice + BCE`

## 目录说明

- `models/`：模型定义（`freformer.py`、`fusion_modules.py`、`decoder_modules.py`、`scfusion_net.py`）
- `utils/losses.py`：论文对应联合损失实现
- `data/`：RGB-T 数据读取与配对增强（IR 单通道 + VIS 三通道）
- `train.py` / `test.py`：训练与推理评估入口

## 快速开始

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 准备数据（默认结构）

```text
data/
  ir/
  vis/
  mask/         # 可选
  splits/
    train.txt
    test.txt
```

3. 训练

```bash
python train.py --config configs/vt5000_config.py
```

4. 测试

```bash
python test.py --config configs/vt5000_config.py --ckpt checkpoints/ckpt_epoch_1.pth
```

## 配置

可在 `configs/base_config.py` 中修改：

- `model`: 输入通道、基础通道、是否共享编码器、是否深监督、每阶段块数
- `loss`: 联合损失系数与频段数

