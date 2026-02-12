# SCFusion 论文实现对齐说明

本文档记录本仓库与论文结构的一一对应关系，便于后续复现与扩展。

## 架构映射

- `FREFormer Encoder` → `models/freformer.py`
  - `DownsamplingLayer`: `LN -> Conv(stride=2) -> LN`
  - `FREFormerBlock`: `X + LSF(LN(X))`，再 `+ ChannelMLP(LN(.))`
  - 支持共享主干（IR/VIS 双 stem + shared backbone）

- `CFGIM` → `models/fusion_modules.py`
  - `HFRB`: 全局通道权重 + 局部通道频率校准
  - `HSFB`: 空间频域幅相分解、通道频域校准、结构保持重组
  - `CFGIM`: `HFRB(IR/VIS)` 后进入 `HSFB` 融合

- `FRGM Decoder` → `models/decoder_modules.py`
  - 四阶段解码
  - 每阶段引入 `FRGM`（四频带掩码+可学习频段权重）

- `Joint Loss` → `utils/losses.py`
  - `L_total = λ1 * L_ff + λ2 * L_dice + λ3 * L_ce`
  - `L_ff` 基于边缘图 FFT 的多频段频域保真约束

## 训练与数据

- `train.py`
  - 使用 `SCFusionLoss`（联合损失）
  - 支持深监督输出与断点续训

- `data/rgbt_dataset.py`
  - IR 默认单通道加载（`L`）
  - VIS 为三通道（`RGB`）

- `data/transforms.py`
  - IR 单通道归一化，VIS 三通道归一化
  - 保证 mask 判空逻辑稳定

