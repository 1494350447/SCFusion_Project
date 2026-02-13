# SCFusion_Project


本仓库包含：模型实现、数据加载、训练/评估/推理脚本与常用配置（已支持 VT5000 数据集结构）。

主要功能亮点
- `models/`: `FREFormer` 编码器、`CFGIM` 融合模块、`FRGM` 频域重建等模块实现。
- 支持 VT5000 数据集布局（`Train/` 与 `Test/` 下的 `RGB/`, `T/`, `GT/` 以及 `train.csv`/`test.csv`）。
- 训练脚本与启动器：`train.py`、`scripts/run_full_train.sh`（可覆盖超参）、`scripts/quick_train.py`（快速验证）。
- 评估与推理：`scripts/evaluate.py`（计算 MAE、E-measure、S-measure、Fβ、Weighted F-measure）、`scripts/inference.py`（单张/批量推理并保存预测图）。

依赖
- 请使用 Python 环境并安装依赖：

```bash
pip install -r requirements.txt
```

（`requirements.txt` 包含 `torch`, `torchvision`, `numpy`, `opencv-python`, `tqdm`, `matplotlib`）

数据准备（VT5000）
- 仓库内若存在 `VT5000/`，默认配置 `configs/vt5000_paper_recommended.py` 已指向该目录。
- VT5000 期望目录结构（仓库根或自定义 `dataset_root`）：

```
VT5000/
  Train/
    RGB/
    T/
    GT/
    train.csv
  Test/
    RGB/
    T/
    GT/
    test.csv
```

训练
- 快速验证（调试用，1 epoch / 小子集）：

```bash
python3 scripts/quick_train.py --num_samples 50 --epochs 1 --batch_size 4 --num_workers 0
```

- 完整训练（使用可配置的启动脚本 `run_full_train.sh`）：

```bash
chmod +x scripts/run_full_train.sh
./scripts/run_full_train.sh --gpus 0 --epochs 300 --batch_size 8 --save_dir checkpoints/vt5000_run
```

- 默认情况下脚本会生成临时配置并调用 `train.py`，检查点默认保存在 `checkpoints/vt5000_run`。

评估（测试集）
- 使用训练好的 `.pth` 运行评估并计算指标：

```bash
python3 scripts/evaluate.py --ckpt checkpoints/vt5000_run/ckpt_epoch_1.pth --batch_size 4 --num_workers 4 --save_preds preds_eval
```

- 输出指标：
  - MAE (Mean Absolute Error)
  - Fβ (最大 F-measure，paper 中通常使用 β=0.3 权重)
  - E-measure (Enhanced-measure)
  - S-measure (Structure-measure)
  - Weighted F-measure (ωFβ)

推理（保存预测图）
- 单张推理：

```bash
python3 scripts/inference.py --ckpt checkpoints/vt5000_run/ckpt_epoch_1.pth --ir path/to/T.png --vis path/to/RGB.png --out_dir preds_single
```

- 批量推理（输入目录包含 `RGB/` 和 `T/` 子文件夹）：

```bash
python3 scripts/inference.py --ckpt checkpoints/vt5000_run/ckpt_epoch_1.pth --input_dir VT5000/Test --out_dir preds_test
```

备注与注意事项
- `utils/metrics.py` 中实现了评估指标的实用（近似）实现：S-measure 与 ωFβ 为常用近似实现，若需严格与某篇论文完全一致的度量，请告知我以接入权威实现。
- 评估/推理脚本默认会把预测图以 8 位灰度 PNG 保存到 `--save_preds`/`--out_dir`。
- 在 GPU 上运行完整训练会消耗较多显存与时间，请根据可用资源修改 `--gpus`、`--batch_size` 和 `--num_workers` 参数。
