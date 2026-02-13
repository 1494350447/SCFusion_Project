#!/usr/bin/env bash

# 运行完整训练，支持灵活覆盖 VT5000 或其他数据集的配置。

# 使用示例：

#  ./scripts/run_full_train.sh                       # 使用默认值运行（VT5000 配置）

#  ./scripts/run_full_train.sh --epochs 100 --batch_size 8 --gpus 0

#  ./scripts/run_full_train.sh --config configs/custom.py --lr 1e-4 --save_dir my_ckpts

#

# 注意：

# - 此脚本通过加载提供的配置生成一个临时配置文件，

#   并应用通过命令行选项传递的顶层 / 模型 / 损失函数覆盖参数。

# - 然后它调用 `python3 train.py --config <tmp_config> --save_dir <save_dir> [--resume <ckpt>]`。

# - 请确保您的 Python 环境已安装所需的包（参见 requirements.txt）。

# - 确认资源：训练可能非常消耗 GPU 资源；设置 `--gpus`（CUDA 设备索引）或

#   `--device cpu` 以强制使用 CPU。如果您在 GPU 上运行，请确保有足够的显存用于所选的批大小。

set -euo pipefail

### 默认值（如果需要其他默认值，请在此处更改） ###

DEFAULT_CONFIG="configs/vt5000_paper_recommended.py"

DEFAULT_SAVE_DIR="checkpoints/vt5000_run"

DEFAULT_BATCH_SIZE=8

DEFAULT_EPOCHS=300

DEFAULT_LR=3e-5

DEFAULT_NUM_WORKERS=4

DEFAULT_DEVICE="cuda"

DEFAULT_INPUT_SIZE="(384, 384)"

######################################

# 解析命令行参数

######################################

CONFIG="${DEFAULT_CONFIG}"

SAVE_DIR="${DEFAULT_SAVE_DIR}"

BATCH_SIZE=${DEFAULT_BATCH_SIZE}

EPOCHS=${DEFAULT_EPOCHS}

LR=${DEFAULT_LR}

NUM_WORKERS=${DEFAULT_NUM_WORKERS}

DEVICE="${DEFAULT_DEVICE}"

INPUT_SIZE="${DEFAULT_INPUT_SIZE}"

RESUME=""

GPUS=""  # 例如 "0" 或 "0,1"

print_usage() {

  echo "Usage: $0 [--config PATH] [--save_dir DIR] [--batch_size N] [--epochs N] [--lr LR] [--num_workers N] [--device cpu|cuda] [--gpus '0,1'] [--resume PATH]"

}

while [[ $# -gt 0 ]]; do

  case "$1" in

    --config) CONFIG="$2"; shift 2;;

    --save_dir) SAVE_DIR="$2"; shift 2;;

    --batch_size) BATCH_SIZE="$2"; shift 2;;

    --epochs) EPOCHS="$2"; shift 2;;

    --lr) LR="$2"; shift 2;;

    --num_workers) NUM_WORKERS="$2"; shift 2;;

    --device) DEVICE="$2"; shift 2;;

    --gpus) GPUS="$2"; shift 2;;

    --input_size) INPUT_SIZE="$2"; shift 2;;

    --resume) RESUME="$2"; shift 2;;

    -h|--help) print_usage; exit 0;;

    *) echo "Unknown arg: $1"; print_usage; exit 1;;

  esac

done

mkdir -p "${SAVE_DIR}"

if [[ -n "${GPUS}" ]]; then

  export CUDA_VISIBLE_DEVICES="${GPUS}"

  echo "Using CUDA_VISIBLE_DEVICES=${GPUS}"

fi

if [[ "${DEVICE}" == "cpu" ]]; then

  export CUDA_VISIBLE_DEVICES=""

  echo "Forcing CPU device (CUDA_VISIBLE_DEVICES cleared)"

fi

echo "Config: ${CONFIG}"

echo "Save dir: ${SAVE_DIR}"

echo "Batch size: ${BATCH_SIZE}, Epochs: ${EPOCHS}, LR: ${LR}, Num workers: ${NUM_WORKERS}"

echo "Device: ${DEVICE}"

######################################

# 创建包含覆盖参数的临时配置文件

######################################

TMP_CONFIG=$(mktemp /tmp/tmp_cfg_XXXX.py)

cleanup() { rm -f "${TMP_CONFIG}"; }

trap cleanup EXIT

# 将覆盖参数导出为环境变量，以便 Python 辅助脚本可以稳健地读取它们

export BATCH_SIZE="${BATCH_SIZE}"

export EPOCHS="${EPOCHS}"

export LR="${LR}"

export NUM_WORKERS="${NUM_WORKERS}"

export INPUT_SIZE="${INPUT_SIZE}"

# 使用一个小型的 Python 辅助脚本加载原始配置字典，应用环境变量中的覆盖参数，并写入新的配置文件。

python3 - <<PY

import os, sys, importlib.util, ast

orig_cfg_path = os.path.abspath(os.path.join(os.getcwd(), "${CONFIG}"))

out_path = os.path.abspath("${TMP_CONFIG}")

spec = importlib.util.spec_from_file_location('cfg_module', orig_cfg_path)

mod = importlib.util.module_from_spec(spec)

spec.loader.exec_module(mod)

cfg = getattr(mod, 'cfg')

# 从环境变量中读取覆盖参数

def getenv_int(name, default):

  v = os.environ.get(name)

  return int(v) if v is not None and v != '' else default

def getenv_float(name, default):

  v = os.environ.get(name)

  return float(v) if v is not None and v != '' else default

def getenv_str(name, default):

  v = os.environ.get(name)

  return v if v is not None and v != '' else default

cfg['batch_size'] = getenv_int('BATCH_SIZE', cfg.get('batch_size', 8))

cfg['epochs'] = getenv_int('EPOCHS', cfg.get('epochs', 300))

cfg['lr'] = getenv_float('LR', cfg.get('lr', 3e-5))

cfg['num_workers'] = getenv_int('NUM_WORKERS', cfg.get('num_workers', 4))

# 如果提供了输入尺寸，则进行解析（期望类似元组的字符串）

ins = getenv_str('INPUT_SIZE', None)

if ins is not None:

  try:

    cfg['input_size'] = ast.literal_eval(ins)

  except Exception:

    parts = [int(x) for x in ins.replace('(', '').replace(')', '').split(',') if x.strip()]

    cfg['input_size'] = tuple(parts)

# 安全地确保 'model' 条目存在，并根据需要设置通用的模型覆盖参数

cfg['model'] = dict(cfg.get('model', {}))

cfg['model'].setdefault('in_ch_ir', 1)

cfg['model'].setdefault('in_ch_vis', 3)

# 写入一个定义 cfg 的简单 Python 配置文件

with open(out_path, 'w') as f:

  f.write('cfg = ' + repr(cfg) + '\\n')

print('Wrote temporary config to', out_path)

PY

######################################

# 使用生成的配置运行 train.py

######################################

CMD=(python3 train.py --config "${TMP_CONFIG}" --save_dir "${SAVE_DIR}")

if [[ -n "${RESUME}" ]]; then

  CMD+=(--resume "${RESUME}")

fi

echo "Running training command: ${CMD[*]}"

"${CMD[@]}"

echo "Training finished. Models/checkpoints will be stored under ${SAVE_DIR}."
