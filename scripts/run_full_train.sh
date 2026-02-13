#!/usr/bin/env bash
# Run full training with flexible overrides for VT5000 or other datasets.
# Usage examples:
#  ./scripts/run_full_train.sh                       # run with defaults (VT5000 config)
#  ./scripts/run_full_train.sh --epochs 100 --batch_size 8 --gpus 0
#  ./scripts/run_full_train.sh --config configs/custom.py --lr 1e-4 --save_dir my_ckpts
#
# Notes:
# - This script generates a temporary config file by loading the provided config
#   and applying top-level / model / loss overrides passed via CLI options.
# - It then calls `python3 train.py --config <tmp_config> --save_dir <save_dir> [--resume <ckpt>]`.
# - Make sure your Python environment has the required packages (see requirements.txt).
# - Confirm resources: training can be GPU-intensive; set `--gpus` (CUDA device indices) or
#   `--device cpu` to force CPU. If you run on GPU, ensure enough VRAM for chosen batch size.

set -euo pipefail

### Defaults (change here if you want other defaults) ###
DEFAULT_CONFIG="configs/vt5000_paper_recommended.py"
DEFAULT_SAVE_DIR="checkpoints/vt5000_run"
DEFAULT_BATCH_SIZE=8
DEFAULT_EPOCHS=300
DEFAULT_LR=3e-5
DEFAULT_NUM_WORKERS=4
DEFAULT_DEVICE="cuda"
DEFAULT_INPUT_SIZE="(384, 384)"

######################################
# Parse CLI args
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
GPUS=""  # e.g. "0" or "0,1"

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
# Create a temporary config file with overrides
######################################
TMP_CONFIG=$(mktemp /tmp/tmp_cfg_XXXX.py)
cleanup() { rm -f "${TMP_CONFIG}"; }
trap cleanup EXIT

# Export overrides as environment variables so the python helper can read them robustly
export BATCH_SIZE="${BATCH_SIZE}"
export EPOCHS="${EPOCHS}"
export LR="${LR}"
export NUM_WORKERS="${NUM_WORKERS}"
export INPUT_SIZE="${INPUT_SIZE}"

# Use a small Python helper to load the original config dict, apply overrides from env, and write a new config file.
python3 - <<PY
import os, sys, importlib.util, ast
orig_cfg_path = os.path.abspath(os.path.join(os.getcwd(), "${CONFIG}"))
out_path = os.path.abspath("${TMP_CONFIG}")

spec = importlib.util.spec_from_file_location('cfg_module', orig_cfg_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
cfg = getattr(mod, 'cfg')

# Read overrides from environment
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

# parse input size if provided (expecting a tuple-like string)
ins = getenv_str('INPUT_SIZE', None)
if ins is not None:
  try:
    cfg['input_size'] = ast.literal_eval(ins)
  except Exception:
    parts = [int(x) for x in ins.replace('(', '').replace(')', '').split(',') if x.strip()]
    cfg['input_size'] = tuple(parts)

# Safely ensure 'model' entry exists and set common model overrides if desired
cfg['model'] = dict(cfg.get('model', {}))
cfg['model'].setdefault('in_ch_ir', 1)
cfg['model'].setdefault('in_ch_vis', 3)

# write a simple python config file that defines cfg
with open(out_path, 'w') as f:
  f.write('cfg = ' + repr(cfg) + '\n')
print('Wrote temporary config to', out_path)
PY

######################################
# Run train.py with the generated config
######################################
CMD=(python3 train.py --config "${TMP_CONFIG}" --save_dir "${SAVE_DIR}")
if [[ -n "${RESUME}" ]]; then
  CMD+=(--resume "${RESUME}")
fi

echo "Running training command: ${CMD[*]}"
"${CMD[@]}"

echo "Training finished. Models/checkpoints will be stored under ${SAVE_DIR}."
