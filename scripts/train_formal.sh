#!/usr/bin/env bash
# ============================================================
# 亚马逊棋 · 正式训练启动脚本（RTX Pro 6000 / Blackwell）
#
# 用法： bash scripts/train_formal.sh
#   - 后台启动训练，日志写到 logs/train_<时间戳>.log
#   - 实时查看： tail -f logs/train_<时间戳>.log
#   - 优雅停止： kill -TERM <PID>   （触发 SIGTERM 处理器，清理干净）
#
# 迁移到目标机器后，只需改下面「用户配置」段的 3 个变量。
# ============================================================

set -euo pipefail

# ---------- 用户配置（按目标机器修改） ----------
PYTHON_BIN="python3"              # 训练用 Python；若用 venv 见下方 VENV_PATH 激活块
CONFIG="config_train_13x256.yaml"  # 正式训练配置（13块/256，多进程 GPU self-play）
CUDA_VISIBLE="0"                  # 用哪块 GPU。**只用 GPU0**：GPU1 上有别人的任务
# 可选：若 Python 装在 conda 环境，取消下面一行并填环境名
# CONDA_ENV="amazon"
# 可选：若 Python 装在 venv，取消下面一行并填路径
# VENV_PATH="/home/yangch/work/yangg/.venv"
# ------------------------------------------------

# 自动定位项目根目录（脚本位于 <项目>/scripts/ 下）
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

# ---------- 环境激活（三选一，默认直接用系统 python3） ----------
if [[ -n "${CONDA_ENV:-}" ]]; then
    # shellcheck source=/dev/null
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
    PYTHON_BIN="python"
elif [[ -n "${VENV_PATH:-}" ]]; then
    # shellcheck source=/dev/null
    source "${VENV_PATH}/bin/activate"
    PYTHON_BIN="python"
fi
# --------------------------------------------------------------

# ---------- 运行前自检 ----------
"${PYTHON_BIN}" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA 不可用：请检查驱动 / PyTorch 是否按 CUDA 12.8 (cu128) 安装")
print(f"[check] GPU: {torch.cuda.get_device_name(0)}")
print(f"[check] torch: {torch.__version__}")
PY

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE}"

mkdir -p logs
LOG="logs/train_$(date +%Y%m%d_%H%M%S).log"

echo "[launch] 项目目录: ${PROJECT_DIR}"
echo "[launch] 配置文件 : ${CONFIG}"
echo "[launch] 日志文件 : ${LOG}"

nohup "${PYTHON_BIN}" main.py --mode train --config "${CONFIG}" > "${LOG}" 2>&1 &
PID=$!

echo "[launch] 训练进程 PID = ${PID}"
echo "[launch] 实时查看:   tail -f ${LOG}"
echo "[launch] 优雅停止:   kill -TERM ${PID}"
