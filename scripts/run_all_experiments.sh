#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_all_experiments.sh — Launch all experiments for the bifurcation-aware
# power-flow solver.
#
# Usage:
#   bash scripts/run_all_experiments.sh --gpus 0,1,2,3,4,5,6,7   # 8-GPU parallel
#   bash scripts/run_all_experiments.sh --gpus 0,1                # 2-GPU parallel
#   bash scripts/run_all_experiments.sh                           # single GPU, sequential
#
# Prerequisites:
#   1. conda activate pfdelta
#   2. python data_generation.py                     # regenerate dataset with new fields
#   3. python data_generation.py --generate-n3       # generate N-3 test set
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DATA_DIR="data/processed/task4_solvability"
N3_DIR="data/processed/task4_solvability_n3"
CKPT_DIR="checkpoints"
LOG_DIR="logs"
WANDB_PROJECT="pfdelta-bifurcation"

# Common flags shared by all training runs
COMMON="--data-dir ${DATA_DIR} --checkpoint-dir ${CKPT_DIR} --log-dir ${LOG_DIR} \
--wandb-project ${WANDB_PROJECT} --hidden-dim 64 --num-mp-layers 4 --lr 1e-3 \
--batch-size 32 --epochs-stage1 200 --epochs-stage2 50 --seed 42 --amp"

# ─── Parse optional --gpus flag ──────────────────────────────────────────────
GPUS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus) GPUS="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

IFS=',' read -ra GPU_LIST <<< "${GPUS:-0}"
N_GPUS=${#GPU_LIST[@]}
echo "Using ${N_GPUS} GPU(s): ${GPU_LIST[*]}"

# ─── Helper: run a training job ──────────────────────────────────────────────
GPU_IDX=0
PIDS=()

run_train() {
  local name="$1"; shift
  local gpu="${GPU_LIST[$((GPU_IDX % N_GPUS))]}"
  GPU_IDX=$((GPU_IDX + 1))

  echo "[$name] Launching on GPU ${gpu} ..."
  CUDA_VISIBLE_DEVICES="$gpu" python models.py $COMMON "$@" \
    --wandb-run-name "$name" \
    > "${LOG_DIR}/${name}.log" 2>&1 &
  PIDS+=($!)
  echo "[$name] PID $! → log: ${LOG_DIR}/${name}.log"
}

run_eval() {
  local name="$1"; shift
  local ckpt="$1"; shift
  local gpu="${GPU_LIST[$((GPU_IDX % N_GPUS))]}"
  GPU_IDX=$((GPU_IDX + 1))

  echo "[$name] Eval on GPU ${gpu} ..."
  CUDA_VISIBLE_DEVICES="$gpu" python models.py $COMMON "$@" \
    --eval-only "$ckpt" --wandb-run-name "$name" \
    > "${LOG_DIR}/${name}.log" 2>&1 &
  PIDS+=($!)
  echo "[$name] PID $! → log: ${LOG_DIR}/${name}.log"
}

wait_all() {
  echo ""
  echo "Waiting for ${#PIDS[@]} job(s) ..."
  local failed=0
  for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
      echo "  ✗ PID $pid failed"
      failed=$((failed + 1))
    fi
  done
  PIDS=()
  GPU_IDX=0
  if [[ $failed -gt 0 ]]; then
    echo "WARNING: $failed job(s) failed. Check logs in ${LOG_DIR}/"
  else
    echo "All jobs in this batch succeeded."
  fi
  echo ""
}

mkdir -p "$LOG_DIR"

# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 0: Data preparation (run once, blocks until done)
# ═════════════════════════════════════════════════════════════════════════════
echo "============================================================"
echo " PHASE 0: Data Preparation"
echo "============================================================"

if [[ ! -f "${DATA_DIR}/train.pt" ]]; then
  echo "Generating base dataset ..."
  python data_generation.py
else
  echo "Base dataset exists, skipping."
fi

if [[ ! -f "${N3_DIR}/test.pt" ]]; then
  echo "Generating N-3 test set ..."
  python data_generation.py --generate-n3 --n3-output-dir "$N3_DIR"
else
  echo "N-3 test set exists, skipping."
fi

# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 1: Training runs (parallelised across GPUs)
# ═════════════════════════════════════════════════════════════════════════════
echo "============================================================"
echo " PHASE 1: Training (4 runs)"
echo "============================================================"

# E1 — Baseline, adaptive LM, full data, both stages
run_train "E1_baseline_adaptive_lm" \
  --adaptive-lm --T 5 --stage both

# E8B — Baseline, regularised solver, full data, both stages
run_train "E8B_baseline_regularised" \
  --T 5 --stage both

# E3 — N+N-1 only, adaptive LM (for transfer tests)
run_train "E3_n01_adaptive_lm" \
  --adaptive-lm --T 5 --stage both \
  --train-contingency-max 1

# E5B — N+N-1 only, adaptive LM, edge dropout 5%
run_train "E5B_n01_dropout005" \
  --adaptive-lm --T 5 --stage both \
  --train-contingency-max 1 --edge-dropout 0.05

wait_all

# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 2: Evaluation-only runs (use trained checkpoints)
# ═════════════════════════════════════════════════════════════════════════════
echo "============================================================"
echo " PHASE 2: Evaluation (7 runs)"
echo "============================================================"

# Find the final checkpoints produced by Phase 1.
# the trainer saves epoch_<N>_stage2_final_<sig>.pt
find_ckpt() {
  local pattern="$1"
  local ckpt
  ckpt=$(ls -t ${CKPT_DIR}/${pattern} 2>/dev/null | head -1)
  if [[ -z "$ckpt" ]]; then
    echo "ERROR: No checkpoint matching '${pattern}' in ${CKPT_DIR}/" >&2
    return 1
  fi
  echo "$ckpt"
}

CKPT_E1=$(find_ckpt "*stage2*final*.pt" | head -1)    # first match for E1
CKPT_E3=$(find_ckpt "*n01_adaptive*.pt" || true)
CKPT_E5B=$(find_ckpt "*dropout005*.pt" || true)
CKPT_E8B=$(find_ckpt "*regularised*.pt" || true)

# Fallback: if naming doesn't work, try the generic final checkpoint
if [[ -z "$CKPT_E1" ]]; then
  echo "WARNING: Could not auto-detect E1 checkpoint. Trying final_model_*.pt ..."
  CKPT_E1=$(find_ckpt "final_model_*.pt")
fi

echo "E1  checkpoint: $CKPT_E1"
echo "E3  checkpoint: ${CKPT_E3:-NOT FOUND}"
echo "E5B checkpoint: ${CKPT_E5B:-NOT FOUND}"
echo "E8B checkpoint: ${CKPT_E8B:-NOT FOUND}"

# --- E2: Per-κ-bin breakdown (uses E1 checkpoint, default test set) ---
run_eval "E2_per_bin_breakdown" "$CKPT_E1" \
  --adaptive-lm --T 5

# --- E3_eval: N-1→N-2 transfer (trained on n+n-1, test on n-2 only) ---
if [[ -n "$CKPT_E3" ]]; then
  run_eval "E3_eval_n2_transfer" "$CKPT_E3" \
    --adaptive-lm --T 5 \
    --train-contingency-max 1 --test-contingency-filter 2
fi

# --- E4: N-{1,2}→N-3 transfer (E1 checkpoint, N-3 test set) ---
run_eval "E4_n3_transfer" "$CKPT_E1" \
  --adaptive-lm --T 5 \
  --test-data-dir "$N3_DIR" --test-contingency-filter 3

# --- E5A_eval: No-dropout baseline transfer (E3 checkpoint on n-2) ---
if [[ -n "$CKPT_E3" ]]; then
  run_eval "E5A_eval_nodropout_n2" "$CKPT_E3" \
    --adaptive-lm --T 5 \
    --train-contingency-max 1 --test-contingency-filter 2
fi

# --- E5B_eval: Dropout transfer (E5B checkpoint on n-2) ---
if [[ -n "$CKPT_E5B" ]]; then
  run_eval "E5B_eval_dropout_n2" "$CKPT_E5B" \
    --adaptive-lm --T 5 \
    --train-contingency-max 1 --test-contingency-filter 2
fi

# --- E6: Calibration deep-dive (uses E1 checkpoint) ---
run_eval "E6_calibration" "$CKPT_E1" \
  --adaptive-lm --T 5

# --- E8B_eval: Regularised solver eval ---
if [[ -n "$CKPT_E8B" ]]; then
  run_eval "E8B_eval_regularised" "$CKPT_E8B" \
    --T 5
fi

wait_all

echo " All experiments complete!"
echo " Logs: ${LOG_DIR}/"
echo " Checkpoints: ${CKPT_DIR}/"
echo " W&B project: ${WANDB_PROJECT}"