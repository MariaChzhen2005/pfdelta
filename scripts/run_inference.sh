#!/usr/bin/env bash
# ───────────────────
# run_inference.sh — Re-run all evaluation experiments with epsilon=1e-3.
#
# Usage:
#   bash scripts/run_inference.sh --gpus 0,1,2,3   # multi-GPU parallel
#   bash scripts/run_inference.sh                   # single GPU, sequential
# ───────────────────
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DATA_DIR="data/processed/task4_solvability"
N3_DIR="data/processed/task4_solvability_n3"
CKPT_DIR="checkpoints"
LOG_DIR="logs"
WANDB_PROJECT="pfdelta-bifurcation"

COMMON="--data-dir ${DATA_DIR} --checkpoint-dir ${CKPT_DIR} --log-dir ${LOG_DIR} \
--wandb-project ${WANDB_PROJECT} --hidden-dim 64 --num-mp-layers 4 --lr 1e-3 \
--batch-size 256 --epochs-stage1 25 --epochs-stage2 10 --early-stop-patience 8 \
--scheduler-patience 5 --seed 42 --amp --epsilon 1e-3"

# ─── Parse optional --gpus flag ─────────────────
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

# ─── Helper: compute run signature via Python ───
get_sig() {
  python models.py $COMMON "$@" --print-signature --cpu 2>/dev/null
}

# ─── Helper: run eval job ───────────────────────
GPU_IDX=0
PIDS=()

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

run_baseline() {
  local name="$1"; shift
  local gpu="${GPU_LIST[$((GPU_IDX % N_GPUS))]}"
  GPU_IDX=$((GPU_IDX + 1))

  echo "[$name] Baseline on GPU ${gpu} ..."
  CUDA_VISIBLE_DEVICES="$gpu" python models.py $COMMON "$@" \
    --wandb-run-name "$name" \
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
#  Locate checkpoints by signature
# ═════════════════════════════════════════════════════════════════════════════
echo "============================================================"
echo " Computing run signatures ..."
echo "============================================================"

SIG_E1=$(get_sig --adaptive-lm --T 3 --stage both)
SIG_E3=$(get_sig --adaptive-lm --T 3 --stage both --train-contingency-max 1)
SIG_E5B=$(get_sig --adaptive-lm --T 3 --stage both --train-contingency-max 1 --edge-dropout 0.05)
SIG_VANILLA=$(get_sig --adaptive-lm --T 3 --lambda-infeasibility 0)

echo "  E1:      ${SIG_E1}"
echo "  E3:      ${SIG_E3}"
echo "  E5B:     ${SIG_E5B}"
echo "  VANILLA: ${SIG_VANILLA}"

CKPT_E1="${CKPT_DIR}/final_model_${SIG_E1}.pt"
CKPT_E3="${CKPT_DIR}/final_model_${SIG_E3}.pt"
CKPT_E5B="${CKPT_DIR}/final_model_${SIG_E5B}.pt"
CKPT_VANILLA="${CKPT_DIR}/final_model_${SIG_VANILLA}.pt"
CKPT_E1_S1="${CKPT_DIR}/epoch_25_stage1_final_${SIG_E1}.pt"

for name_ckpt in "E1:${CKPT_E1}" "E1_S1:${CKPT_E1_S1}" "E3:${CKPT_E3}" "E5B:${CKPT_E5B}" "VANILLA:${CKPT_VANILLA}"; do
  name="${name_ckpt%%:*}"
  ckpt="${name_ckpt#*:}"
  if [[ -f "$ckpt" ]]; then
    echo "  ✓ ${name} checkpoint found: ${ckpt}"
  else
    echo "  ✗ WARNING: ${name} checkpoint NOT found: ${ckpt}"
  fi
done

# ═════════════════════════════════════════════════════════════════════════════
#  Flat-start baselines
# ═════════════════════════════════════════════════════════════════════════════
echo "============================================================"
echo " Flat-start baselines"
echo "============================================================"

run_baseline "B_FLAT_T5" \
  --adaptive-lm --T 3 --max-iter-inference 5 --baseline flat-start

run_baseline "B_FLAT_T20" \
  --adaptive-lm --T 3 --max-iter-inference 20 --baseline flat-start

run_baseline "B_FLAT_T50" \
  --adaptive-lm --T 3 --max-iter-inference 50 --baseline flat-start

wait_all

# ═════════════════════════════════════════════════════════════════════════════
#  GNN-only baselines (Concern 2)
# ═════════════════════════════════════════════════════════════════════════════
echo "============================================================"
echo " GNN-only baselines"
echo "============================================================"

if [[ -f "$CKPT_VANILLA" ]]; then
  run_eval "B_VANILLA_T0_baseline" "$CKPT_VANILLA" \
    --adaptive-lm --T 3 --max-iter-inference 0 --lambda-infeasibility 0
fi

if [[ -f "$CKPT_E1_S1" ]]; then
  run_eval "E_S1_T0_gnn_direct" "$CKPT_E1_S1" \
    --adaptive-lm --T 3 --max-iter-inference 0

  run_eval "E_S1_T5_plus_solver" "$CKPT_E1_S1" \
    --adaptive-lm --T 3 --max-iter-inference 5

  run_eval "E_S1_T10_plus_solver" "$CKPT_E1_S1" \
    --adaptive-lm --T 3 --max-iter-inference 10
fi

wait_all

# ═════════════════════════════════════════════════════════════════════════════
#  Goal 1: Iteration sweep (E1 model)
# ═════════════════════════════════════════════════════════════════════════════
echo "============================================================"
echo " Goal 1: Iteration sweep"
echo "============================================================"

run_eval "E2_per_bin_breakdown" "$CKPT_E1" \
  --adaptive-lm --T 3

run_eval "E7_T0_gnn_only" "$CKPT_E1" \
  --adaptive-lm --T 3 --max-iter-inference 0

run_eval "E7_T1_iters" "$CKPT_E1" \
  --adaptive-lm --T 3 --max-iter-inference 1

run_eval "E7_T5_iters" "$CKPT_E1" \
  --adaptive-lm --T 3 --max-iter-inference 5

run_eval "E7_T10_iters" "$CKPT_E1" \
  --adaptive-lm --T 3 --max-iter-inference 10

wait_all

# ═════════════════════════════════════════════════════════════════════════════
#  Concern 4: Per-contingency stratified eval
# ═════════════════════════════════════════════════════════════════════════════
echo "============================================================"
echo " Per-contingency stratified eval"
echo "============================================================"

run_eval "E1_N0_base" "$CKPT_E1" \
  --adaptive-lm --T 3 --test-contingency-filter 0

run_eval "E1_N1_contingency" "$CKPT_E1" \
  --adaptive-lm --T 3 --test-contingency-filter 1

run_eval "E1_N2_contingency" "$CKPT_E1" \
  --adaptive-lm --T 3 --test-contingency-filter 2

wait_all

# ═════════════════════════════════════════════════════════════════════════════
#  Goal 2: Infeasibility detection
# ═════════════════════════════════════════════════════════════════════════════
echo "============================================================"
echo " Infeasibility detection"
echo "============================================================"

run_eval "E6A_infeas_learned_only" "$CKPT_E1" \
  --adaptive-lm --T 3 --infeas-detect-mode learned_only

run_eval "E6B_infeas_heuristic_only" "$CKPT_E1" \
  --adaptive-lm --T 3 --infeas-detect-mode heuristic_only

wait_all

# ─── Per-contingency infeasibility calibration ──
echo "--- Learned head per contingency ---"

run_eval "E6A_learned_N0" "$CKPT_E1" \
  --adaptive-lm --T 3 --infeas-detect-mode learned_only \
  --test-contingency-filter 0

run_eval "E6A_learned_N1" "$CKPT_E1" \
  --adaptive-lm --T 3 --infeas-detect-mode learned_only \
  --test-contingency-filter 1

run_eval "E6A_learned_N2" "$CKPT_E1" \
  --adaptive-lm --T 3 --infeas-detect-mode learned_only \
  --test-contingency-filter 2

run_eval "E6A_learned_N3" "$CKPT_E1" \
  --adaptive-lm --T 3 --infeas-detect-mode learned_only \
  --test-data-dir "$N3_DIR" --test-contingency-filter 3

wait_all

echo "--- Heuristic per contingency ---"

run_eval "E6B_heuristic_N0" "$CKPT_E1" \
  --adaptive-lm --T 3 --infeas-detect-mode heuristic_only \
  --test-contingency-filter 0

run_eval "E6B_heuristic_N1" "$CKPT_E1" \
  --adaptive-lm --T 3 --infeas-detect-mode heuristic_only \
  --test-contingency-filter 1

run_eval "E6B_heuristic_N2" "$CKPT_E1" \
  --adaptive-lm --T 3 --infeas-detect-mode heuristic_only \
  --test-contingency-filter 2

run_eval "E6B_heuristic_N3" "$CKPT_E1" \
  --adaptive-lm --T 3 --infeas-detect-mode heuristic_only \
  --test-data-dir "$N3_DIR" --test-contingency-filter 3

wait_all

# ═════════════════════════════════════════════════════════════════════════════
#  Goal 3: Topology transfer
# ═════════════════════════════════════════════════════════════════════════════
echo "============================================================"
echo " Topology transfer"
echo "============================================================"

if [[ -f "$CKPT_E3" ]]; then
  run_eval "E3_eval_n2_transfer" "$CKPT_E3" \
    --adaptive-lm --T 3 \
    --train-contingency-max 1 --test-contingency-filter 2
fi

if [[ -f "$CKPT_E5B" ]]; then
  run_eval "E5B_eval_dropout_n2" "$CKPT_E5B" \
    --adaptive-lm --T 3 \
    --train-contingency-max 1 --test-contingency-filter 2
fi

run_eval "E4_n3_transfer" "$CKPT_E1" \
  --adaptive-lm --T 3 \
  --test-data-dir "$N3_DIR" --test-contingency-filter 3

if [[ -f "$CKPT_E3" ]]; then
  run_eval "E3_eval_n3_transfer" "$CKPT_E3" \
    --adaptive-lm --T 3 \
    --train-contingency-max 1 \
    --test-data-dir "$N3_DIR" --test-contingency-filter 3
fi

if [[ -f "$CKPT_E5B" ]]; then
  run_eval "E5B_eval_n3_transfer" "$CKPT_E5B" \
    --adaptive-lm --T 3 \
    --train-contingency-max 1 --edge-dropout 0.05 \
    --test-data-dir "$N3_DIR" --test-contingency-filter 3
fi

wait_all

echo "============================================================"
echo " All inference runs complete!"
echo " Logs: ${LOG_DIR}/"
echo " W&B project: ${WANDB_PROJECT}"
echo "============================================================"
