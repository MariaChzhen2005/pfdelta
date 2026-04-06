#!/usr/bin/env bash
# ───────────────────
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
# ───────────────────
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
--batch-size 128 --epochs-stage1 80 --epochs-stage2 20 --early-stop-patience 15 \
--seed 42 --amp"

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
  # Pass the same flags that will be used for training to get the deterministic signature
  python models.py $COMMON "$@" --print-signature --cpu 2>/dev/null
}

# ─── Helper: run a training job ─────────────────
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
#  Pre-compute run signatures so Phase 2 can find the right checkpoints
# ═════════════════════════════════════════════════════════════════════════════
echo "============================================================"
echo " Computing run signatures ..."
echo "============================================================"

SIG_E1=$(get_sig --adaptive-lm --T 5 --stage both)
SIG_E3=$(get_sig --adaptive-lm --T 5 --stage both --train-contingency-max 1)
SIG_E5B=$(get_sig --adaptive-lm --T 5 --stage both --train-contingency-max 1 --edge-dropout 0.05)
SIG_VANILLA=$(get_sig --adaptive-lm --T 5 --lambda-infeasibility 0)

echo "  E1  (adaptive LM, full data):  ${SIG_E1}"
echo "  E3  (adaptive LM, n+n-1):     ${SIG_E3}"
echo "  E5B (adaptive LM, n+n-1, dropout): ${SIG_E5B}"
echo "  VANILLA (pure GNN regression): ${SIG_VANILLA}"

# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 1: Training runs (parallelised across GPUs)
# ═════════════════════════════════════════════════════════════════════════════
echo "============================================================"
echo " PHASE 1: Training (4 runs)"
echo "============================================================"

# E1 — Full model: adaptive LM, full data, both stages
run_train "E1_baseline_adaptive_lm" \
  --adaptive-lm --T 5 --stage both

# B_VANILLA — Pure GNN regression (stage 1 only, no infeasibility loss)
run_train "B_VANILLA_GNN" \
  --adaptive-lm --T 5 --stage 1 --lambda-infeasibility 0

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
#  Locate checkpoints by signature
# ═════════════════════════════════════════════════════════════════════════════
CKPT_E1="${CKPT_DIR}/final_model_${SIG_E1}.pt"
CKPT_E3="${CKPT_DIR}/final_model_${SIG_E3}.pt"
CKPT_E5B="${CKPT_DIR}/final_model_${SIG_E5B}.pt"
CKPT_VANILLA="${CKPT_DIR}/final_model_${SIG_VANILLA}.pt"
# Stage-1-only checkpoint from E1 (for concern 2: clean GNN-only baseline)
CKPT_E1_S1="${CKPT_DIR}/epoch_80_stage1_final_${SIG_E1}.pt"

for name_ckpt in "E1:${CKPT_E1}" "E1_S1:${CKPT_E1_S1}" "E3:${CKPT_E3}" "E5B:${CKPT_E5B}" "VANILLA:${CKPT_VANILLA}"; do
  name="${name_ckpt%%:*}"
  ckpt="${name_ckpt#*:}"
  if [[ -f "$ckpt" ]]; then
    echo "  ${name} checkpoint found: ${ckpt}"
  else
    echo "  WARNING: ${name} checkpoint NOT found: ${ckpt}"
  fi
done

# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 2: Evaluation-only runs
# ═════════════════════════════════════════════════════════════════════════════
echo "============================================================"
echo " PHASE 2: Evaluation"
echo "============================================================"

# ─── Concern 1: Baselines for external comparison ────────────────────────────
echo "--- Concern 1: Baselines ---"

# B_FLAT_T5:  Flat-start LM baseline at T=5  (same budget as model)
run_train "B_FLAT_T5" \
  --adaptive-lm --T 5 --max-iter-inference 5 --baseline flat-start

# B_FLAT_T20: Flat-start LM baseline at T=20 (generous budget)
run_train "B_FLAT_T20" \
  --adaptive-lm --T 5 --max-iter-inference 20 --baseline flat-start

# B_FLAT_T50: Flat-start LM baseline at T=50 (very generous)
run_train "B_FLAT_T50" \
  --adaptive-lm --T 5 --max-iter-inference 50 --baseline flat-start

wait_all

# ─── Concern 2: Clean stage-1-only GNN baseline 
echo "--- Concern 2: GNN-only baselines ---"

# B_VANILLA_T0: Pure GNN regression (no infeas head, no solver)
if [[ -f "$CKPT_VANILLA" ]]; then
  run_eval "B_VANILLA_T0_baseline" "$CKPT_VANILLA" \
    --adaptive-lm --T 5 --max-iter-inference 0 --lambda-infeasibility 0
fi

# E_S1_T0:  Stage-1-only checkpoint, no solver (raw GNN quality)
if [[ -f "$CKPT_E1_S1" ]]; then
  run_eval "E_S1_T0_gnn_direct" "$CKPT_E1_S1" \
    --adaptive-lm --T 5 --max-iter-inference 0

  # E_S1_T5:  Stage-1-only checkpoint, T=5 solver at inference
  run_eval "E_S1_T5_plus_solver" "$CKPT_E1_S1" \
    --adaptive-lm --T 5 --max-iter-inference 5

  # E_S1_T10: Stage-1-only checkpoint, T=10 solver
  run_eval "E_S1_T10_plus_solver" "$CKPT_E1_S1" \
    --adaptive-lm --T 5 --max-iter-inference 10
fi

wait_all

# ─── Goal 1: Accurate state estimation ──────────
echo "--- Goal 1: Accurate state estimation ---"

# E2: Per-κ-bin breakdown (E1 checkpoint, default test set)
run_eval "E2_per_bin_breakdown" "$CKPT_E1" \
  --adaptive-lm --T 5

# E7_T0: GNN-only baseline — no solver iterations (warm-start quality)
run_eval "E7_T0_gnn_only" "$CKPT_E1" \
  --adaptive-lm --T 5 --max-iter-inference 0

# E7_T1/T5/T10/T20: Iteration-vs-accuracy sweep (concern 8: wall-clock)
run_eval "E7_T1_iters" "$CKPT_E1" \
  --adaptive-lm --T 5 --max-iter-inference 1

run_eval "E7_T5_iters" "$CKPT_E1" \
  --adaptive-lm --T 5 --max-iter-inference 5

run_eval "E7_T10_iters" "$CKPT_E1" \
  --adaptive-lm --T 5 --max-iter-inference 10

wait_all

# ─── Concern 4: Per-contingency-order stratified eval on E1 ─────────────────
echo "--- Concern 4: E1 per-contingency stratified ---"

# E1_N0: E1 evaluated on N-only (base topology) test slice
run_eval "E1_N0_base" "$CKPT_E1" \
  --adaptive-lm --T 5 --test-contingency-filter 0

# E1_N1: E1 evaluated on N-1 test slice
run_eval "E1_N1_contingency" "$CKPT_E1" \
  --adaptive-lm --T 5 --test-contingency-filter 1

# E1_N2: E1 evaluated on N-2 test slice (concern 3: fills the gap)
run_eval "E1_N2_contingency" "$CKPT_E1" \
  --adaptive-lm --T 5 --test-contingency-filter 2

wait_all

# ─── Goal 2: Reliable feasible/infeasible near the boundary ──────────────────
echo "--- Goal 2: Infeasibility detection ---"

# E6A: Infeasibility ablation — learned head only
run_eval "E6A_infeas_learned_only" "$CKPT_E1" \
  --adaptive-lm --T 5 --infeas-detect-mode learned_only

# E6B: Infeasibility ablation — heuristic only (mu-based)
run_eval "E6B_infeas_heuristic_only" "$CKPT_E1" \
  --adaptive-lm --T 5 --infeas-detect-mode heuristic_only

# E6C skipped — combined mode is the default, identical to E2

wait_all

# ─── Must-have 1: Per-contingency-order calibration (ECE/AUROC per slice) ────
echo "--- Must-have 1: Per-contingency infeasibility calibration ---"

# E6A learned-only, stratified by contingency order
run_eval "E6A_learned_N0" "$CKPT_E1" \
  --adaptive-lm --T 5 --infeas-detect-mode learned_only \
  --test-contingency-filter 0

run_eval "E6A_learned_N1" "$CKPT_E1" \
  --adaptive-lm --T 5 --infeas-detect-mode learned_only \
  --test-contingency-filter 1

run_eval "E6A_learned_N2" "$CKPT_E1" \
  --adaptive-lm --T 5 --infeas-detect-mode learned_only \
  --test-contingency-filter 2

run_eval "E6A_learned_N3" "$CKPT_E1" \
  --adaptive-lm --T 5 --infeas-detect-mode learned_only \
  --test-data-dir "$N3_DIR" --test-contingency-filter 3

wait_all

# E6B heuristic-only, stratified by contingency order
run_eval "E6B_heuristic_N0" "$CKPT_E1" \
  --adaptive-lm --T 5 --infeas-detect-mode heuristic_only \
  --test-contingency-filter 0

run_eval "E6B_heuristic_N1" "$CKPT_E1" \
  --adaptive-lm --T 5 --infeas-detect-mode heuristic_only \
  --test-contingency-filter 1

run_eval "E6B_heuristic_N2" "$CKPT_E1" \
  --adaptive-lm --T 5 --infeas-detect-mode heuristic_only \
  --test-contingency-filter 2

run_eval "E6B_heuristic_N3" "$CKPT_E1" \
  --adaptive-lm --T 5 --infeas-detect-mode heuristic_only \
  --test-data-dir "$N3_DIR" --test-contingency-filter 3

wait_all

# ─── Goal 3: Robustness to unseen contingencies 
echo "--- Goal 3: Topology transfer ---"

# E3_eval: N+N-1 → N-2 transfer (adaptive LM, no dropout)
if [[ -f "$CKPT_E3" ]]; then
  run_eval "E3_eval_n2_transfer" "$CKPT_E3" \
    --adaptive-lm --T 5 \
    --train-contingency-max 1 --test-contingency-filter 2
fi

# E5B_eval: N+N-1 → N-2 transfer (adaptive LM, with edge dropout)
if [[ -f "$CKPT_E5B" ]]; then
  run_eval "E5B_eval_dropout_n2" "$CKPT_E5B" \
    --adaptive-lm --T 5 \
    --train-contingency-max 1 --test-contingency-filter 2
fi

# E4: N-{1,2} → N-3 transfer (E1, full-trained, N-3 test set)
run_eval "E4_n3_transfer" "$CKPT_E1" \
  --adaptive-lm --T 5 \
  --test-data-dir "$N3_DIR" --test-contingency-filter 3

# E3 → N-3 transfer (trained on N+N-1 only, no dropout)
if [[ -f "$CKPT_E3" ]]; then
  run_eval "E3_eval_n3_transfer" "$CKPT_E3" \
    --adaptive-lm --T 5 \
    --train-contingency-max 1 \
    --test-data-dir "$N3_DIR" --test-contingency-filter 3
fi

# E5B → N-3 transfer (trained on N+N-1, with edge dropout)
if [[ -f "$CKPT_E5B" ]]; then
  run_eval "E5B_eval_n3_transfer" "$CKPT_E5B" \
    --adaptive-lm --T 5 \
    --train-contingency-max 1 --edge-dropout 0.05 \
    --test-data-dir "$N3_DIR" --test-contingency-filter 3
fi

wait_all

echo "============================================================"
echo " All experiments complete!"
echo " Logs: ${LOG_DIR}/"
echo " Checkpoints: ${CKPT_DIR}/"
echo " W&B project: ${WANDB_PROJECT}"
echo "============================================================"
