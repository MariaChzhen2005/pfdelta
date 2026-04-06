#!/usr/bin/env bash
# ───────────────────
# monitor.sh — Live dashboard for parallel experiment runs.
#
# Usage:
#   bash scripts/monitor.sh              # default: watch logs/
#   bash scripts/monitor.sh logs/        # explicit log directory
#   bash scripts/monitor.sh -f           # follow mode (like tail -f)
#
# Shows the last status line from each .log file, refreshing every 2s.
# ───────────────────
set -euo pipefail

FOLLOW=false
LOG_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -f|--follow) FOLLOW=true; shift ;;
    *) LOG_DIR="$1"; shift ;;
  esac
done

LOG_DIR="${LOG_DIR:-logs}"

if [[ ! -d "$LOG_DIR" ]]; then
  echo "Log directory not found: $LOG_DIR"
  exit 1
fi

# ─── Formatting helpers ──────────────────────────
BOLD='\033[1m'
GREEN='\033[32m'
YELLOW='\033[33m'
RED='\033[31m'
CYAN='\033[36m'
DIM='\033[2m'
RESET='\033[0m'

status_icon() {
  local line="$1"
  if [[ "$line" == *"Early stopping"* ]] || [[ "$line" == *"DONE"* ]] || [[ "$line" == *"Saved"* ]] || [[ "$line" == *"final"* ]]; then
    printf "${GREEN}✓${RESET}"
  elif [[ "$line" == *"error"* ]] || [[ "$line" == *"Error"* ]] || [[ "$line" == *"FAILED"* ]] || [[ "$line" == *"Traceback"* ]]; then
    printf "${RED}✗${RESET}"
  else
    printf "${YELLOW}⋯${RESET}"
  fi
}

print_dashboard() {
  local files=("$LOG_DIR"/*.log)
  if [[ ! -e "${files[0]}" ]]; then
    echo "No log files in $LOG_DIR"
    return
  fi

  local now
  now=$(date "+%H:%M:%S")
  printf "${BOLD}═══ Experiment Monitor (%s) ═══${RESET}\n" "$now"
  printf "${DIM}%-36s %6s  %s${RESET}\n" "RUN" "EPOCH" "STATUS"
  printf "${DIM}%s${RESET}\n" "────────────────────────────────────────────────────────────────────────────"

  for f in "${files[@]}"; do
    local name
    name=$(basename "$f" .log)

    # Check if process is still running (look for PID in parent script's log)
    local pid_alive=""
    local pid
    pid=$(pgrep -f "wandb-run-name ${name}" 2>/dev/null || true)
    if [[ -n "$pid" ]]; then
      pid_alive="${CYAN}[running]${RESET}"
    fi

    # Try to find the last epoch line
    local last_epoch_line
    last_epoch_line=$(grep -E "^[0-9:]+\s+INFO\s+Ep [0-9]+" "$f" 2>/dev/null | tail -1 || true)

    if [[ -n "$last_epoch_line" ]]; then
      # Extract epoch number and summary
      local ep_num summary
      ep_num=$(echo "$last_epoch_line" | grep -oE "Ep [0-9]+" | head -1 | grep -oE "[0-9]+")
      summary=$(echo "$last_epoch_line" | sed -E 's/^.*Ep [0-9]+ \(stage [0-9]+\)\s+//')
      local icon
      icon=$(status_icon "$last_epoch_line")
      printf " %b %-34s %4s   %s %b\n" "$icon" "$name" "$ep_num" "$summary" "$pid_alive"
    else
      # Maybe it's an eval run or hasn't started — show last meaningful line
      local last_line
      last_line=$(tail -1 "$f" 2>/dev/null || echo "(empty)")
      # Truncate long lines
      if [[ ${#last_line} -gt 70 ]]; then
        last_line="${last_line:0:67}..."
      fi

      # Check for completion or error
      local has_error=""
      if grep -qiE "Traceback|Error|Exception" "$f" 2>/dev/null; then
        has_error=true
      fi
      local has_done=""
      if grep -qiE "saved|complete|Done" "$f" 2>/dev/null; then
        has_done=true
      fi

      local icon
      if [[ -n "$has_done" ]]; then
        icon="${GREEN}✓${RESET}"
      elif [[ -n "$has_error" ]]; then
        icon="${RED}✗${RESET}"
      elif [[ -n "$pid_alive" ]]; then
        icon="${YELLOW}⋯${RESET}"
      else
        icon="${DIM}?${RESET}"
      fi
      printf " %b %-34s %4s   %s %b\n" "$icon" "$name" "—" "$last_line" "$pid_alive"
    fi
  done

  # Summary counts
  local total running done errored
  total=${#files[@]}
  running=$(pgrep -f "wandb-run-name" 2>/dev/null | wc -l | tr -d ' ')
  done=$(grep -rlE "saved|complete|Done" "$LOG_DIR"/*.log 2>/dev/null | wc -l | tr -d ' ')
  errored=$(grep -rlE "Traceback|FAILED" "$LOG_DIR"/*.log 2>/dev/null | wc -l | tr -d ' ')

  printf "\n${DIM}%d total  │  ${CYAN}%d running${DIM}  │  ${GREEN}%d done${DIM}  │  ${RED}%d error${RESET}\n" \
    "$total" "$running" "$done" "$errored"
}

# ─── Main ────────────────────────────────────────
if $FOLLOW; then
  while true; do
    clear
    print_dashboard
    sleep 2
  done
else
  print_dashboard
fi
