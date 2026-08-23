#!/usr/bin/env bash
# Progress of the Table 2 / Table 3 retraining sweep.
#   bash scripts/t23_status.sh            # one-shot
#   watch -n 300 bash scripts/t23_status.sh
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_TAG="${RUN_TAG:-paper_t23_100k}"
TARGET="${TARGET:-100000}"
TOTAL_RUNS="${TOTAL_RUNS:-21}"

printf '%-34s %10s %7s %10s %s\n' RUN ITER PCT ETA STATE
done_n=0; run_n=0
slowest=0
for d in outputs/gaze2hoi/${RUN_TAG}_*/; do
  [ -d "$d" ] || continue
  name=$(basename "$d")
  if [ -f "$d/iteration_0$(printf '%06d' $((TARGET/10*10)))0.pth" ] 2>/dev/null; then :; fi
  last=$(grep -o "Iteration [0-9]*/${TARGET}" "$d/log.txt" 2>/dev/null | tail -1 | sed 's/Iteration //;s#/.*##')
  last=${last:-0}
  pct=$(( last * 100 / TARGET ))
  if [ -f "$d/iteration_0100000.pth" ] || [ "$last" -ge "$TARGET" ]; then
    state=DONE; eta="-"; done_n=$((done_n+1))
  else
    # rate from the log's first and last timestamped iteration lines
    first_t=$(grep -o "^\[[^]]*\] \[.*Iteration [0-9]*/" "$d/log.txt" 2>/dev/null | head -1 | cut -d']' -f1 | tr -d '[')
    last_t=$(grep -o "^\[[^]]*\] \[.*Iteration [0-9]*/" "$d/log.txt" 2>/dev/null | tail -1 | cut -d']' -f1 | tr -d '[')
    first_i=$(grep -o "Iteration [0-9]*/${TARGET}" "$d/log.txt" 2>/dev/null | head -1 | sed 's/Iteration //;s#/.*##')
    eta="?"
    if [ -n "$first_t" ] && [ -n "$last_t" ] && [ "${first_i:-0}" -lt "$last" ]; then
      s=$(( $(date -d "$last_t" +%s) - $(date -d "$first_t" +%s) ))
      di=$(( last - first_i ))
      if [ "$s" -gt 0 ] && [ "$di" -gt 0 ]; then
        rem=$(( (TARGET - last) * s / di ))
        eta=$(printf '%dh%02dm' $((rem/3600)) $(((rem%3600)/60)))
        [ "$rem" -gt "$slowest" ] && slowest=$rem
      fi
    fi
    if pgrep -f "gaze2hoi.exp.name=${name}" >/dev/null; then state=RUNNING; run_n=$((run_n+1)); else state=STOPPED; fi
  fi
  printf '%-34s %10s %6s%% %10s %s\n' "$name" "$last" "$pct" "$eta" "$state"
done

echo
if [ "$done_n" -eq 0 ] && [ "$run_n" -eq 0 ]; then
  if pgrep -f retrain_table23_100k.sh >/dev/null; then
    echo "queued: waiting for the obj_loss scoring jobs to release the GPUs"
  else
    echo "not started"
  fi
fi
echo "runs finished: ${done_n}/${TOTAL_RUNS}   running now: ${run_n}"
if [ "$slowest" -gt 0 ]; then
  waves=$(( (TOTAL_RUNS - done_n + 3) / 4 ))
  echo "slowest active run ETA: $(printf '%dh%02dm' $((slowest/3600)) $(((slowest%3600)/60)))   remaining waves: ~${waves}"
fi
