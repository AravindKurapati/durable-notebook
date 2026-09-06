#!/usr/bin/env bash
# Overnight Gate 1 harvester for a TPD-capped Groq model.
#
# Groq's tokens-per-day limit is a continuously-refilling token bucket, not a
# midnight reset: 200,000/86,400s = ~139 tokens/min. The 429 bodies confirm it
# ("Used 199,639 ... try again in 9m27s" is exactly the time to refill the
# 1,313 tokens that request was short). So a capped model cannot be run in one
# batch, but CAN be drained slowly as the bucket refills.
#
# One durable-notebook episode costs roughly 5,500 tokens, which is about 40
# minutes of refill. Hence: one episode per cycle, one seed offset per cycle so
# the chunks are distinct episodes rather than the same one re-run, each to its
# own dump file for offline merging later.
#
# Deliberately NOT run against gpt-oss-20b or qwen3.6-27b: both score 0.000 on
# the VISIBLE bucket, i.e. they cannot answer even when the facts are still on
# screen, so their ablation numbers carry no information.

set -u
cd "$(dirname "$0")/.." || exit 1

MODEL="openai/gpt-oss-120b"
OUT="eval/results/drip"
STOP_AT="${1:-04:30}"          # local HH:MM to stop launching new cycles
SLEEP_SECS="${2:-2400}"        # 40 min, matched to the refill rate above

mkdir -p "$OUT"
LOG="$OUT/drip.log"

# Single-instance lock.
#
# Four copies of this script ended up running at once on 2026-08-18: stopping
# the harness task did not reap the detached bash/sleep children, so each
# "restart" added a loop instead of replacing one. They shared $LOG and the
# seed sequence, so they overwrote each other's dumps AND made four times the
# requests -- which defeats the entire point of pacing to the refill rate.
LOCK="$OUT/drip.lock"
if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "drip already running as PID $(cat "$LOCK"); refusing to start a second" | tee -a "$LOG"
  exit 1
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT INT TERM

echo "=== drip started $(date '+%F %T') pid=$$, stop at $STOP_AT, model $MODEL ===" >> "$LOG"

seed=100   # 0-99 reserved for the interactive runs already done
while [ "$(date +%H:%M)" '<' "$STOP_AT" ]; do
  # Resume-safe: a seed already harvested is not paid for twice.
  #
  # Tests for a USABLE rollout, not merely for the file. A cycle that 429s
  # still writes a dump (containing one errored output), so an existence
  # check would mark every rate-limited seed as done and never retry it --
  # silently discarding episodes on any restart.
  if [ -e "$OUT/seed${seed}.json" ]      && python -c "import json,sys; d=json.load(open(sys.argv[1],encoding='utf-8')); sys.exit(0 if any(not o.get('error') for o in d['outputs']) else 1)"         "$OUT/seed${seed}.json" 2>/dev/null; then
    seed=$((seed + 10))
    continue
  fi
  echo "--- cycle seed=$seed at $(date '+%F %T') ---" >> "$LOG"
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 python eval/gate1_persistence_ablation.py \
      --n-episodes 1 --n-turns 10 --n-questions 3 --compaction-window 3 \
      --model "$MODEL" --max-concurrent 1 --start-seed "$seed" \
      --dump "$OUT/seed${seed}.json" >> "$LOG" 2>&1
  echo "--- cycle seed=$seed exit=$? at $(date '+%F %T') ---" >> "$LOG"
  seed=$((seed + 10))
  sleep "$SLEEP_SECS"
done
echo "=== drip finished $(date '+%F %T') ===" >> "$LOG"
