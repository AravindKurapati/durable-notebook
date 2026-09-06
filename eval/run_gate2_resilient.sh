#!/bin/bash
# Self-relaunching driver for gate2_exploitability_pilot.py.
#
# Background bash tasks in this sandbox have been killed mid-run twice in a
# row on this exact job (2026-08-29, 2026-08-30), with no error in the
# process's own output -- consistent with the known Windows
# background-process-reliability issue, not a TPD or code problem. Gate 2's
# per-model checkpointing (--dump-dir) already protects completed models,
# but a kill mid-model wastes that model's in-flight rollouts, since
# checkpointing is per-model, not per-episode. This loop just keeps
# relaunching the same command until every model named in $MODELS has a
# checkpoint pair on disk, so a kill costs at most one model's worth of
# rework instead of requiring a human to notice and restart by hand.
set -u
cd "$(dirname "$0")/.."

DUMP_DIR="${DUMP_DIR:-eval/results/gate2_real_n15}"
MODELS="${MODELS:-openai/gpt-oss-120b openai/gpt-oss-20b qwen/qwen3.6-27b}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-8}"

all_checkpointed() {
    for m in $MODELS; do
        safe=$(echo "$m" | sed 's/[^A-Za-z0-9_.-]/_/g')
        [ -f "$DUMP_DIR/${safe}__naive.json" ] || return 1
        [ -f "$DUMP_DIR/${safe}__hardened.json" ] || return 1
    done
    return 0
}

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    if all_checkpointed; then
        echo "[driver] all models already checkpointed, done."
        exit 0
    fi
    echo "[driver] attempt $attempt/$MAX_ATTEMPTS: launching gate2_exploitability_pilot.py"
    python eval/gate2_exploitability_pilot.py \
        --n-episodes 15 --verbose --dump-dir "$DUMP_DIR" --models $MODELS
    status=$?
    echo "[driver] attempt $attempt exited with status $status"
    if all_checkpointed; then
        echo "[driver] all models checkpointed after attempt $attempt, done."
        exit 0
    fi
    echo "[driver] not all models checkpointed yet, retrying in 15s"
    sleep 15
done

echo "[driver] gave up after $MAX_ATTEMPTS attempts, still missing checkpoints"
exit 1
