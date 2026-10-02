#!/usr/bin/env bash
# Unattended finisher when a suite is split across two machines with
# `mogapvqnn suite ... --shard K --num-shards 2` (or --job-list): waits for the
# local and the remote suite, pulls the remote run records, then runs the
# post-hoc sweeps, tables and figures.
#
#   REMOTE=user@host RDIR='~/mogapvqnn' \
#   nohup setsid bash scripts/distributed/finish.sh > results/_local/finish.log 2>&1 < /dev/null &
#
# Progress: tail -f results/_local/finish.log ; final report: results/_local/FINISHED.txt
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
REMOTE=${REMOTE:?set REMOTE=user@host of the second machine}
RDIR=${RDIR:-'~/mogapvqnn'}
MAX_WAIT_H=${MAX_WAIT_H:-10}
JOBS=${JOBS:-8}
LOCAL=results/_local
mkdir -p "$LOCAL"
log() { echo "$(date '+%F %T') $*"; }

deadline=$(( $(date +%s) + MAX_WAIT_H * 3600 ))
log "waiting for the local suite to finish"
# anchored pattern: matches only the suite python process itself
while pgrep -f '^\.venv/bin/python -m mogapvqnn suite' >/dev/null; do
  [ "$(date +%s)" -gt "$deadline" ] && { log "timeout waiting for local suite"; break; }
  sleep 120
done
log "local suite finished"

log "waiting for the remote suite to finish"
while true; do
  running=$(ssh -o BatchMode=yes -o ConnectTimeout=15 "$REMOTE" \
            "pgrep -fc '^\./\.venv/bin/python -m mogapvqnn suite'" 2>/dev/null)
  if [ -n "$running" ] && [ "$running" -eq 0 ]; then log "remote suite finished"; break; fi
  [ "$(date +%s)" -gt "$deadline" ] && { log "timeout waiting for remote; continuing with what exists"; break; }
  sleep 120
done

log "pulling remote run records"
for i in 1 2 3; do
  rsync -az "$REMOTE:$RDIR/results/runs/" results/runs/ && break
  log "rsync failed (attempt $i)"; sleep 60
done
rsync -az "$REMOTE:$RDIR/results/_local/logs/" "$LOCAL/logs_remote/" 2>/dev/null

summary=$(for v in full no_proxy measured_noise exact_features; do
  echo "$v: $(find results/runs/$v -name 'seed*.json' 2>/dev/null | wc -l) runs"; done)
log "run counts:"; echo "$summary"

log "post-hoc readout + shot sweeps"; $PY scripts/posthoc.py --jobs "$JOBS"; ph=$?
log "tables";  $PY -m mogapvqnn aggregate > "$LOCAL/aggregate.log" 2>&1; ag=$?
log "figures"; $PY scripts/make_figures.py; fg=$?

{
  echo "Finished: $(date '+%F %T')"
  echo "$summary"
  echo "posthoc exit=$ph aggregate exit=$ag figures exit=$fg"
  echo "posthoc records: $(find results/posthoc -name 'seed*.json' | wc -l)"
  echo "Tables: results/tables/  Figures: results/figures/"
} > "$LOCAL/FINISHED.txt"
log "done -> $LOCAL/FINISHED.txt"
