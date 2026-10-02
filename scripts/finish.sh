#!/usr/bin/env bash
# Unattended finisher for the 10-seed campaign: waits for both machines,
# pulls remote results, runs the post-hoc sweeps, tables and figures.
#   nohup setsid bash scripts/finish.sh > results/finish.log 2>&1 < /dev/null &
# Progress: tail -f results/finish.log ; final report: results/FINISHED.txt
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
# remote worker running part of the suite (user@host) and its checkout path
REMOTE=${REMOTE:?set REMOTE=user@host of the second machine}
RDIR=${RDIR:-'~/mogapvqnn'}
MAX_WAIT_H=${MAX_WAIT_H:-10}
log() { echo "$(date '+%F %T') $*"; }

deadline=$(( $(date +%s) + MAX_WAIT_H * 3600 ))
log "waiting for local suite (seeds 3-9) to finish"
# anchored pattern: matches only the suite python process itself
while pgrep -f '^\.venv/bin/python -m mogapvqnn suite' >/dev/null; do
  [ "$(date +%s)" -gt "$deadline" ] && { log "timeout waiting for local suite"; break; }
  sleep 120
done
log "local suite finished: $(grep -cE '\] ' results/suite_seeds10.log) runs reported, $(grep -c FAILED results/suite_seeds10.log) failed"

log "waiting for remote suite to finish"
while true; do
  state=$(ssh -o BatchMode=yes -o ConnectTimeout=15 $REMOTE "pgrep -fc '^\./\.venv/bin/python -m mogapvqnn suite' ; grep -cE '\] ' $RDIR/results/suite_seeds10.log" 2>/dev/null | tr '\n' ' ')
  running=$(echo "$state" | awk '{print $1}'); done_n=$(echo "$state" | awk '{print $2}')
  if [ -n "$running" ] && [ "$running" -eq 0 ]; then log "remote suite finished ($done_n runs reported)"; break; fi
  [ "$(date +%s)" -gt "$deadline" ] && { log "timeout waiting for remote (state: $state); continuing with what exists"; break; }
  sleep 120
done

log "pulling remote results"
for i in 1 2 3; do
  rsync -az $REMOTE:$RDIR/results/runs/ results/runs/ && break
  log "rsync failed (attempt $i)"; sleep 60
done
mkdir -p results/logs_remote
rsync -az $REMOTE:$RDIR/results/logs/ results/logs_remote/ 2>/dev/null
rsync -az $REMOTE:$RDIR/results/thermal.log results/logs_remote/ 2>/dev/null

summary=$(for v in full no_proxy measured_noise exact_features; do
  echo "$v: $(find results/runs/$v -name 'seed*.json' 2>/dev/null | wc -l) runs"; done)
log "run counts:"; echo "$summary"

log "post-hoc readout + shot sweeps"
$PY scripts/posthoc.py --jobs 8 --workers 4
ph=$?

log "tables"
$PY -m mogapvqnn aggregate > results/aggregate.log 2>&1
ag=$?
log "figures"
$PY scripts/make_figures.py
fg=$?

{
  echo "Finished: $(date '+%F %T')"
  echo "$summary"
  echo "posthoc exit=$ph aggregate exit=$ag figures exit=$fg"
  echo "posthoc records: $(find results/posthoc -name 'seed*.json' | wc -l)"
  grep -h "CHECK MISMATCH\|FAILED" results/finish.log 2>/dev/null | head -5
  echo "Tables: results/tables/SUMMARY.md  Figures: results/figures/"
} > results/FINISHED.txt
log "done -> results/FINISHED.txt"
