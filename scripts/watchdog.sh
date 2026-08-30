#!/usr/bin/env bash
# Poll a rented box for ALL_DONE (or a hard deadline) and destroy the instance.
#
# The in-container `poweroff` in run_batch_selfstop.sh halts the container but
# does NOT stop vast.ai billing. Only `vastai destroy` from outside does. Run
# this locally, backgrounded, alongside any unattended batch.
#
#   ./scripts/watchdog.sh <instance_id> <ssh_host> <ssh_port> [max_hours]
#
# Pulls EVERYTHING under results/. An earlier version globbed `results/gpu-*`,
# hardcoded from the first batch. A later run wrote `fragcount-*`, the glob
# matched nothing, and the box was destroyed with the results still on it.
# Never narrow this glob to the filenames of one experiment.
set -uo pipefail
ID=$1; HOST=$2; PORT=$3; MAX_H=${4:-8}
KEY=~/.ssh/vastai_kvadmission
DEADLINE=$(( $(date +%s) + MAX_H * 3600 ))

while :; do
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    echo "watchdog: deadline hit, destroying $ID"; break
  fi
  out=$(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -i "$KEY" \
        -p "$PORT" "root@$HOST" 'tail -3 ~/kvAdmission/batch.log 2>/dev/null' 2>/dev/null)
  if printf '%s' "$out" | grep -q ALL_DONE; then
    echo "watchdog: ALL_DONE seen, pulling results then destroying $ID"
    mkdir -p results
    scp -r -o StrictHostKeyChecking=no -i "$KEY" -P "$PORT" \
        "root@$HOST:~/kvAdmission/results/." results/
    if [ $? -ne 0 ]; then
      echo "watchdog: RESULT PULL FAILED - leaving $ID alive so nothing is lost"
      echo "watchdog: destroy it by hand once results are safe"
      exit 1
    fi
    n=$(ls results/ 2>/dev/null | wc -l)
    echo "watchdog: pulled $n files"
    break
  fi
  sleep 300
done

yes | vastai destroy instance "$ID"
sleep 5
vastai show instances --raw | python3 -c 'import json,sys; print("instances remaining:", len(json.load(sys.stdin)))'
