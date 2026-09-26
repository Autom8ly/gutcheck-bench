#!/usr/bin/env bash
# Start a /v1/systemone server, wait until it answers, record peak GPU memory, score one or more suites with
# gutcheck, then stop the server.
#
#   harness/bench.sh <name> <port> <server-workdir> <model-name> <suite[,suite...]> -- <server command...>
#
# Suites are gutcheck JSONL files or Kev frozen-suite directories. Results land in $RESULTS/<name>/<suite>/
# (default ./results/runs). GUTCHECK_DIR is this repository (default: the directory above this script).
set -u
export PATH=$HOME/.local/bin:$PATH
NAME=$1 PORT=$2 WORKDIR=$3 MODEL=$4 SUITES=$5; shift 5; [ "${1:-}" = "--" ] && shift
GUTCHECK_DIR=${GUTCHECK_DIR:-$(cd "$(dirname "$0")/.." && pwd)}
RESULTS=${RESULTS:-$GUTCHECK_DIR/results/runs}
R=$RESULTS/$NAME; mkdir -p "$R"
PY="uv run --no-project --with numpy python"

cd "$WORKDIR"
setsid "$@" > "$R/server.log" 2>&1 < /dev/null & SPID=$!
( peak=0; while sleep 0.5; do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "$m" -gt "$peak" ] && { peak=$m; echo $peak > "$R/peak_vram_mib"; }; done ) & MPID=$!
cleanup() { kill -- -$SPID 2>/dev/null; kill $MPID 2>/dev/null; sleep 3; kill -9 -- -$SPID 2>/dev/null; }
trap cleanup EXIT

t0=$(date +%s)
for i in $(seq 1 2160); do   # up to 3 h: a first start may download the model
  curl -sf -o /dev/null "http://127.0.0.1:$PORT/v1/models" && break
  curl -sf -o /dev/null "http://127.0.0.1:$PORT/health" && break
  kill -0 $SPID 2>/dev/null || { echo "SERVER DIED"; tail -30 "$R/server.log"; exit 1; }
  sleep 5
done
curl -sf -o /dev/null "http://127.0.0.1:$PORT/v1/models" || curl -sf -o /dev/null "http://127.0.0.1:$PORT/health" || { echo "SERVER NEVER READY"; exit 1; }
echo "ready after $(( $(date +%s) - t0 ))s; idle VRAM $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"

cd "$GUTCHECK_DIR"
IFS=',' read -ra LIST <<< "$SUITES"
for S in "${LIST[@]}"; do
  SN=$(basename "${S%.jsonl}")
  echo "== $NAME on $SN"
  rm -rf "$R/$SN"
  $PY -m gutcheck.benchmark --endpoint "http://127.0.0.1:$PORT" --model "$MODEL" --suite "$S" --out "$R/$SN" 2>&1 | tail -12
done
echo "peak VRAM MiB: $(cat "$R/peak_vram_mib" 2>/dev/null)"
