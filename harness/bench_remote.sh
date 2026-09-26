#!/usr/bin/env bash
# Usage: bench_remote.sh <name> <port> <workdir> <model-id-for-request> <suite-or-data> -- <server command...>
# Starts a /v1/systemone server, waits for it, scores a suite with kev.benchmark --remote, records peak VRAM, stops it.
set -u
NAME=$1 PORT=$2 WORKDIR=$3 MODEL=$4 SUITE=$5; shift 5; [ "$1" = "--" ] && shift
export PATH=$HOME/.local/bin:$PATH HF_HOME=$HOME/jev-eval/hf
R=$HOME/jev-eval/results/$NAME; rm -rf "$R"; mkdir -p "$R"
cd "$WORKDIR"
setsid "$@" > "$R/server.log" 2>&1 < /dev/null & SPID=$!
cleanup() { kill -- -$SPID 2>/dev/null; kill $MPID 2>/dev/null; sleep 3; kill -9 -- -$SPID 2>/dev/null; }
trap cleanup EXIT
# peak VRAM sampler
( peak=0; while sleep 0.5; do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$m" -gt "$peak" ] && { peak=$m; echo $peak > "$R/peak_vram_mib"; }; done ) & MPID=$!
t0=$(date +%s)
for i in $(seq 1 2160); do   # up to 3h: first start may download the model
  curl -sf -o /dev/null "http://127.0.0.1:$PORT/v1/models" && break
  curl -sf -o /dev/null "http://127.0.0.1:$PORT/health" && break
  kill -0 $SPID 2>/dev/null || { echo "SERVER DIED"; tail -30 "$R/server.log"; exit 1; }
  sleep 5
done
curl -sf -o /dev/null "http://127.0.0.1:$PORT/v1/models" || curl -sf -o /dev/null "http://127.0.0.1:$PORT/health" || { echo "SERVER NEVER READY"; exit 1; }
echo "ready after $(( $(date +%s) - t0 ))s; idle VRAM $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
cd $HOME/jev-eval/kev
IFS=',' read -ra SUITES <<< "$SUITE"
for S in "${SUITES[@]}"; do
  if [ -f "$S" ]; then SARG="--data $S"; else SARG="--suite $S"; fi
  echo "== $NAME on $(basename $S)"
  uv run --no-sync python -m kev.benchmark --remote "http://127.0.0.1:$PORT" --remote-model "$MODEL" $SARG --out "$R/run-$(basename ${S%.jsonl})" 2>&1 | tail -25
done
echo "peak VRAM MiB: $(cat $R/peak_vram_mib 2>/dev/null)"
