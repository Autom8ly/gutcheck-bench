#!/usr/bin/env bash
# Every setup from the article, one at a time on one 8 GB GPU, on the independent jabr v2 suite; then hosted Jev.
# Tool checkouts live under $TOOLS/<tool> (default ~/tools); set HF_HOME to share one model cache across tools.
# Hosted Jev reads its key from JEV_KEY_FILE (default ~/.config/typesafe/jev.key) or the JEV_API_KEY variable.
set -u
export PATH=$HOME/.local/bin:$PATH
GUTCHECK_DIR=${GUTCHECK_DIR:-$(cd "$(dirname "$0")/.." && pwd)}; export GUTCHECK_DIR
export RESULTS=${RESULTS:-$GUTCHECK_DIR/results/runs}
JEV_KEY_FILE=${JEV_KEY_FILE:-$HOME/.config/typesafe/jev.key}
B="$GUTCHECK_DIR/harness/bench.sh"; S="$GUTCHECK_DIR/suites/jabr-v2.jsonl"; T=${TOOLS:-$HOME/tools}
K="env KEV_FUSED=0 KEV_CUDA_GRAPHS=0"; UV=~/.local/bin/uv

$B von 8101 $T/von von-latest "$S" -- $UV run von serve --host 127.0.0.1 --port 8101 --device cuda
LAYA_PORT=8102 LAYA_HOST=127.0.0.1 $B laya 8102 $T/laya laya "$S" -- .venv/bin/laya-serve
$B rizzo-4b-q8 8103 $T/rizzo-flow rizzo-latest "$S" -- $UV run --no-sync rizzo serve --size 4b --quant q8_0 --device cuda --port 8103
$B rizzo-4b-q4 8105 $T/rizzo-flow rizzo-latest "$S" -- $UV run --no-sync rizzo serve --size 4b --quant q4_k_m --device cuda --port 8105
$B rizzo-1.7b-q8 8104 $T/rizzo-flow rizzo-latest "$S" -- $UV run --no-sync rizzo serve --size 1.7b --quant q8_0 --device cuda --port 8104
$B nanojev-0.6b 8208 $T/NanoJev nanojev "$S" -- env PORT=8208 .venv/bin/python nanojev_server.py
$B kev-0.8b 8008 $T/kev kev-latest "$S" -- $UV run --no-sync python -m kev.serve --run jaredpalmer/kev-0.8b --port 8008
$B kev-4b-4bit 8202 $T/kev kev-latest "$S" -- $K KEV_QUANT=4bit $UV run --no-sync python -m kev.serve --run jaredpalmer/kev-4b --port 8202
$B kev-4b-8bit 8201 $T/kev kev-latest "$S" -- $K KEV_QUANT=8bit $UV run --no-sync python -m kev.serve --run jaredpalmer/kev-4b --port 8201
$B kev-9b-4bit 8203 $T/kev kev-latest "$S" -- $K KEV_QUANT=4bit PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $UV run --no-sync python -m kev.serve --run jaredpalmer/kev-9b --port 8203
$B semif-qwen3.5-4b-4bit 8205 $T/SemIf-OpenJev semif "$S" -- env SEMIF_QUANT=4bit PORT=8205 .venv/bin/python semif_server.py
$B semif-qwen3.5-4b-8bit 8204 $T/SemIf-OpenJev semif "$S" -- env SEMIF_QUANT=8bit PORT=8204 .venv/bin/python semif_server.py
$B ollama-qwen3-8b 8206 $GUTCHECK_DIR/adapters ollama "$S" -- env PORT=8206 OLLAMA_MODEL=qwen3:8b python3 ollama_server.py
curl -s localhost:11434/api/generate -d '{"model":"qwen3:8b","keep_alive":0}' >/dev/null

# hosted Jev: no server to start
cd "$GUTCHECK_DIR" && rm -rf "$RESULTS/jev-hosted/jabr-v2" && mkdir -p "$RESULTS/jev-hosted" && \
  uv run --no-project --with numpy python -m gutcheck.benchmark --endpoint https://api.typesafe.ai --model jev-latest \
    --suite "$S" --out "$RESULTS/jev-hosted/jabr-v2" $( [ -f "$JEV_KEY_FILE" ] && echo --api-key-file "$JEV_KEY_FILE" ) 2>&1 | tail -12
echo "JABR V2 DONE"
