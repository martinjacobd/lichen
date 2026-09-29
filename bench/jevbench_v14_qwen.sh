#!/usr/bin/env bash
# Qwen3.8-27B-FP8 through lichen backend-split (masked read) on JevBench v1.4 public, 231 items.
# vLLM on :8010, processed_logprobs, MAX_NUM_SEQS=1 (deterministic: read noise measured 0).
set -euo pipefail
LICHEN=$HOME/lichen-mask; JB=$HOME/jevbench; OUT=$HOME/lichen-exp/v14-qwen38-mask; PORT=8766
IMAGE=(--repeat 2 --permute --batch --fibers 2 --fiber-map --shrink --temperature 1.25)
mkdir -p "$OUT"
run_one() {                                   # run_one <name> <lichen flags...>
  local name="$1"; shift
  echo "=============== $name : $* ==============="
  (cd "$LICHEN" && exec .venv/bin/python -m lichen.server --model qwen3.8-27b-fp8 \
     --vllm-endpoint http://127.0.0.1:8010 --host 127.0.0.1 --port $PORT "$@") > "$OUT/$name.server.log" 2>&1 &
  local pid=$!
  for i in $(seq 1 60); do
    curl -s -m 3 -o /dev/null -X POST "http://127.0.0.1:$PORT/v1/systemone" -H 'Content-Type: application/json' -d '{}' && break
    sleep 1
  done
  for tier in easy original hard; do
    (cd "$JB" && python3 -m jevbench.cli run --tasks "$JB/datasets/public/$tier.jsonl" \
      --adapter typesafe --endpoint "http://127.0.0.1:$PORT" --key-env "" \
      --price-in-per-m 0 --price-out-per-m 0 --run-label "qwen38-mask-$name" \
      --results "$OUT/$name.$tier.jsonl" --ledger "$OUT/$name.ledger.jsonl" \
      --raw-dir "$OUT/raw-$name") 2>&1 | tail -2
  done
  kill $pid; wait $pid 2>/dev/null || true
}
run_one plain
run_one image "${IMAGE[@]}"
run_one image-rerun "${IMAGE[@]}"
echo "=== done; results in $OUT ==="
