#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# vLLM serve benchmark sweep: 1K -> 32K context
# ============================================================
#
# Runs vllm bench serve twice per context length (warmup + measured),
# discards the warmup run, and prints a summary table with:
#   - Mean TTFT (ms)
#   - True prefill throughput (input_len / mean TTFT). The steady-state
#     "Input token throughput" averages input tokens over the whole window
#     including decode time, which massively understates prefill speed at
#     short contexts; input_len / TTFT measures the prefill itself.
#   - True decode throughput (1000 / mean TPOT). The steady-state "Output
#     token throughput" divides output tokens by the whole window including
#     prefill time, which understates decode badly at long contexts; TPOT
#     measures pure inter-token latency, excluding the first token.
#
# The warmup and measured runs use different --seed values so the measured
# TTFT is a cold prefill, not a prefix-cache hit on the warmup's prompts
# (with --enable-prefix-caching on the server, identical prompts would make
# TTFT (and thus prefill tok/s) look far better than real cold-start).
#
# Usage:
#   ./bench_32k_sweep.sh API_KEY MODEL_NAME TOKENIZER [BASE_URL]
#
# Example:
#   ./bench_32k_sweep.sh "$OPENAI_API_KEY" \
#     Qwen3.6-35B-A3B-FP6-W6A6 \
#     /media/fmodels/TheHouseOfTheDude/qwen3-6_35B-A3B_moe_fp6 \
#     http://localhost:8001
#
# Optional env overrides (defaults match the standard 32K sweep):
#   NUM_PROMPTS=8
#   OUTPUT_LEN=256
#   MAX_CONCURRENCY=1
#   CONTEXT_LENGTHS="1024 4096 8192 16384 32768"
# ============================================================

usage() {
  echo "Usage: $0 API_KEY MODEL_NAME TOKENIZER [BASE_URL]" >&2
  echo "       BASE_URL defaults to http://localhost:8001" >&2
  exit 2
}

if [[ $# -lt 3 || $# -gt 4 ]]; then
  usage
fi

API_KEY="$1"
MODEL_NAME="$2"
TOKENIZER="$3"
BASE_URL="${4:-http://localhost:8001}"

export OPENAI_API_KEY="$API_KEY"

NUM_PROMPTS="${NUM_PROMPTS:-8}"
OUTPUT_LEN="${OUTPUT_LEN:-256}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
CONTEXT_LENGTHS="${CONTEXT_LENGTHS:-1024 4096 8192 16384 32768}"

# shellcheck disable=SC2206
LENGTHS=( $CONTEXT_LENGTHS )

# --- Preflight: make sure a vLLM server is listening and the key works ---
# Reports the actual HTTP status so connection problems, bad ports, and auth
# failures are distinguishable instead of all looking like "no server".
HTTP_STATUS="$(curl -s -o /dev/null -m 5 -w '%{http_code}' \
  -H "Authorization: Bearer ${API_KEY}" \
  "${BASE_URL}/v1/models" || true)"

case "$HTTP_STATUS" in
  200)
    echo "Preflight OK: server at ${BASE_URL} accepted the API key."
    ;;
  000)
    echo "ERROR: could not connect to ${BASE_URL} (no response at all)." >&2
    echo "Check that the server is running and the host/port are correct." >&2
    exit 1
    ;;
  401|403)
    echo "ERROR: server at ${BASE_URL} is up but REJECTED the API key (HTTP ${HTTP_STATUS})." >&2
    echo "Either the key is wrong, or the server was launched with a corrupted" >&2
    echo "key (e.g. a launch script with CRLF line endings appends an invisible" >&2
    echo "\\r to --api-key; run: sed -i 's/\\r\$//' <launch-script> and relaunch)." >&2
    exit 1
    ;;
  *)
    echo "WARNING: unexpected HTTP ${HTTP_STATUS} from ${BASE_URL}/v1/models; continuing anyway." >&2
    ;;
esac

parse_bench_output() {
  # Prefer the Steady-State Metrics block (last scores vLLM prints). Fall back
  # to the main Serving Benchmark Result section when steady-state is absent.
  awk '
    function trim(s) {
      sub(/^[ \t]+/, "", s)
      sub(/[ \t]+$/, "", s)
      return s
    }
    function value_after_colon(line,    parts) {
      split(line, parts, ":")
      return trim(parts[2])
    }
    /^============ Steady-State Metrics =============/ {
      in_steady = 1
      in_main = 0
      next
    }
    /^================= Serving Benchmark Result =================/ {
      in_main = 1
      in_steady = 0
      next
    }
    /^={10,}/ && !/^============ Steady-State Metrics =============/ {
      if (in_steady) in_steady = 0
      if (in_main) in_main = 0
      next
    }
    /Mean TTFT \(ms\):/ {
      if (in_steady) ss_ttft = value_after_colon($0)
      else if (in_main) main_ttft = value_after_colon($0)
      next
    }
    /Input token throughput \(tok\/s\):/ {
      if (in_steady) ss_prefill = value_after_colon($0)
      else if (in_main) main_prefill = value_after_colon($0)
      next
    }
    /Mean TPOT \(ms\):/ {
      if (in_steady) ss_tpot = value_after_colon($0)
      else if (in_main) main_tpot = value_after_colon($0)
      next
    }
    END {
      ttft = (ss_ttft != "" ? ss_ttft : main_ttft)
      prefill = (ss_prefill != "" ? ss_prefill : main_prefill)
      tpot = (ss_tpot != "" ? ss_tpot : main_tpot)
      if (ttft == "" || tpot == "") {
        print "PARSE_ERROR"
        exit 1
      }
      if (prefill == "") prefill = "n/a"
      # True decode rate: pure inter-token speed, excluding the first token.
      # Output-token throughput over the whole window is diluted by prefill
      # time (badly so at long contexts with cold prefills).
      decode = (tpot + 0 > 0) ? sprintf("%.2f", 1000 / tpot) : "n/a"
      printf "%s\t%s\t%s\n", ttft, prefill, decode
    }
  '
}

run_bench() {
  local len="$1"
  local seed="$2"
  vllm bench serve \
    --base-url "$BASE_URL" \
    --model "$MODEL_NAME" \
    --tokenizer "$TOKENIZER" \
    --dataset-name random \
    --random-input-len "$len" \
    --random-output-len "$OUTPUT_LEN" \
    --num-prompts "$NUM_PROMPTS" \
    --max-concurrency "$MAX_CONCURRENCY" \
    --temperature 0 \
    --seed "$seed"
}

declare -a RESULT_CTX RESULT_TTFT RESULT_PREFILL RESULT_DECODE

echo "Benchmark target: ${MODEL_NAME}"
echo "Tokenizer:        ${TOKENIZER}"
echo "Base URL:         ${BASE_URL}"
echo "Prompts/run:      ${NUM_PROMPTS}  output_len=${OUTPUT_LEN}  concurrency=${MAX_CONCURRENCY}"
echo

for LEN in "${LENGTHS[@]}"; do
  # Distinct seeds -> distinct prompts, so the measured run cannot hit the
  # prefix cache entries created by the warmup run.
  echo "=== ctx ${LEN} (warmup, discarded) ==="
  run_bench "$LEN" "$((LEN + 1))" >/dev/null

  echo "=== ctx ${LEN} (measured) ==="
  bench_output="$(run_bench "$LEN" "$((LEN + 2))" 2>&1 | tee /dev/stderr)"
  parsed="$(printf '%s\n' "$bench_output" | parse_bench_output)" || {
    echo "ERROR: failed to parse vllm bench output for ctx ${LEN}" >&2
    exit 1
  }

  IFS=$'\t' read -r ttft _ decode <<<"$parsed"
  # True prefill rate: tokens the prefill pass ingested per second of TTFT.
  prefill="$(awk -v len="$LEN" -v ttft_ms="$ttft" \
    'BEGIN { if (ttft_ms + 0 > 0) printf "%.2f", len / (ttft_ms / 1000); else print "n/a" }')"
  RESULT_CTX+=("$LEN")
  RESULT_TTFT+=("$ttft")
  RESULT_PREFILL+=("$prefill")
  RESULT_DECODE+=("$decode")
  echo
done

printf '\n'
printf '%-10s %12s %22s %18s\n' "Context" "TTFT (ms)" "True Prefill (tok/s)" "Decode (tok/s)"
printf '%-10s %12s %22s %18s\n' "-------" "---------" "--------------------" "--------------"
for i in "${!RESULT_CTX[@]}"; do
  printf '%-10s %12s %22s %18s\n' \
    "${RESULT_CTX[$i]}" \
    "${RESULT_TTFT[$i]}" \
    "${RESULT_PREFILL[$i]}" \
    "${RESULT_DECODE[$i]}"
done
