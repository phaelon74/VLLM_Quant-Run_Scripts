#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# Qwen3-VL-30B-A3B-Instruct-Heretic INT8-PTQ (W8A16) on 1x GPU via vLLM
# ============================================================
#
# Model:
#   - Path:  /media/fmodels/TheHouseOfTheDude/Qwen3-VL-30B-A3B-Instruct-Heretic_INT8-PTQ
#   - Base:  catplusplus/Qwen3-VL-30B-A3B-Instruct-Heretic
#            (finetune of Qwen/Qwen3-VL-30B-A3B-Instruct)
#   - Arch:  qwen3_vl_moe / Qwen3VLMoeForConditionalGeneration
#            48 layers, 128 experts, 8 active / token, ~31B params
#   - Quant: compressed-tensors PTQ W8A16 (INT8 weights, BF16 acts)
#            IGNORED (kept BF16): lm_head, model.visual.*, mlp.gate (router)
#            Experts ARE quantized. Vision tower preserved.
#
# IMPORTANT — Marlin MoE + channel-wise W8A16:
#   PTQ scheme W8A16 writes group_size=null (channel). Some vLLM nightlies
#   crash in marlin_moe_padded_intermediate with:
#     TypeError: '>' not supported between instances of 'NoneType' and 'int'
#   Fix once on the checkpoint (rewrites null -> -1, backs up *.bak):
#     python ../Qwen3_VL/patch_w8a16_moe_group_size.py "$MODEL_DIR"
#
# GPU selection (4-GPU host, indices 0,1,2,3):
#   - This script targets physical nvidia-smi GPU 2 (CUDA:2).
#   - CUDA_VISIBLE_DEVICES=2 exposes only that card; inside vLLM it is
#     logical cuda:0 with TP_SIZE=1.
#   - Override if needed: CUDA_VISIBLE_DEVICES=3 ./this-script.sh KEY
#
# Context:
#   - Default MAX_MODEL_LEN=65536 (65K) as requested.
#   - Native model context is 262144; raise only if VRAM allows.
#
# This script does not store API keys. Pass the key as arg1 or set
# VLLM_API_KEY in the environment.
# ============================================================

# --- Memory Management ---
export PYTORCH_ALLOC_CONF=expandable_segments:True

# --- GPU Selection ---
# Physical GPU 2 on a 4-GPU (0,1,2,3) system.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

# --- vLLM / CUDA behavior ---
export VLLM_NO_USAGE_STATS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_SLEEP_WHEN_IDLE=1
export VLLM_MARLIN_USE_ATOMIC_ADD=1
export VLLM_USE_FLASHINFER_SAMPLER=0

# --- Loading / CPU threading ---
export SAFETENSORS_FAST_GPU=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# ============================================================
# Model / serving configuration
# ============================================================

API_KEY="${1:-${VLLM_API_KEY:-}}"

if [[ -z "$API_KEY" ]]; then
  echo "Usage: $0 API-KEY-HERE" >&2
  echo "       or: VLLM_API_KEY=API-KEY-HERE $0" >&2
  exit 2
fi

MODEL_DIR="${MODEL_DIR:-/media/fmodels/TheHouseOfTheDude/Qwen3-VL-30B-A3B-Instruct-Heretic_INT8-PTQ}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3-VL-30B-A3B-Heretic-INT8-W8A16}"
TOKENIZER="${TOKENIZER:-$MODEL_DIR}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8003}"

# Single-GPU serve (this card only).
TP_SIZE="${TP_SIZE:-1}"

# ------------------------------------------------------------
# Capacity defaults for single-GPU MoE VL at 65K context.
# Tune GPU_MEMORY_UTILIZATION / MAX_NUM_SEQS if KV OOM or underfill.
# ------------------------------------------------------------
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.93}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"

TOKENIZER_MODE="${TOKENIZER_MODE:-hf}"
CONFIG_FORMAT="${CONFIG_FORMAT:-auto}"
LOAD_FORMAT="${LOAD_FORMAT:-auto}"
QUANTIZATION="${QUANTIZATION:-compressed-tensors}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"

# ============================================================
# Qwen3-VL toggles (vision / reasoning / tools)
# ============================================================

# Qwen3 family reasoning parser, OFF by default for this checkpoint.
#
# Heretic is an Instruct (non-thinking) finetune. With the qwen3 parser on,
# output that never emits </think> can be routed entirely into
# `reasoning_content`, leaving `content: null`. Clients that do len(content)
# then die with "object of type 'NoneType' has no len()".
#
# Turn it back on only if you actually want <think> split out:
#   REASONING_PARSER=qwen3 ./qwen3-vl-30b-a3b-heretic-int8.sh KEY
REASONING_PARSER="${REASONING_PARSER:-}"

ENABLE_TOOL_CALLING="${ENABLE_TOOL_CALLING:-1}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-hermes}"

# Vision preserved in the W8A16 quant. Set TEXT_ONLY=1 to skip multimodal
# profiling and free VRAM for KV / longer context.
#
# LIMIT_MM_IMAGE is a hard per-request cap. Exceeding it returns
#   HTTP 400 "At most N image(s) may be provided in one prompt."
# Frame-sampling clients (video analyzers, storyboard promptors) commonly
# send 8-16 frames as images, so the default is 16 here. Each additional
# allowed image raises vLLM's multimodal profiling reservation, so lower
# this if you need the VRAM for KV cache instead.
TEXT_ONLY="${TEXT_ONLY:-0}"
LIMIT_MM_IMAGE="${LIMIT_MM_IMAGE:-16}"
LIMIT_MM_VIDEO="${LIMIT_MM_VIDEO:-1}"

ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"

# CUDA graphs: decode-only is safer on MoE + multimodal single-GPU loads.
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-full_decode_only}"
CUDAGRAPH_CAPTURE_SIZES="${CUDAGRAPH_CAPTURE_SIZES:-[1,2,4]}"
if [[ -z "${COMPILATION_CONFIG:-}" ]]; then
  COMPILATION_CONFIG="{\"mode\":3,\"cudagraph_mode\":\"${CUDAGRAPH_MODE}\",\"cudagraph_capture_sizes\":${CUDAGRAPH_CAPTURE_SIZES}}"
fi
MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"

# ============================================================
# Assemble args
# ============================================================

VLLM_ARGS=(
  "$MODEL_DIR"
  --served-model-name "$SERVED_MODEL_NAME"
  --api-key "$API_KEY"
  --host "$HOST"
  --port "$PORT"
  --tensor-parallel-size "$TP_SIZE"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --kv-cache-dtype "$KV_CACHE_DTYPE"
  --tokenizer "$TOKENIZER"
  --tokenizer-mode "$TOKENIZER_MODE"
  --config-format "$CONFIG_FORMAT"
  --load-format "$LOAD_FORMAT"
  --dtype auto
)

if [[ "$ENFORCE_EAGER" == "1" ]]; then
  VLLM_ARGS+=(--enforce-eager)
else
  VLLM_ARGS+=(--compilation-config "$COMPILATION_CONFIG")
fi

if [[ -n "$QUANTIZATION" ]]; then
  VLLM_ARGS+=(--quantization "$QUANTIZATION")
fi

if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
  VLLM_ARGS+=(--trust-remote-code)
fi

if [[ -n "$REASONING_PARSER" ]]; then
  VLLM_ARGS+=(--reasoning-parser "$REASONING_PARSER")
fi

if [[ "$ENABLE_TOOL_CALLING" == "1" ]]; then
  VLLM_ARGS+=(--enable-auto-tool-choice --tool-call-parser "$TOOL_CALL_PARSER")
fi

if [[ "$TEXT_ONLY" == "1" ]]; then
  VLLM_ARGS+=(--limit-mm-per-prompt '{"image": 0, "video": 0}')
  VLLM_ARGS+=(--language-model-only)
else
  VLLM_ARGS+=(--limit-mm-per-prompt "{\"image\": ${LIMIT_MM_IMAGE}, \"video\": ${LIMIT_MM_VIDEO}}")
fi

if [[ "$ENABLE_PREFIX_CACHING" == "1" ]]; then
  VLLM_ARGS+=(--enable-prefix-caching)
fi

if [[ -n "$MAX_CUDAGRAPH_CAPTURE_SIZE" ]]; then
  VLLM_ARGS+=(--max-cudagraph-capture-size "$MAX_CUDAGRAPH_CAPTURE_SIZE")
fi

echo "Launching vLLM with:"
echo "  MODEL_DIR=${MODEL_DIR}"
echo "  SERVED_MODEL_NAME=${SERVED_MODEL_NAME}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}  (logical cuda:0 inside process)"
echo "  TP_SIZE=${TP_SIZE}"
echo "  MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "  GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"
echo "  QUANTIZATION=${QUANTIZATION}"
echo "  REASONING_PARSER=${REASONING_PARSER}"
echo "  TOOL_CALLING=${ENABLE_TOOL_CALLING} (parser=${TOOL_CALL_PARSER})"
echo "  TEXT_ONLY=${TEXT_ONLY}"
echo "  PREFIX_CACHING=${ENABLE_PREFIX_CACHING}"
echo "  HOST=${HOST} PORT=${PORT}"
echo

vllm serve "${VLLM_ARGS[@]}"
