#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# CohereLabs/command-a-plus-05-2026-fp8 on 4x RTX 6000 Pro
# Workstation Blackwell (96 GiB / GPU, 384 GiB total)
# Single-node vLLM serve (text + vision)
# ============================================================
#
# Model: CohereLabs/command-a-plus-05-2026-fp8
#   - 218B total / ~25B active sparse MoE (cohere2_moe)
#   - 128 routed experts, 8 active/token + 4 shared experts
#   - Hybrid attention: 3 sliding-window (4K) : 1 full-attention
#   - Cohere2Vision VLM (SigLIP tower + language MoE)
#   - Native FP8 checkpoint (F8_E4M3 weights, not compressed-tensors)
#   - max_position_embeddings: 200,000 (text_config in config.json)
#
# References:
#   - https://huggingface.co/CohereLabs/command-a-plus-05-2026-fp8
#   - Cohere_Labs/Command-A-Plus/config.json
#
# Runtime requirements (from the model card):
#   pip install "vllm>=0.21.0"
#   pip install "cohere_melody>=0.9.0"   # accurate reasoning/tool parsing
#   transformers >= 5.8 with cohere2_vision / cohere2_moe support
#
# GPU topology:
#   - 4x RTX 6000 Pro Blackwell, PCIe only (no NVLink)
#   - Official FP8 recipe targets 2x B200; 4x 96 GiB is comfortable here
#
# This script does not store API keys. Pass the key as the first argument,
# or set VLLM_API_KEY in the environment.
# ============================================================

# --- Memory Management ---
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- GPU Selection (all four cards for TP + EP) ---
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3

# --- NCCL (PCIe workstation, no InfiniBand) ---
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=PHB
# If startup hangs in NCCL, try:
# export NCCL_P2P_DISABLE=1

# --- vLLM Behavior ---
export VLLM_NO_USAGE_STATS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export SAFETENSORS_FAST_GPU=1
export OMP_NUM_THREADS=8

# ============================================================
# API key
# ============================================================

API_KEY="${1:-${VLLM_API_KEY:-}}"

if [[ -z "$API_KEY" ]]; then
  echo "Usage: $0 API-KEY-HERE" >&2
  echo "       or: VLLM_API_KEY=API-KEY-HERE $0" >&2
  exit 2
fi

# ============================================================
# Model / serving configuration
# ============================================================

MODEL_DIR="${MODEL_DIR:-/media/fmodels/CohereLabs/command-a-plus-05-2026-fp8/}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-command-a-plus-fp8}"
TOKENIZER="${TOKENIZER:-$MODEL_DIR}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"

# Official HF vLLM example uses -tp 4 on B200. With EP enabled, vLLM
# computes expert-parallel size as TP x DP (4 x 1 = 4).
TP_SIZE="${TP_SIZE:-4}"
DP_SIZE="${DP_SIZE:-1}"
ALL2ALL_BACKEND="${ALL2ALL_BACKEND:-allgather_reducescatter}"

# 200K matches text_config.max_position_embeddings in config.json.
# The public model card lists 128K input; this script follows your
# local config. Lower at launch if KV profiling fails, e.g.:
#   MAX_MODEL_LEN=131072 ./CommandA-Plus_218B-A25B.sh KEY
MAX_MODEL_LEN="${MAX_MODEL_LEN:-200000}"

GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.94}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"

# fp8 KV helps at 200K on the 8 full-attention layers; sliding layers
# stay capped at sliding_window=4096 regardless of MAX_MODEL_LEN.
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"

TOKENIZER_MODE="${TOKENIZER_MODE:-hf}"
CONFIG_FORMAT="${CONFIG_FORMAT:-auto}"
LOAD_FORMAT="${LOAD_FORMAT:-auto}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"

# Cohere Command A+ parsers (requires cohere_melody for full fidelity).
REASONING_PARSER="${REASONING_PARSER:-cohere_command4}"
ENABLE_TOOL_CALLING="${ENABLE_TOOL_CALLING:-1}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-cohere_command4}"

# Multimodal: set TEXT_ONLY=1 to skip vision profiling and free VRAM
# for longer context or more concurrent sequences.
TEXT_ONLY="${TEXT_ONLY:-0}"
LIMIT_MM_IMAGE="${LIMIT_MM_IMAGE:-4}"
LIMIT_MM_AUDIO="${LIMIT_MM_AUDIO:-0}"

ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-1}"

# CUDA graphs: conservative capture on heterogeneous SWA/full layers.
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-full_decode_only}"
CUDAGRAPH_CAPTURE_SIZES="${CUDAGRAPH_CAPTURE_SIZES:-[1,2,4]}"
if [[ -z "${COMPILATION_CONFIG:-}" ]]; then
  COMPILATION_CONFIG="{\"mode\":3,\"cudagraph_mode\":\"${CUDAGRAPH_MODE}\",\"cudagraph_capture_sizes\":${CUDAGRAPH_CAPTURE_SIZES}}"
fi
MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"

# ============================================================
# Assemble vLLM arguments
# ============================================================

VLLM_ARGS=(
  "$MODEL_DIR"
  --served-model-name "$SERVED_MODEL_NAME"
  --api-key "$API_KEY"
  --host "$HOST"
  --port "$PORT"
  --tensor-parallel-size "$TP_SIZE"
  --data-parallel-size "$DP_SIZE"
  --dtype auto
  --trust-remote-code
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --kv-cache-dtype "$KV_CACHE_DTYPE"
  --tokenizer "$TOKENIZER"
  --tokenizer-mode "$TOKENIZER_MODE"
  --config-format "$CONFIG_FORMAT"
  --load-format "$LOAD_FORMAT"
  --disable-custom-all-reduce
)

if [[ "$ENFORCE_EAGER" == "1" ]]; then
  VLLM_ARGS+=(--enforce-eager)
else
  VLLM_ARGS+=(--compilation-config "$COMPILATION_CONFIG")
fi

if [[ "$ENABLE_EXPERT_PARALLEL" == "1" ]]; then
  VLLM_ARGS+=(--enable-expert-parallel --all2all-backend "$ALL2ALL_BACKEND")
fi

if [[ -n "$REASONING_PARSER" ]]; then
  VLLM_ARGS+=(--reasoning-parser "$REASONING_PARSER")
fi

if [[ "$ENABLE_TOOL_CALLING" == "1" ]]; then
  VLLM_ARGS+=(--enable-auto-tool-choice --tool-call-parser "$TOOL_CALL_PARSER")
fi

if [[ "$TEXT_ONLY" == "1" ]]; then
  VLLM_ARGS+=(--language-model-only)
  VLLM_ARGS+=(--limit-mm-per-prompt '{"image": 0, "audio": 0}')
else
  VLLM_ARGS+=(--limit-mm-per-prompt "{\"image\": ${LIMIT_MM_IMAGE}, \"audio\": ${LIMIT_MM_AUDIO}}")
fi

if [[ "$ENABLE_PREFIX_CACHING" == "1" ]]; then
  VLLM_ARGS+=(--enable-prefix-caching)
fi

if [[ -n "$MAX_CUDAGRAPH_CAPTURE_SIZE" ]]; then
  VLLM_ARGS+=(--max-cudagraph-capture-size "$MAX_CUDAGRAPH_CAPTURE_SIZE")
fi

echo "Launching vLLM Command A+ FP8:"
echo "  MODEL_DIR=${MODEL_DIR}"
echo "  SERVED_MODEL_NAME=${SERVED_MODEL_NAME}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  TP_SIZE=${TP_SIZE}  DP_SIZE=${DP_SIZE}  EP=$((TP_SIZE * DP_SIZE))"
echo "  MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "  KV_CACHE_DTYPE=${KV_CACHE_DTYPE}"
echo "  GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"
echo "  MAX_NUM_SEQS=${MAX_NUM_SEQS}"
echo "  REASONING_PARSER=${REASONING_PARSER}"
echo "  TOOL_CALLING=${ENABLE_TOOL_CALLING} (parser=${TOOL_CALL_PARSER})"
if [[ "$TEXT_ONLY" == "1" ]]; then
  echo "  TEXT_ONLY=1 (language model only, vision disabled)"
else
  echo "  MULTIMODAL image=${LIMIT_MM_IMAGE} audio=${LIMIT_MM_AUDIO}"
fi
echo "  PREFIX_CACHING=${ENABLE_PREFIX_CACHING}"
echo "  ENFORCE_EAGER=${ENFORCE_EAGER}"
echo

vllm serve "${VLLM_ARGS[@]}"
