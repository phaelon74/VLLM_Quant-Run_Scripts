#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# Qwen/Qwen3.6-27B-FP8 on 1x RTX PRO 6000 (Blackwell) via vLLM
# ============================================================
#
# Purpose: baseline comparison against the B12X MX-FP6 (W6A6) build served by
# qwen3.6-27b-fp6.sh. Same model family, same serving knobs (MTP spec decode,
# 128K ctx, full_decode_only cudagraphs), so the tok/s difference isolates the
# quantization path: Qwen's official FP8 (vLLM stock kernels) vs B12X FP6.
#
# Model:
#   - Qwen/Qwen3.6-27B-FP8 (official Qwen release).
#   - Quant: fine-grained FP8, weight block size 128 (config.json
#     quant_method "fp8"). vLLM AUTO-DETECTS this -- do NOT pass
#     --quantization; the stock vLLM FP8 block-quant path serves it.
#
# VRAM fit on 1x 96 GiB Blackwell:
#   - FP8 weights ~28 GiB + BF16 residue; KV cache (16 Gated Attention
#     layers, BF16): ~8 GiB per 128K sequence. Fits with margin at
#     MAX_NUM_SEQS=4.
#
# This script does not store API keys. Pass the key as the first argument, or
# set VLLM_API_KEY in the environment.
# ============================================================

# --- Memory Management ---
export PYTORCH_ALLOC_CONF=expandable_segments:True

# --- GPU Selection (single Blackwell card) ---
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# --- B12X gates: OFF. This is the stock-vLLM FP8 baseline; make sure the
# B12X plugin cannot claim the checkpoint. ---
export B12X_ENABLE_FP6=0

# --- vLLM / CUDA behavior ---
export VLLM_NO_USAGE_STATS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_SLEEP_WHEN_IDLE=1
export VLLM_USE_FLASHINFER_SAMPLER=0

# --- Loading / CPU threading ---
export SAFETENSORS_FAST_GPU=1
export OMP_NUM_THREADS=8

# --- Torch profiler (registers POST /start_profile and /stop_profile) ---
# Recent vLLM nightlies gate these endpoints on the --profiler-config CLI arg;
# PROFILE=1 enables. Traces (.json.gz) land in PROFILE_DIR; summarize with
# b12x scripts/summarize_vllm_trace.py.
PROFILE="${PROFILE:-0}"
PROFILE_DIR="${PROFILE_DIR:-/tmp/vllm_prof}"
if [[ "$PROFILE" == "1" ]]; then
  mkdir -p "$PROFILE_DIR"
  export VLLM_TORCH_PROFILER_DIR="$PROFILE_DIR"
fi

# ============================================================
# Model / serving configuration
# ============================================================

API_KEY="${1:-${VLLM_API_KEY:-}}"

if [[ -z "$API_KEY" ]]; then
  echo "Usage: $0 API-KEY-HERE" >&2
  echo "       or: VLLM_API_KEY=API-KEY-HERE $0" >&2
  exit 2
fi

# HF model id by default; point MODEL_DIR at a local download to skip the hub.
MODEL_DIR="${MODEL_DIR:-Qwen/Qwen3.6-27B-FP8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.6-27B-FP8}"
TOKENIZER="${TOKENIZER:-$MODEL_DIR}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"

# Single GPU -> no tensor parallelism.
TP_SIZE="${TP_SIZE:-1}"

# Match the FP6 launch sizing so the comparison is apples-to-apples.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"

KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"

TOKENIZER_MODE="${TOKENIZER_MODE:-hf}"
CONFIG_FORMAT="${CONFIG_FORMAT:-auto}"
LOAD_FORMAT="${LOAD_FORMAT:-auto}"

# IMPORTANT: leave empty. vLLM auto-detects the checkpoint's fp8 block-quant
# config; forcing a method here can select the wrong kernel path.
QUANTIZATION="${QUANTIZATION:-}"

# Qwen3.6 ships a custom modeling file (qwen3_5 / qwen3_next family).
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"

# ============================================================
# Qwen3.6 reasoning / tool-call / MTP / VLM toggles (same as FP6 script)
# ============================================================

REASONING_PARSER="${REASONING_PARSER:-qwen3}"

ENABLE_TOOL_CALLING="${ENABLE_TOOL_CALLING:-1}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"

# Multi-Token Prediction: same spec config as the FP6 launch.
ENABLE_MTP="${ENABLE_MTP:-1}"
if [[ -z "${MTP_SPEC:-}" ]]; then
  MTP_SPEC='{"method":"qwen3_next_mtp","num_speculative_tokens":2}'
fi

TEXT_ONLY="${TEXT_ONLY:-0}"
if [[ -z "${MEDIA_IO_KWARGS:-}" ]]; then
  MEDIA_IO_KWARGS='{"video":{"num_frames":-1}}'
fi

ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"

# CUDA graph setup: identical to the FP6 launch for a fair comparison.
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-full_decode_only}"
CUDAGRAPH_CAPTURE_SIZES="${CUDAGRAPH_CAPTURE_SIZES:-[1,2,4]}"
if [[ -z "${COMPILATION_CONFIG:-}" ]]; then
  COMPILATION_CONFIG="{\"mode\":3,\"cudagraph_mode\":\"${CUDAGRAPH_MODE}\",\"cudagraph_capture_sizes\":${CUDAGRAPH_CAPTURE_SIZES}}"
fi

MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-}"

# Stock FP8 path has no JIT-in-forward issue; eager only for debugging.
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"

# ============================================================
# Assemble the argument list
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

if [[ "$PROFILE" == "1" ]]; then
  VLLM_ARGS+=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${PROFILE_DIR}\"}")
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

if [[ "$ENABLE_MTP" == "1" ]]; then
  VLLM_ARGS+=(--speculative-config "$MTP_SPEC")
fi

if [[ "$TEXT_ONLY" == "1" ]]; then
  VLLM_ARGS+=(--language-model-only)
elif [[ -n "$MEDIA_IO_KWARGS" ]]; then
  VLLM_ARGS+=(--media-io-kwargs "$MEDIA_IO_KWARGS")
fi

if [[ "$ENABLE_PREFIX_CACHING" == "1" ]]; then
  VLLM_ARGS+=(--enable-prefix-caching)
fi

if [[ -n "$MAX_CUDAGRAPH_CAPTURE_SIZE" ]]; then
  VLLM_ARGS+=(--max-cudagraph-capture-size "$MAX_CUDAGRAPH_CAPTURE_SIZE")
fi

echo "Launching vLLM (FP8 baseline) with:"
echo "  MODEL_DIR=${MODEL_DIR}"
echo "  SERVED_MODEL_NAME=${SERVED_MODEL_NAME}"
echo "  TOKENIZER=${TOKENIZER}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  B12X_ENABLE_FP6=0 (stock vLLM FP8 path)"
echo "  TP_SIZE=${TP_SIZE}"
echo "  MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "  KV_CACHE_DTYPE=${KV_CACHE_DTYPE}"
echo "  QUANTIZATION=${QUANTIZATION:-auto-detect (fp8 block-quant)}"
echo "  REASONING_PARSER=${REASONING_PARSER}"
echo "  TOOL_CALLING=${ENABLE_TOOL_CALLING} (parser=${TOOL_CALL_PARSER})"
echo "  MTP=${ENABLE_MTP} (${MTP_SPEC})"
echo "  TEXT_ONLY=${TEXT_ONLY}"
echo "  PREFIX_CACHING=${ENABLE_PREFIX_CACHING}"
if [[ "$PROFILE" == "1" ]]; then
  echo "  PROFILER=torch -> ${PROFILE_DIR} (POST /start_profile + /stop_profile)"
else
  echo "  PROFILER=disabled (PROFILE=1 to enable)"
fi
if [[ "$ENFORCE_EAGER" == "1" ]]; then
  echo "  ENFORCE_EAGER=1 (torch.compile + CUDA graphs DISABLED)"
else
  echo "  COMPILATION_CONFIG=${COMPILATION_CONFIG}"
fi
echo

vllm serve "${VLLM_ARGS[@]}"
