#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# TheHouseOfTheDude/Qwen3.6-27B-INT8 on 2x RTX 3090 via vLLM
# ============================================================
#
# Model:
#   - HF id:   TheHouseOfTheDude/Qwen3.6-27B-INT8
#   - Base:    Qwen/Qwen3.6-27B  (arch tag: qwen3_5, ~28B params)
#   - Quant:   compressed-tensors PTQ, W8A16 (INT8 weights, BF16 acts)
#              Linear layers only.
#              IGNORED (kept BF16): lm_head, visual, linear_attn, mtp
#
# Architecture notes (from the upstream Qwen/Qwen3.6-27B card):
#   - Hybrid 64-layer stack:
#       16 x (3 x (Gated DeltaNet -> FFN) + 1 x (Gated Attention -> FFN))
#     i.e. 48 linear-attention (DeltaNet) layers + 16 full-attention layers.
#     KV cache is only allocated for the 16 Gated Attention layers, which
#     keeps long-context KV usage much lower than a 64-full-attn dense LM.
#   - Gated Attention: 24 Q heads, 4 KV heads, head_dim=256, partial RoPE.
#   - Gated DeltaNet: 48 V heads, 16 QK heads, head_dim=128 (no KV cache).
#   - Vision encoder is present (this is a VL model).
#   - MTP head is trained-in -> use --speculative-config qwen3_next_mtp.
#   - Native ctx: 262,144 tokens, extensible to ~1,010,000 with YaRN.
#
# IMPORTANT VRAM / FIT NOTES for 2x 24 GiB RTX 3090:
#   - INT8 linear weights for ~28B params ~= 28 GiB on disk, BUT the
#     ignored-from-quant pieces (visual ViT, all 48 DeltaNet layers,
#     lm_head, MTP head) stay BF16. Realistic resident weight footprint
#     is ~35-40 GiB total -> ~17-20 GiB per GPU after TP=2.
#   - That leaves ~3-6 GiB / GPU for KV cache, activations, CUDA graphs,
#     vision encoder workspace, and the MTP draft. Defaults below are
#     tuned for this envelope at full-VLM serving.
#   - If you only need text serving, set TEXT_ONLY=1 to free the vision
#     profiling reservation and roughly double the KV cache budget.
#
# This script intentionally does not store API keys. Pass the key as the
# first argument, or set VLLM_API_KEY in the environment.
#
# GPU selection:
#   - Physical nvidia-smi GPUs 0,5 are exposed to the process (the two
#     idle 3090s on this 6-GPU host; GPUs 1-4 are reserved for behemoth).
#   - Inside vLLM they appear as logical CUDA devices 0,1.
#   - PCI topology: 0 is at 01:00.0 and 5 is at 49:00.0 -- opposite ends
#     of the PCIe layout, so NCCL P2P will go via PHB (CPU root complex).
#     If you see NCCL hangs or sluggish all-reduce, uncomment the NCCL
#     overrides below.
# ============================================================

# --- Memory Management ---
export PYTORCH_ALLOC_CONF=expandable_segments:True

# --- GPU Selection ---
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,5

# --- vLLM / CUDA behavior ---
export VLLM_NO_USAGE_STATS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_SLEEP_WHEN_IDLE=1
export VLLM_MARLIN_USE_ATOMIC_ADD=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export disable_custom_all_reduce=True

# --- Loading / CPU threading ---
export SAFETENSORS_FAST_GPU=1
export OMP_NUM_THREADS=8

# --- NCCL notes for consumer PCIe GPUs (no NVLink between 3090s) ---
# Uncomment/tune only if you see NCCL hangs or poor peer-to-peer behavior.
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_LEVEL=PHB
# export NCCL_MIN_NCHANNELS=8

# --- Long-context (YaRN) escape hatch ---
# Set this and pass --hf-overrides '...rope_parameters...' if you ever push
# MAX_MODEL_LEN past the native 262144 limit. Not needed for the defaults.
# export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# ============================================================
# Model / serving configuration
# ============================================================

API_KEY="${1:-${VLLM_API_KEY:-}}"

if [[ -z "$API_KEY" ]]; then
  echo "Usage: $0 API-KEY-HERE" >&2
  echo "       or: VLLM_API_KEY=API-KEY-HERE $0" >&2
  exit 2
fi

# Local W8A16 compressed-tensors checkpoint.
# Override at launch time if needed:
#   MODEL_DIR=/path/to/model ./qwen3.6-27b-int8.sh
MODEL_DIR="${MODEL_DIR:-/media/fmodels/TheHouseOfTheDude/Qwen3.6-27B-INT8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.6-27B-INT8-W8A16}"
# The compressed-tensors checkpoint includes tokenizer.json/tokenizer_config.json
# and chat_template.jinja, so keep tokenizer resolution local by default.
TOKENIZER="${TOKENIZER:-$MODEL_DIR}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"

TP_SIZE="${TP_SIZE:-2}"

# ------------------------------------------------------------
# Capacity sizing for 2x 24 GiB at full VLM (vision enabled).
#
# KV cache math (Gated Attention layers only, BF16):
#   per-token KV / GPU = 16 layers * 2 (K+V) * (4 KV heads / TP=2) * 256 = 32 KiB
#   64K ctx, 2 seqs    = 64K * 32 KiB * 2 = 4 GiB / GPU
#  128K ctx, 2 seqs    = 8 GiB / GPU  <- only safe with TEXT_ONLY=1
#
# Defaults below assume vision is on; bump MAX_MODEL_LEN to 131072 (or
# higher) when TEXT_ONLY=1.
# ------------------------------------------------------------
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.93}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"

# Use auto by default. fp8 KV cache is rejected on some 3090/vLLM combos
# and the Gated Attention layers are already a small fraction of total
# memory (only 16/64 layers cache anything), so the win is marginal.
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"

TOKENIZER_MODE="${TOKENIZER_MODE:-hf}"
CONFIG_FORMAT="${CONFIG_FORMAT:-auto}"
LOAD_FORMAT="${LOAD_FORMAT:-auto}"

# This checkpoint is compressed-tensors W8A16. The CLI spelling uses a hyphen.
QUANTIZATION="${QUANTIZATION:-compressed-tensors}"

# Qwen3.6 ships a custom modeling file (qwen3_5 / qwen3_next family). vLLM
# can register it without trust_remote_code on recent builds, but turning it
# on is the safer default and matches the sibling 35B-A3B launch script.
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"

# ============================================================
# Qwen3.6 reasoning / tool-call / MTP / VLM toggles
# ============================================================

# Qwen3.6 thinks by default. The official model card prescribes the qwen3
# reasoning parser for vLLM (works for the whole Qwen3.x family, including
# 3.6). The compressed-tensors quant preserves the tokenizer including the
# <think>/</think> markers, so this is safe.
REASONING_PARSER="${REASONING_PARSER:-qwen3}"

# Tool calling: per the Qwen3.6-27B model card, qwen3_coder is the correct
# parser for OpenAI-style tool use. Defaults to ON because the model is
# trained for agentic / coding workflows.
ENABLE_TOOL_CALLING="${ENABLE_TOOL_CALLING:-1}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"

# Multi-Token Prediction (speculative decoding using the trained-in MTP head).
# Per Qwen3.6 model card: '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'.
# Disable (set ENABLE_MTP=0) if you hit a Mamba/DeltaNet cudagraph assert
# during MTP draft capture, or if you are deliberately benchmarking non-spec
# throughput.
ENABLE_MTP="${ENABLE_MTP:-1}"
if [[ -z "${MTP_SPEC:-}" ]]; then
  MTP_SPEC='{"method":"qwen3_next_mtp","num_speculative_tokens":2}'
fi

# Vision. This is a VL model; the W8A16 quant left the visual tower in BF16
# so it works out of the box. Set TEXT_ONLY=1 to skip vision profiling and
# free its workspace for additional KV cache (use this if you're only
# routing text traffic).
TEXT_ONLY="${TEXT_ONLY:-0}"

# Optional: per the Qwen3.6 model card, this lets clients drive video frame
# sampling via extra_body={"mm_processor_kwargs": {"fps": ...}}. Harmless
# when no video is sent. Comment out if it causes a parser warning.
if [[ -z "${MEDIA_IO_KWARGS:-}" ]]; then
  MEDIA_IO_KWARGS='{"video":{"num_frames":-1}}'
fi

# Prefix caching is generally a win for chat/agent workloads.
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"

# CUDA graph setup:
# - full_decode_only keeps CUDA graphs on the decode path and avoids the
#   highest-memory full-prefill captures (important on 24 GiB cards).
# - Capture sizes [1,2,4] match the behemoth profile and play nicely with
#   small MAX_NUM_SEQS.
# - Qwen3.6's Gated DeltaNet layers occasionally trip a cudagraph cache
#   assert ("Mamba/DeltaNet cuda-graph cache" path). If you see it, set
#   MAX_CUDAGRAPH_CAPTURE_SIZE=4 (or 1) below as a fallback.
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-full_decode_only}"
CUDAGRAPH_CAPTURE_SIZES="${CUDAGRAPH_CAPTURE_SIZES:-[1,2,4]}"
if [[ -z "${COMPILATION_CONFIG:-}" ]]; then
  COMPILATION_CONFIG="{\"mode\":3,\"cudagraph_mode\":\"${CUDAGRAPH_MODE}\",\"cudagraph_capture_sizes\":${CUDAGRAPH_CAPTURE_SIZES}}"
fi

# Fallback for older vLLM versions that do not understand cudagraph_capture_sizes
# inside --compilation-config, OR the DeltaNet cudagraph cache assert above.
MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-}"

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
  --compilation-config "$COMPILATION_CONFIG"
  --disable-custom-all-reduce
)

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

echo "Launching vLLM with:"
echo "  MODEL_DIR=${MODEL_DIR}"
echo "  SERVED_MODEL_NAME=${SERVED_MODEL_NAME}"
echo "  TOKENIZER=${TOKENIZER}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  TP_SIZE=${TP_SIZE}"
echo "  MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "  KV_CACHE_DTYPE=${KV_CACHE_DTYPE}"
echo "  QUANTIZATION=${QUANTIZATION}"
echo "  REASONING_PARSER=${REASONING_PARSER}"
echo "  TOOL_CALLING=${ENABLE_TOOL_CALLING} (parser=${TOOL_CALL_PARSER})"
echo "  MTP=${ENABLE_MTP} (${MTP_SPEC})"
echo "  TEXT_ONLY=${TEXT_ONLY}"
echo "  PREFIX_CACHING=${ENABLE_PREFIX_CACHING}"
echo "  COMPILATION_CONFIG=${COMPILATION_CONFIG}"
echo

vllm serve "${VLLM_ARGS[@]}"
