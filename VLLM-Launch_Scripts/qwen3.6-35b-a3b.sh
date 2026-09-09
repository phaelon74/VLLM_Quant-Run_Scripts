
#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# Qwen3.6-35B-A3B INT8 on 8x RTX 3060 Ti
# Single-node text-serving defaults for vLLM
# ============================================================

# --- Memory Management ---
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- GPU Selection ---
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# --- NCCL (consumer PCIe GPUs, no InfiniBand) ---
#export NCCL_IB_DISABLE=1
# Set P2P level based on your PCIe topology:
#   NVL  = same NVLink clique
#   PIX  = same PCIe switch
#   PHB  = same PCIe host bridge / CPU socket
#   SYS  = cross-socket via QPI/UPI
# For most consumer boards with 8 GPUs, PHB or SYS is typical.
#export NCCL_P2P_LEVEL=PHB
#export NCCL_MIN_NCHANNELS=8

# --- vLLM Behavior ---
export VLLM_NO_USAGE_STATS=1

# --- Loading Acceleration ---
export SAFETENSORS_FAST_GPU=1

# --- CPU Threads (adjust to your system: total cores / 8) ---
export OMP_NUM_THREADS=4

# ============================================================
# Qwen3.6-35B-A3B notes:
# - 35B total params, ~3B activated per token
# - 256 experts, with 8 routed + 1 shared expert active
# - Native context is 262,144 tokens
# - This script defaults to text-only serving to save VRAM
#
# vLLM EP notes:
# - Expert parallel size is computed automatically as TP x DP
# - On this 8-GPU single-node setup, TP=8 and DP=1 gives EP=8
# - allgather_reducescatter is the safest single-node backend
#
# Not enabled by default on 8 GB cards:
# - EPLB can improve MoE balance, but redundant experts cost VRAM
# - DeepEP / FlashInfer all2all backends are more relevant to
#   newer NVLink or multi-node deployments
# ============================================================

MODEL_DIR="/media/fmodels/TheHouseOfTheDude/Qwen3.6-35B-A3B_INT8/"
SERVED_MODEL_NAME="qwen36-35b-a3b-int8"

TP_SIZE=4
DP_SIZE=1
ALL2ALL_BACKEND="allgather_reducescatter"

# Qwen recommends 262K natively, but 128K is a safer default on
# 8x 3060 Ti while still preserving long-context behavior better
# than the old 64K script.
MAX_MODEL_LEN=262144
GPU_MEMORY_UTILIZATION=0.94
MAX_NUM_SEQS=8
MAX_NUM_BATCHED_TOKENS=8192

ENABLE_PREFIX_CACHING=1
ENABLE_MTP=1
ENABLE_EPLB=0

# Official Qwen3.6 model card uses qwen3_next_mtp with 2 tokens.
MTP_SPEC='{"method":"qwen3_next_mtp","num_speculative_tokens":2}'

# If startup fails with a Mamba/Deltanet cuda-graph cache assert,
# set this to 4.
MAX_CUDAGRAPH_CAPTURE_SIZE=""

# EPLB can help skewed MoE routing, but it increases VRAM use.
EPLB_CONFIG='{"window_size":1000,"step_interval":3000,"log_balancedness":true}'

VLLM_ARGS=(
  "$MODEL_DIR"
  --served-model-name "$SERVED_MODEL_NAME"
  --tensor-parallel-size "$TP_SIZE"
  --data-parallel-size "$DP_SIZE"
  --enable-expert-parallel
  --all2all-backend "$ALL2ALL_BACKEND"
  --dtype auto
  --trust-remote-code
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --reasoning-parser qwen3
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  --language-model-only
  --api-key REDACTED
  --host 0.0.0.0
  --port 8080
)

if [[ "$ENABLE_PREFIX_CACHING" == "1" ]]; then
  VLLM_ARGS+=(--enable-prefix-caching)
fi

if [[ "$ENABLE_MTP" == "1" ]]; then
  VLLM_ARGS+=(--speculative-config "$MTP_SPEC")
fi

if [[ "$ENABLE_EPLB" == "1" ]]; then
  VLLM_ARGS+=(--enable-eplb --eplb-config "$EPLB_CONFIG")
fi

if [[ -n "$MAX_CUDAGRAPH_CAPTURE_SIZE" ]]; then
  VLLM_ARGS+=(--max-cudagraph-capture-size "$MAX_CUDAGRAPH_CAPTURE_SIZE")
fi

vllm serve "${VLLM_ARGS[@]}"
