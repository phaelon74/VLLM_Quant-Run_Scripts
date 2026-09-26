#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Qwen3.8-27B (dense, hybrid GDN/attention) INT8 W8A16
# 8x RTX 3060 12GB, PCIe Gen4 x4, vLLM (MRv2)
# Checkpoint: lued/Qwen3.8-27B-INT8-W8A16-MTP
#
# Bring-up order: STAGE=base -> STAGE=mtp -> soak.
#   base : no speculation. Proves load, TP8 comms, kernels, output.
#   mtp  : adds MTP (3 drafts). The GDN+MTP crash classes live here.
# ============================================================
STAGE="${STAGE:-base}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_NO_USAGE_STATS=1
export SAFETENSORS_FAST_GPU=1
export OMP_NUM_THREADS=4

# --- NCCL / P2P ---
# Leave unset only if p2pBandwidthLatencyTest shows real peer access.
#export NCCL_P2P_DISABLE=1
#export NCCL_CUMEM_ENABLE=0

# FlashInfer all-reduce: A/B showed no difference on this rig; 8.6 has no SymmMem and
# custom allreduce is off for >2 PCIe GPUs, so 0 = PyNCCL (verified in server log).
export VLLM_ALLREDUCE_USE_FLASHINFER="${VLLM_ALLREDUCE_USE_FLASHINFER:-0}"

# Last-resort masking for vllm#53726-class IMAs (large decode cost):
#export CUDA_LAUNCH_BLOCKING=1

: "${VLLM_API_KEY:?export VLLM_API_KEY first (e.g. openssl rand -hex 32)}"

MODEL_DIR="${MODEL_DIR:-/media/fmodels/lued/Qwen3.8-27B-INT8-W8A16-MTP/}"
SERVED_MODEL_NAME="qwen38-27b-int8"

# PP must stay 1 with MTP until vllm#55506 lands (stale-row state copy).
TP_SIZE=8
DP_SIZE=1

MAX_MODEL_LEN=262144
GPU_MEMORY_UTILIZATION=0.92
MAX_NUM_SEQS=2
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"   # lower = shorter decode stalls for the other stream during a prefill

# Async scheduling is ON when the flag is omitted. 1 = force off.
DISABLE_ASYNC_SCHEDULING="${DISABLE_ASYNC_SCHEDULING:-0}"

# Prefix caching: validated 2026-09-16 with prefix_cache_test.py (89 EXACT,
# 7 near-tie, 0 FAIL; block 400, MTP on) at vllm bfd713bf.
# #48375 / #55766 are still open upstream: re-record the OFF reference and
# re-run the test after any vLLM update or launch-flag change.
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"

VLLM_ARGS=(
  "$MODEL_DIR"
  --served-model-name "$SERVED_MODEL_NAME"
  --tensor-parallel-size "$TP_SIZE"
  --pipeline-parallel-size 1
  --data-parallel-size "$DP_SIZE"
  --dtype bfloat16
  --trust-remote-code
  --language-model-only
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --enable-chunked-prefill
  --mamba-cache-mode align
  --reasoning-parser qwen3
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  --default-chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true}'
  --override-generation-config '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"repetition_penalty":1.0,"presence_penalty":0.0}'
  --api-key "$VLLM_API_KEY"
  --host 0.0.0.0
  --port 8080
)

if [[ "$STAGE" == "mtp" ]]; then
  VLLM_ARGS+=(--speculative-config '{"method":"mtp","num_speculative_tokens":3}')
fi

if [[ "$DISABLE_ASYNC_SCHEDULING" == "1" ]]; then
  VLLM_ARGS+=(--no-async-scheduling)
fi

if [[ "$ENABLE_PREFIX_CACHING" == "1" ]]; then
  VLLM_ARGS+=(--enable-prefix-caching)
else
  VLLM_ARGS+=(--no-enable-prefix-caching)
fi

# Extra flags for A/B tests without editing this file, e.g.
#   EXTRA_VLLM_ARGS="--attention-backend FLASHINFER"
if [[ -n "${EXTRA_VLLM_ARGS:-}" ]]; then
  read -r -a _extra <<<"$EXTRA_VLLM_ARGS"
  VLLM_ARGS+=("${_extra[@]}")
fi

echo "STAGE=$STAGE  batched_tokens=$MAX_NUM_BATCHED_TOKENS  prefix_caching=$ENABLE_PREFIX_CACHING  async_off=$DISABLE_ASYNC_SCHEDULING  flashinfer_ar=$VLLM_ALLREDUCE_USE_FLASHINFER  extra=${EXTRA_VLLM_ARGS:-none}"
exec vllm serve "${VLLM_ARGS[@]}"
