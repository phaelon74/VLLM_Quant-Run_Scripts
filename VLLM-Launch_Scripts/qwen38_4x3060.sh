#!/usr/bin/env bash
# ============================================================
# Qwen3.8-27B INT8 W8A8 (v2-a16) on FOUR RTX 3060 12GB
#   TheHouseOfTheDude/Qwen3.8-27B_INT8-W8A8-v2-a16
#
# Mixed-precision checkpoint:
#   W8A8 INT8  -> MLP, o_proj, linear_attn.out_proj   (CUTLASS INT8, Ampere)
#   W8A16      -> q/k/v_proj, linear_attn.in_proj_*   (Marlin / AllSpark)
#   BF16       -> MTP head, vision tower, lm_head, GDN state params
#
# Requirements on the serving box:
#   - vLLM new enough for Qwen3.5/3.8 hybrid (GDN + MTP) AND for
#     compressed-tensors format="mixed-precision" with per-group formats.
#     Verify at startup: the log must show BOTH
#       "CutlassInt8ScaledMMLinearKernel for CompressedTensorsW8A8Int8"
#       "...LinearKernel for CompressedTensorsWNA16"   (Marlin or AllSpark)
#     If only one appears, the two config groups collapsed and the build is
#     not running as intended.
#   - ~30 GB free VRAM for weights: 7.3 GB per card at TP4, leaving roughly
#     3 GB per card for KV cache.
#
# Usage:
#   export VLLM_API_KEY=$(openssl rand -hex 32)
#   ./qwen38_4x3060.sh                 # 96K context, BF16 KV  (default)
#   PROFILE=long ./qwen38_4x3060.sh    # 131K context, FP8 KV
# ============================================================
set -euo pipefail

: "${VLLM_API_KEY:?export VLLM_API_KEY first, e.g. openssl rand -hex 32}"

MODEL_DIR="${MODEL_DIR:-/models/Qwen3.8-27B_INT8-W8A8-v2-a16}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen38-27b}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_NO_USAGE_STATS=1
export SAFETENSORS_FAST_GPU=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

# FlashInfer all-reduce brought no measurable gain on consumer PCIe cards in
# our testing (SymmMem needs sm90+, custom all-reduce is off for >2 PCIe GPUs),
# so this stays on plain PyNCCL. Flip to 1 to A/B it.
export VLLM_ALLREDUCE_USE_FLASHINFER="${VLLM_ALLREDUCE_USE_FLASHINFER:-0}"

# Uncomment if NCCL hangs at init or peer-to-peer is unverified on this board.
#export NCCL_P2P_DISABLE=1
#export NCCL_CUMEM_ENABLE=0

TP_SIZE="${TP_SIZE:-4}"

# --- context / KV profile -----------------------------------------------
# 4x12 GB leaves ~13 GB total for KV after weights and overhead, and this
# hybrid model costs roughly 110 KB per token of context at TP4 (attention KV
# plus Gated DeltaNet state pages). That is about 120K tokens of headroom, so
# 131K with BF16 KV is right at the edge. Default to 96K; PROFILE=long halves
# the attention half with an FP8 KV cache to reach 131K.
PROFILE="${PROFILE:-agent}"
case "$PROFILE" in
  agent) DEF_LEN=98304;  DEF_KV=auto ;;
  long)  DEF_LEN=131072; DEF_KV=fp8  ;;
  *) echo "ERROR: PROFILE must be agent or long" >&2; exit 1 ;;
esac
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$DEF_LEN}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-$DEF_KV}"

GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
STAGE="${STAGE:-mtp}"                 # mtp | base (base = no speculation)
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
DISABLE_ASYNC_SCHEDULING="${DISABLE_ASYNC_SCHEDULING:-0}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"

[[ -d "$MODEL_DIR" ]] || { echo "ERROR: model directory not found: $MODEL_DIR" >&2; exit 1; }

VLLM_ARGS=(
  "$MODEL_DIR"
  --served-model-name "$SERVED_MODEL_NAME"
  --tensor-parallel-size "$TP_SIZE"
  --pipeline-parallel-size 1
  --dtype bfloat16
  --trust-remote-code
  --language-model-only
  --kv-cache-dtype "$KV_CACHE_DTYPE"
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
  --host "$HOST"
  --port "$PORT"
)

# MTP speculative decoding: ~3.5 accepted tokens per step on this checkpoint,
# worth roughly 2x decode. Drop to STAGE=base if it ever misbehaves.
[[ "$STAGE" == "mtp" ]] && VLLM_ARGS+=(--speculative-config '{"method":"mtp","num_speculative_tokens":3}')

# Prefix caching on hybrid GDN+MTP had open upstream bugs (vllm#48375,
# vllm#55766). It passed correctness testing on our build, but validate on
# yours before trusting it in production.
if [[ "$ENABLE_PREFIX_CACHING" == "1" ]]; then
  VLLM_ARGS+=(--enable-prefix-caching)
else
  VLLM_ARGS+=(--no-enable-prefix-caching)
fi

# Async scheduling is ON when the flag is omitted; 1 forces it off.
[[ "$DISABLE_ASYNC_SCHEDULING" == "1" ]] && VLLM_ARGS+=(--no-async-scheduling)

if [[ -n "${EXTRA_VLLM_ARGS:-}" ]]; then
  read -r -a _extra <<<"$EXTRA_VLLM_ARGS"
  VLLM_ARGS+=("${_extra[@]}")
fi

echo "============================================================"
echo " Qwen3.8-27B INT8 W8A8 v2-a16 / 4x RTX 3060 12GB"
echo "============================================================"
echo " model      : $MODEL_DIR"
echo " profile    : $PROFILE   context $MAX_MODEL_LEN   KV $KV_CACHE_DTYPE"
echo " parallel   : TP$TP_SIZE on GPUs $CUDA_VISIBLE_DEVICES"
echo " batching   : $MAX_NUM_SEQS seqs / $MAX_NUM_BATCHED_TOKENS batched tokens"
echo " mtp        : $STAGE      prefix caching: $ENABLE_PREFIX_CACHING"
echo " listen     : $HOST:$PORT"
echo "============================================================"
echo "After startup, check the KV cache line and both kernels:"
echo "  grep -E 'GPU KV cache size|Maximum concurrency' <log>"
echo "  grep -oE '(Selected|Using) [A-Za-z0-9]+ (Kernel )?for [A-Za-z0-9]+' <log> | sort | uniq -c"
echo "============================================================"

exec vllm serve "${VLLM_ARGS[@]}"
