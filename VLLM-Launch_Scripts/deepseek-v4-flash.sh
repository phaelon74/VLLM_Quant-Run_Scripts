#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# DeepSeek-V4-Flash on 4x RTX 6000 PRO Workstation (Blackwell, SM120)
# Single-node text-serving defaults for vLLM (native `vllm serve`)
#
# Model: deepseek-ai/DeepSeek-V4-Flash
#   - 284B total params, 13B activated (MoE)
#   - FP4 (MoE experts) + FP8 (attention/norm/router) mixed checkpoint
#   - Hybrid attention: Compressed Sparse Attn (c4a) + Heavily
#     Compressed Attn (c128a) + short SWA, reaching 1M context
#   - Manifold-Constrained Hyper-Connections (mHC)
#   - 3 reasoning modes: Non-think, Think High, Think Max
#     (Think Max requires max-model-len >= 393,216)
#
# References used to build this file:
#   - https://vllm.ai/blog/deepseek-v4
#   - https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash
#   - https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash
#
# IMPORTANT RUNTIME REQUIREMENT
#   vLLM V4 support landed with PR #40760 and custom c4a/c128a
#   attention + FP4 MoE kernels. Upstream packages these in the
#   `vllm/vllm-openai:deepseekv4-cu130` Docker image. This script
#   uses NATIVE `vllm serve`, so you MUST be running a nightly (or
#   source build) built against CUDA 13.0 that includes the V4
#   model registry, the `deepseek_v4` tokenizer mode, and the V4
#   attention kernels. If `vllm --version` predates that PR, this
#   script will fail at model load.
#
# SM120 (RTX 6000 Pro Blackwell) CAVEAT
#   The reference V4 build targets SM100 (B200) / SM103 (B300).
#   SM120 support for NVFP4 MoE kernels has been landing piecemeal
#   (see vllm#31085, vllm#33417). Expect one of:
#     (a) FlashInfer/CUTLASS NVFP4 MoE kernels work natively on SM120
#         -> full speed
#     (b) Backend selector falls back to Marlin for FP4 MoE
#         -> functional but slower; set VLLM_MXFP4_USE_MARLIN=1 if
#            it doesn't auto-fall-back cleanly
#     (c) A kernel asserts on sm_120
#         -> open an issue upstream; in the meantime try --enforce-eager
#            and disable FP4 indexer cache
# ============================================================

# --- Memory / Allocator ---
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- GPU Selection (TP/EP across all four cards) ---
export CUDA_VISIBLE_DEVICES=0,1,2,3

# --- NCCL on PCIe-only consumer/pro-viz topology ---
# 4x RTX 6000 Pro Blackwell = no NVLink, PCIe Gen5 x16 only.
# This is the single biggest perf throttle on this box vs a B200 node:
# all2all for EP MoE must traverse PCIe (~63 GB/s/dir) instead of
# NVLink 5 (~1.8 TB/s/GPU). Plan decode throughput accordingly.
export NCCL_IB_DISABLE=1
# PHB is the safe default for 4 GPUs sharing a CPU root complex.
# If you have a PLX/PEX switch between GPU pairs, PIX is usually
# slightly faster. Use `nvidia-smi topo -m` to check.
export NCCL_P2P_LEVEL=PHB
# Consumer PCIe P2P can be flaky; if you see NCCL hangs on startup,
# try disabling P2P entirely as a first debug step:
# export NCCL_P2P_DISABLE=1

# --- vLLM behavior ---
export VLLM_NO_USAGE_STATS=1

# --- Loading acceleration ---
export SAFETENSORS_FAST_GPU=1

# --- CPU threads (tune to total cores / num GPUs, roughly) ---
export OMP_NUM_THREADS=8

# --- SM120 FP4 fallback knob (uncomment if FP4 MoE asserts on sm_120) ---
# export VLLM_MXFP4_USE_MARLIN=1

# ============================================================
# Parallelism: DP=4 + Expert Parallel (the V4 recommended pattern)
#
# Why not TP=4 for this model:
#   - V4 uses MLA-style attention with shared/compressed KV and a
#     per-request rolling compressor residual. The vLLM V4 code
#     paths are designed around DP attention (each rank runs full
#     attention for its own subset of the batch, zero attention-
#     time cross-GPU comm). TP on MLA adds a per-layer AllReduce
#     and does not meaningfully shard the tiny latent cache.
#   - Every official V4-Flash recipe on recipes.vllm.ai uses
#     DP=4 + EP. TP=4 + EP on V4 is untested publicly and may
#     silently fall through to an unoptimized path or assert.
#   - On PCIe-only topology, stacking TP AllReduce ON TOP OF EP
#     all2all is especially painful.
#
# Best-guess perf delta on this specific hardware:
#   DP=4 + EP (this script) : baseline
#   TP=4 + EP               : ~20-50% slower decode, worse at long
#                             context. Included below as a toggle.
# ============================================================

PARALLEL_MODE="dp_ep"   # "dp_ep" (recommended) or "tp_ep" (experimental)

case "$PARALLEL_MODE" in
  dp_ep)
    TP_SIZE=1
    DP_SIZE=4
    ;;
  tp_ep)
    TP_SIZE=4
    DP_SIZE=1
    ;;
  *)
    echo "Unknown PARALLEL_MODE: $PARALLEL_MODE" >&2
    exit 1
    ;;
esac

MODEL_DIR="/media/fmodels/deepseek-ai/DeepSeek-V4-Flash"
SERVED_MODEL_NAME="deepseek-v4-flash"

# ------------------------------------------------------------
# Context length
#
# The user picked full 1M per the blog's KV-cache math (fp8 attn +
# fp4 indexer ~= 5 GiB/seq at 1M). Think Max requires >= 393,216.
#
# Per-GPU VRAM envelope on 96 GB cards (DP=4 + EP=4):
#   weights (replicated non-expert + 1/4 expert shard)   ~55-65 GB
#   CUDA graphs / activations / workspace                ~3-6 GB
#   leftover for KV                                       ~25-35 GB
# At 5 GiB/seq@1M that means ~5-6 active 1M-context sequences
# per rank. Start conservative with max-num-seqs and watch.
# ------------------------------------------------------------
MAX_MODEL_LEN=1048576

GPU_MEMORY_UTILIZATION=0.92
MAX_NUM_SEQS=16
MAX_NUM_BATCHED_TOKENS=16384
BLOCK_SIZE=256          # V4 requires 256-native-token logical block

# ------------------------------------------------------------
# Optional optimizations from the vLLM V4 blog
# ------------------------------------------------------------
ENABLE_PREFIX_CACHING=1
ENABLE_MTP=1                 # Multi-Token Prediction (native spec decode)
ENABLE_FP4_INDEXER_CACHE=1   # Halves indexer cache vs bf16 estimate

# Upstream vLLM currently advertises num_speculative_tokens=1 as
# the stable default for DeepSeek MTP (V3/V3.2/V4 share the code
# path). k=2 has shown up in some downstream forks but is not
# upstream-guaranteed yet for V4.
MTP_SPEC='{"method":"deepseek_mtp","num_speculative_tokens":1}'

# Torch.compile / Inductor on the dsv4 preview branch hits
# `AssertionError: auto_functionalized was not removed` during
# the decompose_auto_functionalized post-grad pass, because one
# of V4's new custom ops (likely sparse_attn_indexer) isn't yet
# fully lowerable by Inductor. The official V4 recipes work
# around this with --enforce-eager, so we do the same here.
#
# When the PR merges and the custom-op registration is finalized,
# set ENFORCE_EAGER=0 and switch back to the full compile path:
#   COMPILATION_CONFIG='{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'
ENFORCE_EAGER=1
COMPILATION_CONFIG='{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'

# ------------------------------------------------------------
# Assemble the argument list
# ------------------------------------------------------------
VLLM_ARGS=(
  "$MODEL_DIR"
  --served-model-name "$SERVED_MODEL_NAME"
  --tensor-parallel-size "$TP_SIZE"
  --data-parallel-size "$DP_SIZE"
  --enable-expert-parallel
  --trust-remote-code
  --dtype auto

  # V4 hybrid KV cache essentials
  --kv-cache-dtype fp8
  --block-size "$BLOCK_SIZE"

  # Tokenizer + V4 chat/tool/reasoning parsers
  --tokenizer-mode deepseek_v4
  --tool-call-parser deepseek_v4
  --enable-auto-tool-choice
  --reasoning-parser deepseek_v4

  # Capacity planning
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"

  # Serving
  --api-key REDACTED
  --host 0.0.0.0
  --port 8080
  --disable-uvicorn-access-log
)

if [[ "$ENABLE_PREFIX_CACHING" == "1" ]]; then
  VLLM_ARGS+=(--enable-prefix-caching)
fi

if [[ "$ENABLE_MTP" == "1" ]]; then
  VLLM_ARGS+=(--speculative-config "$MTP_SPEC")
fi

if [[ "$ENABLE_FP4_INDEXER_CACHE" == "1" ]]; then
  # Listed in the blog as an optional optimization alongside MTP.
  # Halves indexer KV-cache memory with negligible quality impact.
  VLLM_ARGS+=(--attention_config.use_fp4_indexer_cache=True)
fi

if [[ "$ENFORCE_EAGER" == "1" ]]; then
  # Skip torch.compile / Inductor. CUDA graphs are still captured
  # (vLLM captures them independently of Inductor). This avoids the
  # decompose_auto_functionalized assertion on the dsv4 branch.
  VLLM_ARGS+=(--enforce-eager)
else
  VLLM_ARGS+=(--compilation-config "$COMPILATION_CONFIG")
fi

echo "============================================================"
echo "Launching DeepSeek-V4-Flash"
echo "  parallel mode : $PARALLEL_MODE  (TP=$TP_SIZE DP=$DP_SIZE)"
echo "  max model len : $MAX_MODEL_LEN"
echo "  max num seqs  : $MAX_NUM_SEQS"
echo "  mtp           : $ENABLE_MTP (k=1)"
echo "  fp4 indexer   : $ENABLE_FP4_INDEXER_CACHE"
echo "  prefix cache  : $ENABLE_PREFIX_CACHING"
echo "  enforce eager : $ENFORCE_EAGER"
echo "============================================================"

vllm serve "${VLLM_ARGS[@]}"
