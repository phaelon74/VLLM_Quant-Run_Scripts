#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# nvidia/MiniMax-M3-NVFP4 on 4x RTX PRO 6000 Blackwell Workstation via vLLM
# ============================================================
#
# Model:
#   - HF id:  nvidia/MiniMax-M3-NVFP4  (local: /media/fmodels/nvidia/MiniMax-M3-NVFP4)
#   - Base:   MiniMaxAI/MiniMax-M3 (minimax_m3_vl), 428B total / ~23B active MoE,
#             native multimodal (text + image + video), 1M-token context.
#   - Quant:  NVFP4 (W4A4) produced with nvidia-modelopt v0.44.0. vLLM
#             AUTO-DETECTS this from hf_quant_config.json / quantization_config,
#             so we do NOT pass --quantization. Forcing "modelopt_fp4" here can
#             select the wrong kernel path.
#
# Architecture notes that drive the flags below:
#   - MSA (MiniMax Sparse Attention): scores fixed 128-token KV blocks, selects
#     top-k blocks per query/GQA group, runs sparse GQA over the selection.
#     ==> --block-size 128 is MANDATORY. The vLLM default (16) misaligns the
#         sparse index cache and the engine will not start.
#   - Hybrid layers: some layers are dense attention, some route to the MSA
#     backend. The recipe says to let vLLM pick per layer and treats
#     TRITON_ATTN as AMD-only, but that assumes sm_90/sm_100. On sm_120 the
#     dense layers pick FlashInfer, which cannot serve MSA's 128-token KV
#     blocks, so we pin Triton. See ATTENTION_BACKEND below.
#   - Vision encoder is small relative to the MoE, so at TP>1 the recipe runs it
#     data-parallel (--mm-encoder-tp-mode data) with FlashInfer attention and a
#     host shared-memory processor cache.
#   - No MTP. MiniMax confirmed M3 ships no multi-token-prediction head
#     (MiniMax-AI/MiniMax-M3 issue #13, closed "Don't support MTP now").
#     The supported spec-decode path is EAGLE3 with an external draft head --
#     see the EAGLE3 section below.
#
# VRAM fit on 4x 96 GiB (384 GiB total):
#   - NVFP4 weights land around ~230 GiB on disk (4-bit packed MoE + FP8 scales
#     + BF16 vision tower / embeddings / norms). At TP=4 that is ~57-58 GiB of
#     resident weights per card.
#   - At GPU_MEMORY_UTILIZATION=0.90 (~86 GiB usable) that leaves roughly
#     ~28 GiB/GPU for KV cache + activations + vision workspace + CUDA graphs
#     + the EAGLE3 draft head. Comfortable, but NOT enough for the full 1M
#     window -- hence the 128K default below.
#   - fp8 KV cache is on by default: the vLLM MiniMax-M3 recipe calls it
#     "lossless in our testing across the full native context" and it buys
#     ~1.5x the KV pool.
#
# IMPORTANT hardware caveat (read before first launch):
#   RTX PRO 6000 Blackwell Workstation cards are sm_120, not sm_100 (B200).
#   The NVFP4 checkpoint was validated by NVIDIA on B200/sm_100, and much of
#   that path is sm_100-gated. Observed kernel selection on this host:
#     MoE:      MARLIN, out of [FLASHINFER_TRTLLM, FLASHINFER_CUTLASS, MARLIN]
#     MSA:      Triton  (no fmha_sm100)
#     Indexer:  Triton  (sm100=False)
#     Linear:   FlashInferCutlassMxfp8LinearKernel for the MXFP8 tensors
#   The Marlin selection comes with this expected warning:
#     "Your GPU does not have native support for FP4 computation ...
#      Weight-only FP4 compression will be used leveraging the Marlin kernel."
#   That is informational, not an error. sm_120 has FP4 tensor cores, but vLLM's
#   native NVFP4 GEMM is compiled for sm_100a only, so the MoE runs weight-only
#   (W4A16-style) through Marlin: correct results, unpacking cost on the compute
#   -heavy path. Nothing to fix here -- it is the ceiling of this hardware until
#   an sm_120 NVFP4 GEMM lands upstream. Do NOT set MOE_BACKEND=triton to
#   "work around" it; Marlin is already the best available choice.
#
#   These boxes also have no NVLink and no PCIe P2P, so custom all-reduce is
#   disabled by default and NCCL is pinned to a host-staged path. Expect
#   "SymmMemCommunicator: Device capability 12.0 not supported" (harmless --
#   vLLM falls back to PYNCCL for the all-reduce).
#
# vLLM REQUIREMENTS:
#   - MiniMax-M3 support is not in a stable release yet. Use the dedicated
#     image (docker pull vllm/vllm-openai:minimax-m3) or a nightly that
#     contains PR #45381 (M3 support) AND PR #46380 (M3 NVFP4 support).
#     Sanity check:  vllm serve --help | grep minimax_m3
#
# This script does not store API keys. Pass the key as the first argument, or
# set VLLM_API_KEY in the environment.
# ============================================================

# --- Memory Management ---
export PYTORCH_ALLOC_CONF=expandable_segments:True

# --- GPU Selection (four Blackwell workstation cards) ---
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

# --- vLLM / CUDA behavior ---
export VLLM_NO_USAGE_STATS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0

# Required by the validated NVFP4 recipe: keeps the fp32 accumulate paths in
# high precision so NVFP4 activation quantization does not compound error.
export VLLM_FLOAT32_MATMUL_PRECISION="${VLLM_FLOAT32_MATMUL_PRECISION:-high}"

# Loading a ~230 GiB checkpoint across 4 workers can exceed the default engine
# readiness timeout on spinning/NFS storage.
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}"

# --- NCCL for PCIe-only workstation Blackwell (no NVLink, no P2P) ---
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-8}"

# --- Loading / CPU threading ---
export SAFETENSORS_FAST_GPU=1
export OMP_NUM_THREADS=8

# --- Torch profiler (registers POST /start_profile and /stop_profile) ---
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

MODEL_DIR="${MODEL_DIR:-/media/fmodels/nvidia/MiniMax-M3-NVFP4}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-MiniMax-M3-NVFP4}"
TOKENIZER="${TOKENIZER:-$MODEL_DIR}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"

# Four cards, tensor parallel across all of them.
TP_SIZE="${TP_SIZE:-4}"

# Expert parallel: shards MoE experts instead of slicing every expert across
# ranks. On an NVLink node this is usually a win for a 428B MoE. On these
# PCIe-only workstation cards the MoE all-to-all has to cross the CPU root
# complex, which frequently costs more than the kernel efficiency it buys.
# Default OFF (this also matches NVIDIA's validated NVFP4 baseline, which is
# plain TP). Flip to 1 and measure before trusting either way.
ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-0}"

# ------------------------------------------------------------
# Context sizing
#
# The config advertises 1,048,576 tokens. Do NOT let vLLM default to that --
# it will try to size the KV pool for a 1M window and fail allocation. MSA
# keeps per-token attention compute bounded, but the KV cache itself is still
# stored densely (sparsity is applied in the compute path, not in storage),
# and M3 additionally keeps a separate indexer K cache.
#
# 131072 (128K) is the default here: covers essentially every coding/agent
# session, and leaves KV headroom for MAX_NUM_SEQS concurrent requests plus
# the vision encoder and the EAGLE3 draft cache.
#
# Escape valves if you need more:
#   MAX_MODEL_LEN=262144 ....... 256K, drop MAX_NUM_SEQS to 2
#   TEXT_ONLY=1 ................ skips loading the vision tower entirely
#   SPEC_DRAFT_MODE=off ........ frees the draft head + its KV
#   GPU_MEMORY_UTILIZATION=0.93  squeezes ~3 GiB/GPU more
# ------------------------------------------------------------
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"

# MANDATORY for MiniMax-M3: MSA's sparse block granularity is 128 tokens and
# the vLLM cache block size must match it. Do not change this.
BLOCK_SIZE="${BLOCK_SIZE:-128}"

# The recipe reports fp8 KV as lossless for M3 across the full native context
# and it buys ~1.5x KV pool. Set to "auto" to fall back to native dtype.
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"

TOKENIZER_MODE="${TOKENIZER_MODE:-hf}"
CONFIG_FORMAT="${CONFIG_FORMAT:-auto}"
LOAD_FORMAT="${LOAD_FORMAT:-auto}"

# Leave EMPTY. vLLM reads the ModelOpt NVFP4 config out of the checkpoint;
# naming a method here can pick a different (wrong) kernel path.
QUANTIZATION="${QUANTIZATION:-}"

# M3 ships custom modeling/processor code and the NVFP4 recipe passes this.
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"

# Attention backend for the TARGET model. Pinned to TRITON_ATTN, which is NOT
# what the NVIDIA recipe does -- it leaves this to vLLM's per-layer auto-select.
# On sm_120 auto-select is wrong:
#
#   MSA and the indexer already choose Triton on their own (both log
#   "no fmha_sm100"), but the model's DENSE attention layers choose FlashInfer:
#     "Using FLASHINFER attention backend out of potential backends:
#      ['FLASHINFER', 'TRITON_ATTN']"
#   FlashInfer cannot serve MSA's mandatory 128-token KV blocks, and vLLM needs
#   one kernel block size that every attention group can satisfy, so KV-cache
#   profiling dies with "ValueError: No common block size for 128."
#   Moving only the EAGLE3 draft head to Triton is NOT enough -- the failure
#   still reports FlashInferBackend, because the target's own dense group is the
#   holdout. It has to be Triton on both sides.
#
# Pinning Triton also clears the companion warning that was silently destroying
# decode performance:
#   "CUDAGraphMode.FULL_DECODE_ONLY is not supported with spec-decode for
#    attention backend FlashInferBackend ...; setting cudagraph_mode=NONE"
#
# This matches the AMD recipe (which pins TRITON_ATTN unconditionally) and the
# one published sm_12x MiniMax-M3 config. Set to empty to restore auto-select
# if a future vLLM lands a 128-block-capable FlashInfer path for sm_120.
#
# NOTE: this does not affect the vision encoder, which negotiates separately
# via MM_ENCODER_ATTN_BACKEND and is fine on FlashInfer (it has no KV cache).
ATTENTION_BACKEND="${ATTENTION_BACKEND:-TRITON_ATTN}"

# MoE kernel selection. Leave EMPTY. vLLM's oracle already picks MARLIN on
# sm_120 (the fused FlashInfer NVFP4 paths are sm_100-gated), which is the
# fastest thing available here. Only override to debug a suspected MoE kernel
# bug.
MOE_BACKEND="${MOE_BACKEND:-}"

# ============================================================
# Reasoning / tool-call / thinking-mode
# ============================================================
#
# Both parsers are "minimax_m3" -- NOT "minimax_m2", which is what the earlier
# MiniMax releases used. Using the m2 parsers on M3 silently mis-splits the
# <mm:think> block into content.
REASONING_PARSER="${REASONING_PARSER:-minimax_m3}"

ENABLE_TOOL_CALLING="${ENABLE_TOOL_CALLING:-1}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-minimax_m3}"

# M3 has three thinking modes, selected per request via
#   extra_body={"chat_template_kwargs": {"thinking_mode": "<mode>"}}
#     enabled  - always think first, including after tool results
#     disabled - answer directly, lowest latency
#     adaptive - model decides (this is M3's own default when unset)
# The same value tunes the minimax_m3 reasoning parser, so reasoning and
# content get split correctly in every mode.
#
# Set a server-side default here. Empty string = leave M3 on "adaptive".
DEFAULT_THINKING_MODE="${DEFAULT_THINKING_MODE:-adaptive}"

# ============================================================
# Multimodal: 8 images per prompt
# ============================================================
#
# TEXT_ONLY=1 passes --language-model-only, which skips loading the vision
# tower and frees its VRAM. It is MUTUALLY EXCLUSIVE with the data-parallel
# encoder flags below, so the whole vision block is gated on it.
TEXT_ONLY="${TEXT_ONLY:-0}"

LIMIT_MM_IMAGE="${LIMIT_MM_IMAGE:-8}"
LIMIT_MM_VIDEO="${LIMIT_MM_VIDEO:-1}"

# Vision encoder placement. The encoder is small next to the MoE, so running
# it data-parallel across the 4 ranks avoids TP communication overhead for a
# tensor that barely needs splitting -- doubly true on PCIe-only cards.
MM_ENCODER_TP_MODE="${MM_ENCODER_TP_MODE:-data}"
MM_ENCODER_ATTN_BACKEND="${MM_ENCODER_ATTN_BACKEND:-FLASHINFER}"

# Host shared-memory cache for preprocessed media. Keeps decode/resize/patch
# work off the GPU critical path and shares it across the 4 workers.
# NOTE: this wants a real /dev/shm. Under Docker add --shm-size=16g.
MM_PROCESSOR_CACHE_TYPE="${MM_PROCESSOR_CACHE_TYPE:-shm}"

# Video frame sampling. -1 lets the processor decide from the clip.
if [[ -z "${MEDIA_IO_KWARGS:-}" ]]; then
  MEDIA_IO_KWARGS='{"video":{"num_frames":-1}}'
fi

# ============================================================
# EAGLE3 speculative decoding
# ============================================================
#
# M3 has NO MTP head -- MiniMax closed the request for it (MiniMax-M3 #13).
# EAGLE3 is the supported and day-0-validated spec-decode path. Both draft
# heads share M3's embedding table and LM head, and vLLM's MSA decode kernels
# were extended to verify multiple draft tokens in the decode-specialized
# split-K path (rather than falling back to prefill kernels), so this stays
# cudagraph-friendly.
#
# SPEC_DRAFT_MODE:
#   gqa - Inferact/MiniMax-M3-EAGLE3-GQA, 4 KV heads. 16x smaller draft KV
#         cache. DEFAULT HERE: on 4 cards holding a 428B model, draft KV is
#         very much the binding constraint.
#   mha - Inferact/MiniMax-M3-EAGLE3, 64 KV heads. The recipe default; use it
#         only if you cap MAX_MODEL_LEN low and want the baseline acceptance.
#   off - no speculative decoding.
#
# Validation caveat: NVIDIA's B200 EAGLE3 numbers were measured with
# --language-model-only. EAGLE3 layered on top of the multimodal path is the
# less-travelled combination; if you see draft/verify shape errors on image
# requests, set SPEC_DRAFT_MODE=off and re-test.
#
# 3 speculative tokens is the validated starting point (~67% acceptance,
# mean accept length ~3.0 on Sonnet-style traffic). Tune against your own
# acceptance rate before changing it.
SPEC_DRAFT_MODE="${SPEC_DRAFT_MODE:-gqa}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-3}"

# Draft-head attention backend. Two backends were ruled out empirically here,
# in this order:
#
#   FLASH_ATTN (what the recipe says) -- a B200/sm_100 assumption. FlashAttention
#     only serves an fp8 KV cache through FA3 on sm_90 or FA4 on sm_100, and
#     sm_120 has neither, so it hard-fails at drafter load:
#       "Selected backend FLASH_ATTN is not valid ... FP8 KV cache requires
#        FA3 on SM90 or FA4 on SM100"
#
#   FLASHINFER -- loads the draft head fine, but breaks later on two counts:
#       "CUDAGraphMode.FULL_DECODE_ONLY is not supported with spec-decode for
#        attention backend FlashInferBackend
#        (support: AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE);
#        setting cudagraph_mode=NONE"
#     ...and then fatally, during KV-cache profiling:
#       "ValueError: No common block size for 128."
#     MSA forces the KV manager to 128-token blocks, and vLLM has to find one
#     kernel block size every attention group can serve. FlashInfer cannot do
#     128 here, so the intersection across the target's groups and the draft
#     group comes back empty.
#
# TRITON_ATTN handles arbitrary block sizes including 128 and has the better
# cudagraph support class for speculative verification. It is also what the
# only other published sm_12x MiniMax-M3 config uses for both the model and
# the draft head.
SPEC_ATTENTION_BACKEND="${SPEC_ATTENTION_BACKEND:-TRITON_ATTN}"

case "$SPEC_DRAFT_MODE" in
  gqa) SPEC_DRAFT_MODEL="${SPEC_DRAFT_MODEL:-/media/fmodels/Inferact/MiniMax-M3-EAGLE3-GQA}" ;;
  mha) SPEC_DRAFT_MODEL="${SPEC_DRAFT_MODEL:-/media/fmodels/Inferact/MiniMax-M3-EAGLE3}" ;;
  off) SPEC_DRAFT_MODEL="" ;;
  *)   echo "ERROR: SPEC_DRAFT_MODE must be one of: gqa, mha, off (got '$SPEC_DRAFT_MODE')" >&2
       exit 3 ;;
esac

# The draft heads are small (~3.3B). If you have not mirrored them locally,
# point SPEC_DRAFT_MODEL at the hub id instead:
#   SPEC_DRAFT_MODEL=Inferact/MiniMax-M3-EAGLE3-GQA ./minimax-m3-nvfp4-tp4.sh KEY
if [[ -n "$SPEC_DRAFT_MODEL" && "$SPEC_DRAFT_MODEL" == /* && ! -d "$SPEC_DRAFT_MODEL" ]]; then
  echo "ERROR: EAGLE3 draft head not found at $SPEC_DRAFT_MODEL" >&2
  echo "       Either download it:" >&2
  echo "         hf download Inferact/MiniMax-M3-EAGLE3-GQA --local-dir $SPEC_DRAFT_MODEL" >&2
  echo "       or pass the hub id:  SPEC_DRAFT_MODEL=Inferact/MiniMax-M3-EAGLE3-GQA $0 KEY" >&2
  echo "       or disable it:       SPEC_DRAFT_MODE=off $0 KEY" >&2
  exit 4
fi

# ============================================================
# Scheduling / graph capture
# ============================================================

# Large win for agent + multi-turn coding traffic, which is what M3 is for.
# Turn off only when benchmarking.
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"

# Async scheduling overlaps Python scheduling with GPU decode. Note that some
# vLLM builds refuse to combine it with speculative decoding; if the engine
# complains at startup, set this to 0.
ENABLE_ASYNC_SCHEDULING="${ENABLE_ASYNC_SCHEDULING:-0}"

# NOTE on what actually happens to COMPILATION_CONFIG here: on MiniMax-M3 this
# vLLM build auto-enables VLLM_USE_BREAKABLE_CUDAGRAPH=1, and that in turn
# disables the torch.compile/Inductor pipeline entirely -- the log says
#   "Auto-enabling VLLM_USE_BREAKABLE_CUDAGRAPH=1"
#   "VLLM_USE_BREAKABLE_CUDAGRAPH is set, disabling vLLM's torch.compile
#    pipeline. Equivalent to -cc.mode=none."
# So the mode:3 below is silently downgraded to mode NONE, and the two
# "Inductor compilation was disabled by user settings" warnings per rank are
# expected, not a misconfiguration. cudagraph_mode/capture_sizes still apply.
# The breakable-cudagraph path is how M3 keeps the MSA collectives capturable;
# AMD's recipe forces it off (VLLM_USE_BREAKABLE_CUDAGRAPH=0) to regain graphs,
# but on sm_120 the reported failure mode for large-MoE capture is an illegal
# memory access at capture_end(), so leave the auto-choice alone unless you are
# deliberately testing it.
#
# CUDA graphs: full_decode_only keeps graphs on the decode path (where MSA's
# many small indexer/top-k kernels make launch overhead matter most) while
# skipping the memory-hungry full-prefill captures. Capture sizes are kept
# conservative because M3 mixes dense and MSA layers and the NVFP4 MoE path on
# sm_120 is the least-validated part of this stack.
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-full_decode_only}"
CUDAGRAPH_CAPTURE_SIZES="${CUDAGRAPH_CAPTURE_SIZES:-[1,2,4]}"
if [[ -z "${COMPILATION_CONFIG:-}" ]]; then
  COMPILATION_CONFIG="{\"mode\":3,\"cudagraph_mode\":\"${CUDAGRAPH_MODE}\",\"cudagraph_capture_sizes\":${CUDAGRAPH_CAPTURE_SIZES}}"
fi

MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-}"

# Last-resort escape hatch: disables torch.compile and CUDA graphs entirely.
# Costs a lot of decode throughput; use only to isolate a startup crash.
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"

# No NVLink / no PCIe P2P on these cards -- the custom all-reduce kernel
# assumes peer access and will hang or corrupt without it.
DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-1}"

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
  --block-size "$BLOCK_SIZE"
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

if [[ "$DISABLE_CUSTOM_ALL_REDUCE" == "1" ]]; then
  VLLM_ARGS+=(--disable-custom-all-reduce)
fi

if [[ "$ENABLE_EXPERT_PARALLEL" == "1" ]]; then
  VLLM_ARGS+=(--enable-expert-parallel)
fi

if [[ -n "$QUANTIZATION" ]]; then
  VLLM_ARGS+=(--quantization "$QUANTIZATION")
fi

if [[ -n "$MOE_BACKEND" ]]; then
  VLLM_ARGS+=(--moe-backend "$MOE_BACKEND")
fi

if [[ -n "$ATTENTION_BACKEND" ]]; then
  VLLM_ARGS+=(--attention-backend "$ATTENTION_BACKEND")
fi

if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
  VLLM_ARGS+=(--trust-remote-code)
fi

if [[ "$ENFORCE_EAGER" == "1" ]]; then
  VLLM_ARGS+=(--enforce-eager)
else
  VLLM_ARGS+=(--compilation-config "$COMPILATION_CONFIG")
fi

if [[ -n "$MAX_CUDAGRAPH_CAPTURE_SIZE" ]]; then
  VLLM_ARGS+=(--max-cudagraph-capture-size "$MAX_CUDAGRAPH_CAPTURE_SIZE")
fi

if [[ -n "$REASONING_PARSER" ]]; then
  VLLM_ARGS+=(--reasoning-parser "$REASONING_PARSER")
fi

if [[ "$ENABLE_TOOL_CALLING" == "1" ]]; then
  VLLM_ARGS+=(--enable-auto-tool-choice --tool-call-parser "$TOOL_CALL_PARSER")
fi

if [[ -n "$DEFAULT_THINKING_MODE" ]]; then
  VLLM_ARGS+=(--default-chat-template-kwargs "{\"thinking_mode\": \"${DEFAULT_THINKING_MODE}\"}")
fi

if [[ "$TEXT_ONLY" == "1" ]]; then
  VLLM_ARGS+=(--language-model-only)
else
  VLLM_ARGS+=(--limit-mm-per-prompt "{\"image\": ${LIMIT_MM_IMAGE}, \"video\": ${LIMIT_MM_VIDEO}}")
  VLLM_ARGS+=(--mm-encoder-tp-mode "$MM_ENCODER_TP_MODE")
  VLLM_ARGS+=(--mm-encoder-attn-backend "$MM_ENCODER_ATTN_BACKEND")
  VLLM_ARGS+=(--mm-processor-cache-type "$MM_PROCESSOR_CACHE_TYPE")
  if [[ -n "$MEDIA_IO_KWARGS" ]]; then
    VLLM_ARGS+=(--media-io-kwargs "$MEDIA_IO_KWARGS")
  fi
fi

if [[ "$ENABLE_PREFIX_CACHING" == "1" ]]; then
  VLLM_ARGS+=(--enable-prefix-caching)
fi

if [[ "$ENABLE_ASYNC_SCHEDULING" == "1" ]]; then
  VLLM_ARGS+=(--async-scheduling)
fi

if [[ -n "$SPEC_DRAFT_MODEL" ]]; then
  VLLM_ARGS+=(--speculative-config "{\"method\":\"eagle3\",\"model\":\"${SPEC_DRAFT_MODEL}\",\"num_speculative_tokens\":${NUM_SPECULATIVE_TOKENS},\"attention_backend\":\"${SPEC_ATTENTION_BACKEND}\"}")
fi

if [[ "$PROFILE" == "1" ]]; then
  VLLM_ARGS+=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${PROFILE_DIR}\"}")
fi

echo "Launching vLLM (MiniMax-M3 NVFP4, TP4 Blackwell) with:"
echo "  MODEL_DIR=${MODEL_DIR}"
echo "  SERVED_MODEL_NAME=${SERVED_MODEL_NAME}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  TP_SIZE=${TP_SIZE}  EXPERT_PARALLEL=${ENABLE_EXPERT_PARALLEL}"
echo "  BLOCK_SIZE=${BLOCK_SIZE} (mandatory for MSA)"
echo "  MAX_MODEL_LEN=${MAX_MODEL_LEN}  MAX_NUM_SEQS=${MAX_NUM_SEQS}"
echo "  GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"
echo "  KV_CACHE_DTYPE=${KV_CACHE_DTYPE}"
echo "  QUANTIZATION=${QUANTIZATION:-auto-detect (ModelOpt NVFP4)}"
echo "  MOE_BACKEND=${MOE_BACKEND:-auto}  ATTENTION_BACKEND=${ATTENTION_BACKEND:-auto (per-layer)}"
echo "  REASONING_PARSER=${REASONING_PARSER}  THINKING_MODE=${DEFAULT_THINKING_MODE:-<model default: adaptive>}"
echo "  TOOL_CALLING=${ENABLE_TOOL_CALLING} (parser=${TOOL_CALL_PARSER})"
if [[ "$TEXT_ONLY" == "1" ]]; then
  echo "  TEXT_ONLY=1 (--language-model-only, vision tower not loaded)"
else
  echo "  MM image=${LIMIT_MM_IMAGE} video=${LIMIT_MM_VIDEO}"
  echo "  MM encoder: tp_mode=${MM_ENCODER_TP_MODE} attn=${MM_ENCODER_ATTN_BACKEND} cache=${MM_PROCESSOR_CACHE_TYPE}"
fi
if [[ -n "$SPEC_DRAFT_MODEL" ]]; then
  echo "  SPEC=eagle3/${SPEC_DRAFT_MODE} (${SPEC_DRAFT_MODEL}, k=${NUM_SPECULATIVE_TOKENS}, backend=${SPEC_ATTENTION_BACKEND})"
else
  echo "  SPEC=off (M3 has no MTP head; EAGLE3 is the only spec path)"
fi
echo "  PREFIX_CACHING=${ENABLE_PREFIX_CACHING}  ASYNC_SCHEDULING=${ENABLE_ASYNC_SCHEDULING}"
if [[ "$ENFORCE_EAGER" == "1" ]]; then
  echo "  ENFORCE_EAGER=1 (torch.compile + CUDA graphs DISABLED)"
else
  echo "  COMPILATION_CONFIG=${COMPILATION_CONFIG}"
fi
if [[ "$PROFILE" == "1" ]]; then
  echo "  PROFILER=torch -> ${PROFILE_DIR} (POST /start_profile + /stop_profile)"
fi
echo
echo "Recommended sampling per the MiniMax-M3 model card:"
echo "  temperature=1.0, top_p=0.95, top_k=40"
echo "Default system prompt:"
echo "  You are a helpful assistant. Your name is MiniMax-M3 and is built by MiniMax."
echo "Per-request thinking control:"
echo "  extra_body={\"chat_template_kwargs\": {\"thinking_mode\": \"enabled|disabled|adaptive\"}}"
echo

vllm serve "${VLLM_ARGS[@]}"
