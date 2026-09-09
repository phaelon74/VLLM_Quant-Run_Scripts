#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# TheHouseOfTheDude/gemma-4-31B_PTQ (W8A16) on 2x RTX 3090 via vLLM
# ============================================================
#
# Model:
#   - HF id:   TheHouseOfTheDude/gemma-4-31B_PTQ  (branch: W8A16)
#   - Base:    google/gemma-4-31B  (arch: gemma4 / Gemma4ForConditionalGeneration)
#   - Quant:   compressed-tensors PTQ, W8A16 (INT8 channel-wise weights, BF16 acts)
#              Linear layers in the language model only.
#              IGNORED (kept BF16):
#                - lm_head (tied to embed_tokens anyway)
#                - the entire vision_tower (27 SigLIP-style encoder layers)
#                - embed_vision / multi_modal_projector
#
# Architecture notes (from upstream config + Gemma 4 model card):
#   - 60 transformer decoder layers in a 5:1 hybrid pattern:
#       5 x sliding_attention -> 1 x full_attention, repeated 10 times
#     -> 50 sliding-window layers (window=1024 tokens) + 10 full-attention layers
#   - Sliding layers: head_dim=256, 32 Q heads, 16 KV heads (GQA)
#   - Full layers:    global_head_dim=512, 32 Q heads, 4 KV heads (GQA),
#                     attention_k_eq_v=True (K and V are tied in full layers,
#                     "Unified Keys and Values" per the model card),
#                     proportional RoPE with partial_rotary_factor=0.25.
#   - final_logit_softcapping=30.0 (Gemma 4 reintroduces the soft cap from
#     Gemma 2; vLLM handles this internally, no flag needed).
#   - tie_word_embeddings=True (lm_head shares weights with embed_tokens).
#   - Native context: 262,144 tokens (256K).
#   - Vision encoder: SigLIP-style, 27 layers, 1152 hidden, fixed token budget
#     (70 / 140 / 280 (default) / 560 / 1120 tokens per image).
#   - MTP via external assistant drafter (google/gemma-4-31B-it-assistant).
#     vLLM's Gemma 4 MTP path wires the 4-layer drafter to share the
#     target's KV cache. Enabled via --speculative-config.
#
# IMPORTANT FIT NOTES for 2x 24 GiB RTX 3090:
#   - On-disk W8A16 checkpoint: ~36.1 GiB (linear INT8 + BF16 carve-outs).
#     After TP=2: ~18 GiB / GPU resident weight footprint.
#   - That leaves ~5-6 GiB / GPU for KV cache + activations + vision encoder
#     workspace + CUDA graphs. vLLM's official 2x A100 (80 GB) recipe uses
#     max-model-len 32768; on 2x 24 GiB INT8 we have to be MUCH tighter.
#   - Default below: 16384 ctx, 2 sequences. The 5:1 SWA pattern keeps KV
#     small even at this length: only 10 of 60 layers grow with context.
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
#
# vLLM REQUIREMENTS:
#   - Gemma 4 support is recent. You need a vLLM build that registers the
#     `gemma4` model_type, the `gemma4` reasoning parser, AND the `gemma4`
#     tool-call parser (PR #39027 and friends, CUDA 12.9+). If
#     `vllm serve --help | grep gemma4` returns nothing, upgrade first:
#       uv pip install -U vllm --pre \
#         --extra-index-url https://wheels.vllm.ai/nightly/cu129
#   - vLLM auto-selects per-layer attention backends for Gemma 4 (PR #38891)
#     so sliding layers can use FlashAttention while full layers fall back
#     to Triton for the global_head_dim=512 path. Do not pin a single
#     attention backend -- let vLLM choose per-layer.
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

# ============================================================
# Model / serving configuration
# ============================================================

API_KEY="${1:-${VLLM_API_KEY:-}}"

if [[ -z "$API_KEY" ]]; then
  echo "Usage: $0 API-KEY-HERE" >&2
  echo "       or: VLLM_API_KEY=API-KEY-HERE $0" >&2
  exit 2
fi

# Local W8A16 compressed-tensors checkpoint (W8A16 branch of the repo).
# Override at launch time if needed:
#   MODEL_DIR=/path/to/model ./gemma-4-31b-int8.sh
MODEL_DIR="${MODEL_DIR:-/media/fmodels/TheHouseOfTheDude/gemma-4-31B_PTQ}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-gemma-4-31B-INT8-W8A16}"
# The compressed-tensors checkpoint includes tokenizer.json/tokenizer_config.json
# and processor_config.json (multimodal). Keep tokenizer resolution local.
TOKENIZER="${TOKENIZER:-$MODEL_DIR}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"

TP_SIZE="${TP_SIZE:-2}"

# ------------------------------------------------------------
# Capacity sizing for 2x 24 GiB (W8A16) at full multimodal.
#
# KV cache math (BF16, after TP=2 split of KV heads):
#   sliding (50 layers, capped at window=1024):
#     50 * 1024 * (16/2 KV heads) * 256 head_dim * 2 (K+V) * 2B
#       ~= 400 MiB / GPU / sequence  (independent of context length)
#   full (10 layers, K=V tied):
#     10 *  ctx * (4/2 KV heads)  * 512 head_dim * 1 (K=V) * 2B
#     16K ctx -> ~160 MiB / GPU / sequence
#     32K ctx -> ~320 MiB / GPU / sequence
#     64K ctx -> ~640 MiB / GPU / sequence
#
# Empirical capacity from this exact host (gpu_mem_util=0.92, full VLM):
#   vLLM reports: "GPU KV cache size: 9,584 tokens" + "Max concurrency
#   for 16,384 tokens per request: 1.70x" -> total KV budget ~27,852 tokens.
#
# Default below: 24K context => ~27K/24K = 1.13x concurrency. Bump higher
# if you want more single-stream context, with these escape valves:
#   MAX_MODEL_LEN=27648 ........ ~1.0x concurrency (single-stream max)
#   KV_CACHE_DTYPE=fp8 ......... roughly doubles KV budget (~55K total)
#   TEXT_ONLY=1 ................ frees vision profiling (~1-2 GiB more)
#   GPU_MEMORY_UTILIZATION=0.94  squeezes another ~1 GiB
# Stacking fp8 + TEXT_ONLY can realistically push MAX_MODEL_LEN to ~96K.
#
# vLLM's official 2x80GB recipe uses 32768; we now beat that for single-
# stream length thanks to the W8A16 quant + the 5:1 SWA pattern.
# ------------------------------------------------------------
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"

# Gemma 4 supports fp8 KV cache per the official vLLM Gemma4 recipe's memory
# optimization section, but on 3090 it can be flaky and the savings are
# small here because only 10 of 60 layers cache anything substantial.
# Leave auto by default; flip to fp8 if you want to push MAX_MODEL_LEN.
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"

TOKENIZER_MODE="${TOKENIZER_MODE:-hf}"
CONFIG_FORMAT="${CONFIG_FORMAT:-auto}"
LOAD_FORMAT="${LOAD_FORMAT:-auto}"

# This checkpoint is compressed-tensors W8A16. CLI spelling uses a hyphen.
QUANTIZATION="${QUANTIZATION:-compressed-tensors}"

# Gemma 4 is a brand-new architecture (model_type=gemma4). vLLM has native
# support but trust_remote_code is shown as True in the official offline
# inference example, so default it on for safety.
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"

# ============================================================
# Gemma 4 reasoning / tool-call / multimodal toggles
# ============================================================

# Gemma 4 uses structured thinking with custom delimiters
# (<|channel>thought\n ... <channel|>). vLLM ships a dedicated parser that
# extracts this into the OpenAI `reasoning` field on chat completions.
REASONING_PARSER="${REASONING_PARSER:-gemma4}"

# Default thinking ON for the server. Clients can still flip per-request via
#   extra_body={"chat_template_kwargs": {"enable_thinking": false}}
# Set DEFAULT_THINKING=0 to flip the server-side default off.
DEFAULT_THINKING="${DEFAULT_THINKING:-1}"

# Native function calling using custom <|tool_call> ... special tokens.
# vLLM's gemma4 tool-call parser handles extraction.
ENABLE_TOOL_CALLING="${ENABLE_TOOL_CALLING:-1}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-gemma4}"

# Chat template resolution.
#
# vLLM needs an explicit chat template when the tokenizer doesn't declare one
# (otherwise: vllm.entrypoints.chat_utils.ChatTemplateResolutionError).
#
# Resolution order (first match wins):
#   1. CHAT_TEMPLATE env var, if explicitly set by the caller (manual override).
#   2. $MODEL_DIR/chat_template.jinja, if the quant ships one alongside the
#      weights. This is the preferred source -- it is the exact template
#      processor.save_pretrained() wrote during PTQ and is guaranteed to
#      match the tokenizer's special-token ids (e.g. <turn|>=106).
#   3. /home/phaedawg/chat_templates/tool_chat_template_gemma4.jinja
#      (vLLM's upstream Gemma 4 template) as a fallback for older quants
#      that did NOT ship chat_template.jinja.
#
# Older PTQ runs of TheHouseOfTheDude/gemma-4-31B_PTQ shipped without
# chat_template.jinja (the pipeline stripped it). Newer runs preserve it.
# This block makes the script DTRT in both cases without manual intervention.
#
# To force the home-dir fallback, set CHAT_TEMPLATE explicitly:
#   CHAT_TEMPLATE=/home/phaedawg/chat_templates/tool_chat_template_gemma4.jinja \
#     ./gemma-4-31b-int8.sh API_KEY
# To bypass --chat-template entirely, set CHAT_TEMPLATE=''.
#
# One-time setup for the fallback template:
#   mkdir -p /home/phaedawg/chat_templates
#   wget -O /home/phaedawg/chat_templates/tool_chat_template_gemma4.jinja \
#     https://raw.githubusercontent.com/vllm-project/vllm/main/examples/tool_chat_template_gemma4.jinja
FALLBACK_CHAT_TEMPLATE="/home/phaedawg/chat_templates/tool_chat_template_gemma4.jinja"
MODEL_BUNDLED_CHAT_TEMPLATE="$MODEL_DIR/chat_template.jinja"

if [[ -z "${CHAT_TEMPLATE+x}" ]]; then
  if [[ -f "$MODEL_BUNDLED_CHAT_TEMPLATE" ]]; then
    CHAT_TEMPLATE="$MODEL_BUNDLED_CHAT_TEMPLATE"
    echo "CHAT_TEMPLATE: using model-bundled template at $CHAT_TEMPLATE"
  else
    CHAT_TEMPLATE="$FALLBACK_CHAT_TEMPLATE"
    echo "CHAT_TEMPLATE: model dir has no chat_template.jinja, falling back to $CHAT_TEMPLATE"
  fi
else
  echo "CHAT_TEMPLATE: caller-provided override = ${CHAT_TEMPLATE:-<empty / disabled>}"
fi

# Fail fast with a useful message if the chat template doesn't exist on disk,
# so you don't get the cryptic ChatTemplateResolutionError 30 seconds into
# model load.
if [[ -n "$CHAT_TEMPLATE" && ! -f "$CHAT_TEMPLATE" ]]; then
  echo "ERROR: CHAT_TEMPLATE points to a file that does not exist: $CHAT_TEMPLATE" >&2
  echo "       Either:" >&2
  echo "         (a) re-run the PTQ script so the model ships chat_template.jinja, or" >&2
  echo "         (b) download the upstream fallback:" >&2
  echo "             mkdir -p $(dirname "$FALLBACK_CHAT_TEMPLATE")" >&2
  echo "             wget -O $FALLBACK_CHAT_TEMPLATE \\" >&2
  echo "               https://raw.githubusercontent.com/vllm-project/vllm/main/examples/tool_chat_template_gemma4.jinja" >&2
  echo "         (c) set CHAT_TEMPLATE='' to attempt launch without one (vLLM will refuse on the first /v1/chat/completions request)." >&2
  exit 3
fi

# Multimodal limits. Gemma 4 31B is text + image (no audio on this size).
# Set IMAGE limit / AUDIO limit per request. Set TEXT_ONLY=1 to skip
# multimodal profiling entirely (frees the vision encoder reservation,
# letting you push MAX_MODEL_LEN higher).
TEXT_ONLY="${TEXT_ONLY:-0}"
LIMIT_MM_IMAGE="${LIMIT_MM_IMAGE:-4}"
LIMIT_MM_AUDIO="${LIMIT_MM_AUDIO:-0}"   # 31B has no audio encoder

# Per-image vision token budget. Supported values: 70, 140, 280 (default),
# 560, 1120. Higher = more visual detail but more compute and KV usage on
# the prefill of vision tokens. 280 is the model's documented default.
# Lower this to 140 if you want to fit more images per prompt.
VISION_TOKEN_BUDGET="${VISION_TOKEN_BUDGET:-280}"

# ============================================================
# MTP (Multi-Token Prediction) speculative decoding
# ============================================================
# Uses google/gemma-4-31B-it-assistant (a tiny 4-layer drafter, ~few hundred
# MB) to propose multiple tokens that the target verifies in one forward pass.
# The drafter shares the target's KV cache, so extra VRAM cost is minimal.
# Speedups of 1.5-3x on decode depending on workload.
#
# Requires vLLM nightly (post-PR #41745) or v0.22+.
# Set ENABLE_MTP=0 to disable if it causes issues on your vLLM version.
ENABLE_MTP="${ENABLE_MTP:-1}"
MTP_MODEL="${MTP_MODEL:-/media/fmodels/google/gemma-4-31B-it-assistant}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-4}"

# Prefix caching: large win for chat / agent workloads. Disable only when
# benchmarking (vLLM Gemma 4 recipe explicitly turns it off for benchmarks).
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"

# Async scheduling overlaps Python scheduling with GPU decode. Recommended
# in the vLLM Gemma 4 recipe for throughput. Cheap to leave on.
ENABLE_ASYNC_SCHEDULING="${ENABLE_ASYNC_SCHEDULING:-1}"

# CUDA graph setup:
# - full_decode_only keeps CUDA graphs on the decode path and avoids the
#   highest-memory full-prefill captures (essential on 24 GiB).
# - Gemma 4's heterogeneous per-layer attention backends (FA for sliding,
#   Triton for global) sometimes trip on aggressive cudagraph capture.
#   Capture sizes [1,2,4] are conservative.
# - If startup fails inside cudagraph capture, drop to MAX_CUDAGRAPH_CAPTURE_SIZE=1
#   or as a last resort enable ENFORCE_EAGER=1 below.
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-full_decode_only}"
CUDAGRAPH_CAPTURE_SIZES="${CUDAGRAPH_CAPTURE_SIZES:-[1,2,4]}"
if [[ -z "${COMPILATION_CONFIG:-}" ]]; then
  COMPILATION_CONFIG="{\"mode\":3,\"cudagraph_mode\":\"${CUDAGRAPH_MODE}\",\"cudagraph_capture_sizes\":${CUDAGRAPH_CAPTURE_SIZES}}"
fi

# Fallback for older vLLM versions that don't understand cudagraph_capture_sizes
# inside --compilation-config, OR for the per-layer-backend cudagraph asserts.
MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-}"

# Last-resort escape hatch if any of the above causes a startup-time crash.
# Leave 0 in production -- it disables CUDA graphs and torch.compile.
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
  --disable-custom-all-reduce
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

if [[ -n "$CHAT_TEMPLATE" ]]; then
  VLLM_ARGS+=(--chat-template "$CHAT_TEMPLATE")
fi

if [[ "$DEFAULT_THINKING" == "1" ]]; then
  VLLM_ARGS+=(--default-chat-template-kwargs '{"enable_thinking": true}')
fi

# Multimodal config. TEXT_ONLY=1 forces image=0,audio=0 which lets vLLM skip
# the vision encoder profiling pass entirely.
#
# NOTE: Current vLLM (per `vllm serve --help`) requires --limit-mm-per-prompt
# as JSON, NOT the legacy `image=N,audio=M` shorthand the older Gemma 4
# recipe shows. Older shorthand fails with:
#   argument --limit-mm-per-prompt: Value image=... cannot be converted to <function loads ...>
if [[ "$TEXT_ONLY" == "1" ]]; then
  VLLM_ARGS+=(--limit-mm-per-prompt '{"image": 0, "audio": 0}')
else
  VLLM_ARGS+=(--limit-mm-per-prompt "{\"image\": ${LIMIT_MM_IMAGE}, \"audio\": ${LIMIT_MM_AUDIO}}")
  if [[ -n "$VISION_TOKEN_BUDGET" ]]; then
    VLLM_ARGS+=(--mm-processor-kwargs "{\"max_soft_tokens\": ${VISION_TOKEN_BUDGET}}")
  fi
fi

if [[ "$ENABLE_PREFIX_CACHING" == "1" ]]; then
  VLLM_ARGS+=(--enable-prefix-caching)
fi

if [[ "$ENABLE_ASYNC_SCHEDULING" == "1" ]]; then
  VLLM_ARGS+=(--async-scheduling)
fi

if [[ "$ENABLE_MTP" == "1" ]]; then
  VLLM_ARGS+=(--speculative-config "{\"method\":\"mtp\",\"model\":\"${MTP_MODEL}\",\"num_speculative_tokens\":${NUM_SPECULATIVE_TOKENS}}")
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
echo "  REASONING_PARSER=${REASONING_PARSER}  DEFAULT_THINKING=${DEFAULT_THINKING}"
echo "  TOOL_CALLING=${ENABLE_TOOL_CALLING} (parser=${TOOL_CALL_PARSER})"
echo "  CHAT_TEMPLATE=${CHAT_TEMPLATE:-<model default>}"
if [[ "$TEXT_ONLY" == "1" ]]; then
  echo "  TEXT_ONLY=1 (multimodal disabled)"
else
  echo "  MM image=${LIMIT_MM_IMAGE} audio=${LIMIT_MM_AUDIO} budget=${VISION_TOKEN_BUDGET}"
fi
echo "  PREFIX_CACHING=${ENABLE_PREFIX_CACHING}  ASYNC_SCHEDULING=${ENABLE_ASYNC_SCHEDULING}"
if [[ "$ENABLE_MTP" == "1" ]]; then
  echo "  MTP=ON (drafter=${MTP_MODEL}, num_speculative_tokens=${NUM_SPECULATIVE_TOKENS})"
else
  echo "  MTP=OFF"
fi
echo "  ENFORCE_EAGER=${ENFORCE_EAGER}  COMPILATION_CONFIG=${COMPILATION_CONFIG}"
echo
echo "Suggested sampling per Google's Gemma 4 model card:"
echo "  temperature=1.0, top_p=0.95, top_k=64"
echo

vllm serve "${VLLM_ARGS[@]}"
