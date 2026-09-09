#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# TheHouseOfTheDude/Qwen3.6-35B-A3B-FP6 on 1x RTX PRO 6000 (Blackwell) via vLLM
# ============================================================
#
# Model:
#   - Base:    Qwen/Qwen3.6-35B-A3B  (arch tag: qwen3_5 MoE family, ~35B total
#              / ~3B active params)
#   - Quant:   SparkInfer MX-FP6 (W6A6 schema tag): 6-bit weights on disk,
#              activations quantized at runtime to FP8 E4M3 (W6A8 execution).
#              quantization_config = {"quant_method":"modelopt","quant_algo":"W6A6"}
#              Coverage: all 256 routed experts per layer (gate/up/down), the
#              shared expert, MLP + full attention + linear_attn projections.
#              IGNORED / kept BF16: lm_head, visual tower, mtp head, embeddings,
#              router gates (mlp.gate / shared_expert_gate), norms, GDN aux.
#              Routed experts run through the SparkInfer fused MoE kernel
#              (FusedMoE binding); everything else uses the dense FP6 path.
#
# Architecture notes:
#   - 40-layer hybrid stack: 30 Gated DeltaNet (linear_attn) layers + 10 full
#     Gated Attention layers. KV cache only for the 10 full-attention layers.
#   - MoE: 256 routed experts, top-8, + 1 shared expert per layer.
#     Expert dims 2048 (hidden) -> 512 (intermediate).
#   - Vision encoder present (VL model); MTP head trained-in (kept BF16).
#
# ------------------------------------------------------------
# PREREQUISITE -- register the SparkInfer FP6 quantization in vLLM
# ------------------------------------------------------------
#   "sparkinfer_fp6" is NOT a stock vLLM quant method; the sparkinfer package
#   (sparkinfer.integration.vllm.plugin, wired via the vllm.general_plugins
#   entry point in sparkinfer's pyproject) must be installed in the serving
#   venv. The env gates:
#     SPARKINFER_ENABLE_FP6=1        -> select the FP6 path (else native fallback)
#     SPARKINFER_ENABLE_FP6_MICRO=0  -> dense linears use the fused kernel. The
#                                       MoE expert path picks its own backend
#                                       per shape.
#
# VRAM fit on 1x 96 GiB Blackwell:
#   - FP6 checkpoint is ~28.8 GiB on disk; plus BF16 residue (visual/lm_head/
#     embeddings/mtp/router) resident ~= 32 GiB.
#   - KV cache only for the 10 Gated Attention layers -> 128K context is cheap
#     relative to the dense 27B build.
#
# This script does not store API keys. Pass the key as the first argument, or
# set VLLM_API_KEY in the environment.
# ============================================================

# --- Memory Management ---
export PYTORCH_ALLOC_CONF=expandable_segments:True

# --- GPU Selection (single Blackwell card) ---
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# --- SparkInfer FP6 gates (see PREREQUISITE above) ---
export SPARKINFER_ENABLE_FP6="${SPARKINFER_ENABLE_FP6:-1}"
export SPARKINFER_ENABLE_FP6_MICRO="${SPARKINFER_ENABLE_FP6_MICRO:-0}"
# Make sure KLD-scoring-only vars never leak into serving:
unset TORCH_COMPILE_DISABLE SPARKINFER_DYNAMIC_DETERMINISTIC_OUTPUT 2>/dev/null || true

# --- vLLM / CUDA behavior ---
export VLLM_NO_USAGE_STATS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_SLEEP_WHEN_IDLE=1
export VLLM_USE_FLASHINFER_SAMPLER=0

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

# Local SparkInfer MX-FP6 (W6A6) checkpoint. Override at launch time if needed:
#   MODEL_DIR=/path/to/model ./qwen3.6-35b-a3b-fp6.sh
MODEL_DIR="${MODEL_DIR:-/media/fmodels/TheHouseOfTheDude/qwen3-6_35B-A3B_moe_fp6}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.6-35B-A3B-FP6-W6A6}"
# Tell the FP6 vLLM plugin where the weights live (inherited by workers).
# Set UNCONDITIONALLY: a stale SPARKINFER_FP6_MODEL_DIR from a previous launch
# makes the plugin index the wrong checkpoint -> wrong FP6/BF16 classification
# -> loader shape asserts / silent BF16 fallback (the Cydonia lesson; also hit
# the 35B MoE KLD run when the 27B dir was still exported).
export SPARKINFER_FP6_MODEL_DIR="$MODEL_DIR"
unset B12X_FP6_MODEL_DIR 2>/dev/null || true  # legacy name must not shadow it
# The FP6 export copies tokenizer.json/tokenizer_config.json and
# chat_template.jinja, so keep tokenizer resolution local by default.
TOKENIZER="${TOKENIZER:-$MODEL_DIR}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"

# Default 1 GPU. MoE experts shard through vLLM's loader, so TP=2 works;
# the script adds --disable-custom-all-reduce automatically at TP>1.
TP_SIZE="${TP_SIZE:-1}"

# ------------------------------------------------------------
# Capacity sizing for 1x 96 GiB Blackwell at full VLM (vision enabled).
# Only 10 of 40 layers carry KV cache, so long context is cheap here.
# ------------------------------------------------------------
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"

KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"

TOKENIZER_MODE="${TOKENIZER_MODE:-hf}"
CONFIG_FORMAT="${CONFIG_FORMAT:-auto}"
LOAD_FORMAT="${LOAD_FORMAT:-auto}"

# SparkInfer MX-FP6 (W6A6). Set QUANTIZATION="" to let vLLM auto-detect from
# config.json (works because the plugin overrides the modelopt method).
QUANTIZATION="${QUANTIZATION:-sparkinfer_fp6}"

# Qwen3.6 ships a custom modeling file (qwen3_5 / qwen3_next family).
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"

# ============================================================
# Qwen3.6 reasoning / tool-call / MTP / VLM toggles
# ============================================================

# Qwen3.6 thinks by default; qwen3 reasoning parser covers the whole family.
REASONING_PARSER="${REASONING_PARSER:-qwen3}"

# Tool calling: qwen3_coder is the model-card parser for OpenAI-style tool use.
ENABLE_TOOL_CALLING="${ENABLE_TOOL_CALLING:-1}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"

# Multi-Token Prediction (trained-in MTP head, kept BF16 by the FP6 export —
# including its packed MoE experts). Disable (ENABLE_MTP=0) for non-spec
# benchmarking or if MTP draft capture asserts.
ENABLE_MTP="${ENABLE_MTP:-1}"
# NOTE: JSON defaults must NOT live inside ${VAR:-...} — bash closes the
# expansion at the FIRST '}' in the default, appending a stray literal '}'
# to any env-provided value (breaks vllm's JSON parsing).
if [[ -z "${MTP_SPEC:-}" ]]; then
  MTP_SPEC='{"method":"qwen3_next_mtp","num_speculative_tokens":2}'
fi

# Vision. VL model; the FP6 export left the visual tower in BF16 so it works
# out of the box. Set TEXT_ONLY=1 to skip vision profiling and free its
# workspace for additional KV cache (text-only traffic).
TEXT_ONLY="${TEXT_ONLY:-0}"

# Optional: lets clients drive video frame sampling via
# extra_body={"mm_processor_kwargs": {"fps": ...}}. Harmless when no video.
if [[ -z "${MEDIA_IO_KWARGS:-}" ]]; then
  MEDIA_IO_KWARGS='{"video":{"num_frames":-1}}'
fi

# Prefix caching is generally a win for chat/agent workloads.
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"

# CUDA graph setup. The FP6 MoE binding warm-runs the resolved capture sizes
# at weight-load time automatically (it reads vLLM's compilation config;
# SPARKINFER_MOE_WARM_MS overrides) — no manual sync with the plugin is needed.
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-full_decode_only}"
# Default capture sizes are derived from the MTP spec: vLLM pads decode
# batches to the next captured size, so the list must reach at least the
# MTP verify batch (1 + num_speculative_tokens) and ideally contain it
# exactly (padding 5 -> 8 wastes ~37% of every verify GEMM). Users only
# set MTP_SPEC; this stays in sync automatically. The FP6 plugin warms
# whatever sizes vLLM resolves, so no manual sync is needed there either.
if [[ -z "${CUDAGRAPH_CAPTURE_SIZES:-}" ]]; then
  NSPEC=0
  if [[ "$ENABLE_MTP" == "1" && "$MTP_SPEC" =~ \"num_speculative_tokens\"[[:space:]]*:[[:space:]]*([0-9]+) ]]; then
    NSPEC="${BASH_REMATCH[1]}"
  fi
  VERIFY=$((NSPEC + 1))
  CUDAGRAPH_CAPTURE_SIZES="[$(printf '%s\n' 1 2 4 8 "$VERIFY" | sort -un | paste -sd, -)]"
fi
if [[ -z "${COMPILATION_CONFIG:-}" ]]; then
  COMPILATION_CONFIG="{\"mode\":3,\"cudagraph_mode\":\"${CUDAGRAPH_MODE}\",\"cudagraph_capture_sizes\":${CUDAGRAPH_CAPTURE_SIZES}}"
fi

# Fallback for older vLLM versions, OR the DeltaNet cudagraph cache assert.
MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-}"

# Eager mode escape hatch for first bring-up of the MoE binding: if mode-3
# compile or graph capture trips on the FusedMoE path, relaunch with
# ENFORCE_EAGER=1 to isolate kernel correctness from compile/capture issues.
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

# Eager mode disables torch.compile + CUDA graphs, so the compilation-config is
# both unnecessary and conflicting there; only pass it when compiling.
if [[ "$ENFORCE_EAGER" == "1" ]]; then
  VLLM_ARGS+=(--enforce-eager)
else
  VLLM_ARGS+=(--compilation-config "$COMPILATION_CONFIG")
fi

# Blackwell sm_120: vLLM's custom all-reduce kernel crashes during CUDA graph
# capture at TP>1 (custom_all_reduce.cuh 'invalid argument'). Force the NCCL
# fallback (~1-3% slower at TP=2) whenever tensor parallelism is active.
if [[ "$TP_SIZE" -gt 1 ]]; then
  VLLM_ARGS+=(--disable-custom-all-reduce)
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

echo "Launching vLLM with:"
echo "  MODEL_DIR=${MODEL_DIR}"
echo "  SERVED_MODEL_NAME=${SERVED_MODEL_NAME}"
echo "  TOKENIZER=${TOKENIZER}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  SPARKINFER_ENABLE_FP6=${SPARKINFER_ENABLE_FP6} (micro=${SPARKINFER_ENABLE_FP6_MICRO})"
echo "  SPARKINFER_FP6_MODEL_DIR=${SPARKINFER_FP6_MODEL_DIR}"
echo "  TP_SIZE=${TP_SIZE}"
echo "  MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "  KV_CACHE_DTYPE=${KV_CACHE_DTYPE}"
echo "  QUANTIZATION=${QUANTIZATION}"
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
