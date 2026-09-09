"""
Command A+ (CohereLabs/command-a-plus-05-2026-bf16) W4A16 PTQ

  - QuantizationModifier with W4A16 preset (RTN, no AWQ/GPTQ calibration data)
  - AutoModelForImageTextToText preserves VLM paths for vLLM (Cohere2Vision)
  - Unfuses fused MoE expert Parameters into per-expert Linears before PTQ
  - No calibration dataset required

Model: 218B total / ~25B active sparse MoE (cohere2_moe + SigLIP vision)
  - 32 decoder layers, 128 routed experts x 8/tok, 4 shared experts
  - Hybrid sliding-window / full attention (3:1)
  - 128K context, image+text inputs

Requires:
  pip install -U transformers llmcompressor compressed-tensors accelerate
  transformers >= 5.8 with cohere2_vision / cohere2_moe support

vLLM serve (after quant):
  vllm serve <output_dir> -tp <N> \\
    --tool-call-parser cohere_command4 \\
    --reasoning-parser cohere_command4 \\
    --enable-auto-tool-choice

Reference: https://huggingface.co/CohereLabs/command-a-plus-05-2026-bf16
"""
import argparse
import os
import shutil

import torch.nn as nn
from compressed_tensors.offload import dispatch_model
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    GenerationConfig,
)

from cohere2_moe_calibration import unfuse_cohere2_moe_experts

# Transformers v5 compatibility
import transformers.modeling_utils as _tmu

if not hasattr(_tmu, "TORCH_INIT_FUNCTIONS"):
    _tmu.TORCH_INIT_FUNCTIONS = {
        "uniform_": nn.init.uniform_,
        "normal_": nn.init.normal_,
        "trunc_normal_": nn.init.trunc_normal_,
        "constant_": nn.init.constant_,
        "xavier_uniform_": nn.init.xavier_uniform_,
        "xavier_normal_": nn.init.xavier_normal_,
        "kaiming_uniform_": nn.init.kaiming_uniform_,
        "kaiming_normal_": nn.init.kaiming_normal_,
        "uniform": nn.init.uniform,
        "normal": nn.init.normal,
        "xavier_uniform": nn.init.xavier_uniform,
        "xavier_normal": nn.init.xavier_normal,
        "kaiming_uniform": nn.init.kaiming_uniform,
        "kaiming_normal": nn.init.kaiming_normal,
    }

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# =========================
# Parse Command-Line Arguments
# =========================
parser = argparse.ArgumentParser(
    description=(
        "Run W4A16 RTN PTQ on Cohere Command A+ (command-a-plus-05-2026-bf16). "
        "No calibration data needed."
    )
)
parser.add_argument("model_path", type=str, help="Path to the source model directory.")
parser.add_argument("output_path", type=str, help="Path to save quantized model.")
parser.add_argument(
    "--skip-sample-gen",
    action="store_true",
    help="Skip the post-quantization text generation smoke test.",
)

args = parser.parse_args()

# =========================
# Model
# =========================
MODEL_ID = args.model_path

# Cohere2VisionForConditionalGeneration keeps model.language_model.layers.* and
# model.vision_tower / model.multi_modal_projector paths that vLLM expects.
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, dtype="auto", trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
print(f"Loaded model: {MODEL_ID}")
print(f"Model class: {type(model).__name__}")

# MoE: merge on-disk per-expert Linears into fused Parameters at load time.
# Unfuse back to nn.Linear modules so RTN W4A16 can quantize expert weights.
num_moe = unfuse_cohere2_moe_experts(model, calibrate_all_experts=False)
if num_moe == 0:
    print(
        "WARNING: no Cohere2MoeSparseMoeBlock modules found. "
        "Expert weights may not be quantized."
    )

# ----------------------------------------------------------------------
# Preserve source generation_config.json (thinking / stop tokens, etc.)
# ----------------------------------------------------------------------
SRC_GEN_CONFIG_PATH = os.path.join(MODEL_ID, "generation_config.json")
try:
    source_generation_config = GenerationConfig.from_pretrained(
        MODEL_ID, trust_remote_code=True
    )
    print(
        f"Captured source GenerationConfig: "
        f"eos_token_id={source_generation_config.eos_token_id}, "
        f"bos_token_id={source_generation_config.bos_token_id}, "
        f"pad_token_id={source_generation_config.pad_token_id}"
    )
except Exception as e:
    source_generation_config = None
    print(f"WARNING: could not load source GenerationConfig from {MODEL_ID}: {e}")

# =========================
# Quantization recipe — Command A+ W4A16 RTN
# =========================
# W4A16: 4-bit group-128 symmetric weights, BF16 activations.
#
# Ignore rationale (aligned with Cohere W4A4 release + Gemma VLM pattern):
#   - lm_head: tied to embed_tokens; quantizing hurts quality
#   - vision_tower / multi_modal_projector: keep vision path in BF16
#   - mlp.gate: MoE router (Cohere2MoeTopKRouter Parameter, not Linear)
recipe = QuantizationModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=[
        "lm_head",
        "re:.*vision_tower.*",
        "re:.*multi_modal_projector.*",
        "re:.*mlp\\.gate$",
    ],
)

# =========================
# Apply quantization (no calibration data — RTN)
# =========================
print("\n=== Running W4A16 RTN PTQ on Command A+ ===")
oneshot(model=model, recipe=recipe)

# =========================
# Quick sanity generation (text-only)
# =========================
if not args.skip_sample_gen:
    print("\n\n========== SAMPLE GENERATION ==============")
    dispatch_model(model)

    SAMPLE_PROMPT = "Hello my name is"
    messages = [{"role": "user", "content": SAMPLE_PROMPT}]

    _tok = getattr(processor, "tokenizer", processor)
    has_chat_tmpl = getattr(_tok, "chat_template", None) is not None

    if has_chat_tmpl:
        prompt_text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt_text = SAMPLE_PROMPT

    inputs = processor(text=[prompt_text], return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items() if hasattr(v, "to")}
    input_len = inputs["input_ids"].shape[-1]

    output = model.generate(**inputs, max_new_tokens=100)
    print(processor.decode(output[0][input_len:], skip_special_tokens=True))
    print("==========================================\n\n")

# =========================
# Save compressed model
# =========================
SAVE_DIR = args.output_path

if source_generation_config is not None:
    model.generation_config = source_generation_config
    print(
        f"Restored model.generation_config.eos_token_id = "
        f"{model.generation_config.eos_token_id} prior to save."
    )

model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)

dst_gen_config_path = os.path.join(SAVE_DIR, "generation_config.json")
if os.path.isfile(SRC_GEN_CONFIG_PATH):
    shutil.copy2(SRC_GEN_CONFIG_PATH, dst_gen_config_path)
    print(f"Copied source generation_config.json -> {dst_gen_config_path}")
else:
    print(
        f"WARNING: source generation_config.json not found at {SRC_GEN_CONFIG_PATH}; "
        f"verify stop/thinking token settings manually."
    )

print("\n=== Complete ===")
print("Saved to:", SAVE_DIR)
print("\nLoad in vLLM (example):")
print(
    f"  vllm serve {SAVE_DIR} -tp <num_gpus> "
    "--tool-call-parser cohere_command4 "
    "--reasoning-parser cohere_command4 "
    "--enable-auto-tool-choice"
)
