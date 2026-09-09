"""
Magistral-2509-24B-Text-Only W8A16 PTQ (Post-Training Quantization)

  - Target: Darkhn/Magistral-2509-24B-Text-Only (MistralForCausalLM, text-only)
  - QuantizationModifier with preset scheme (no AWQ, no GPTQ)
  - No calibration dataset needed
  - W8A16 preset: INT8 per-channel symmetric weights, BF16 activations
  - Sequential onloading via MistralDecoderLayer for 24B memory efficiency

  Model: 40-layer dense Mistral decoder (5120 hidden, 32/8 GQA heads, 128k ctx).
  Derived from Magistral Small 1.2 with vision encoder removed.

  Requires: pip install -U transformers llmcompressor compressed-tensors accelerate

Usage:
    python MistralAI/magistral-2509-24B-W8A16_PTQ.py <model_path> <output_path>
"""
import argparse
import os
import shutil

import torch.nn as nn
from compressed_tensors.offload import dispatch_model
from transformers import AutoModelForCausalLM, AutoTokenizer

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


def copy_tokenizer_artifacts(source_dir: str, save_dir: str) -> None:
    """Preserve Mistral tekken tokenizer and chat template files for vLLM/llama.cpp."""
    for filename in ("tekken.json", "chat_template.jinja", "params.json"):
        src = os.path.join(source_dir, filename)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(save_dir, filename))
            print(f"Copied {filename} -> {save_dir}")


# =========================
# Parse Command-Line Arguments
# =========================
parser = argparse.ArgumentParser(
    description=(
        "Run W8A16 PTQ on Magistral-2509-24B-Text-Only "
        "(Darkhn/Magistral-2509-24B-Text-Only). No calibration data needed."
    )
)
parser.add_argument("model_path", type=str, help="Path to the source model directory.")
parser.add_argument("output_path", type=str, help="Path to save quantized model.")

args = parser.parse_args()

# =========================
# Model
# =========================
MODEL_ID = args.model_path

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype="auto",
    device_map=None,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
print(f"Loaded model: {MODEL_ID}")

# =========================
# Quantization recipe
# =========================
# W8A16 preset: INT8 per-channel symmetric weights, activations untouched.
# Text-only dense Mistral — only skip lm_head (matches mistral-NVFP4A16.py).
recipe = QuantizationModifier(
    targets="Linear",
    scheme="W8A16",
    ignore=["lm_head"],
)

# =========================
# Apply quantization (no calibration data needed for W8A16 PTQ)
# =========================
print("\n=== Running W8A16 PTQ (sequential onloading) ===")
oneshot(
    model=model,
    recipe=recipe,
    sequential_targets=["MistralDecoderLayer"],
)

# =========================
# Quick sanity generation
# =========================
print("\n\n========== SAMPLE GENERATION ==============")
dispatch_model(model)

SAMPLE_PROMPT = "Hello my name is"
messages = [{"role": "user", "content": SAMPLE_PROMPT}]

if getattr(tokenizer, "chat_template", None):
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
else:
    prompt_text = SAMPLE_PROMPT

inputs = tokenizer(prompt_text, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}
input_len = inputs["input_ids"].shape[-1]

output = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(output[0][input_len:], skip_special_tokens=True))
print("==========================================\n\n")

# =========================
# Save compressed model
# =========================
SAVE_DIR = args.output_path
print(f"Saving to: {SAVE_DIR}")
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

copy_tokenizer_artifacts(MODEL_ID, SAVE_DIR)

print("\n=== Complete ===")
print("Saved to:", SAVE_DIR)
