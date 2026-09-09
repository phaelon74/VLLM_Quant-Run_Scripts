"""
Qwen3.5 BF16-attention + W8A16 PTQ (Post-Training Quantization)

  - QuantizationModifier with preset scheme (no AWQ, no GPTQ)
  - AutoModelForImageTextToText to preserve VLM weight paths for vLLM
  - No calibration dataset needed
  - W8A16 on MLP and other Linears; self_attn q/k/v/o_proj stay BF16 (stability / KLD)
  - Weights loaded in bfloat16; validate mixed layout in vLLM before production

  For full Linear W8A16 including attention, use qwen3_5-W8A16_PTQ.py instead.
"""
import argparse

import torch
import torch.nn as nn
from compressed_tensors.offload import dispatch_model
from transformers import AutoModelForImageTextToText, AutoTokenizer

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
        "Run W8A16 PTQ on Qwen3.5 with self_attn q/k/v/o_proj kept in BF16. "
        "No calibration data needed."
    )
)
parser.add_argument("model_path", type=str, help="Path to the source model directory.")
parser.add_argument("output_path", type=str, help="Path to save quantized model.")

args = parser.parse_args()

# =========================
# Model
# =========================
MODEL_ID = args.model_path

# AutoModelForImageTextToText saves weights with language_model.layers.* paths
# that vLLM's Qwen3.5 implementation expects. AutoModelForCausalLM saves with
# model.layers.* which causes weight loading failures in vLLM.
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
print(f"Loaded model: {MODEL_ID}")

# =========================
# Quantization recipe — W8A16 + BF16 self-attention projections
# =========================
# W8A16 preset: INT8 per-channel symmetric weights, activations untouched.
# Leave q/k/v/o_proj in BF16 (same idea as Gemma4-W8A16_GPTQ --bf16-attention).
recipe = QuantizationModifier(
    targets="Linear",
    scheme="W8A16",
    ignore=[
        "lm_head",
        "re:.*visual.*",
        "re:.*linear_attn.*",
        "re:.*mtp.*",
        r"re:.*self_attn\.(q|k|v|o)_proj$",
    ],
)

# =========================
# Apply quantization (no calibration data needed for W8A16 PTQ)
# =========================
print("\n=== Running W8A16 PTQ (BF16 self_attn q/k/v/o_proj) ===")
oneshot(model=model, recipe=recipe)

# =========================
# Quick sanity generation
# =========================
print("\n\n========== SAMPLE GENERATION ==============")
dispatch_model(model)
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(model.device)
output = model.generate(input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0], skip_special_tokens=True))
print("==========================================\n\n")

# =========================
# Save compressed model
# =========================
SAVE_DIR = args.output_path
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

print("\n=== Complete ===")
print("Saved to:", SAVE_DIR)
