"""
Qwen3.5 W8A16 PTQ — Text-Only Fine-Tunes (Post-Training Quantization)

  - For text-only Qwen3.5 fine-tunes (e.g. LoRA/SFT checkpoints that only
    contain language-model weights, no vision tower).
  - Uses AutoModelForCausalLM (weights saved as model.layers.*).
  - Serve in vLLM with: vllm serve <path> --task generate
  - No calibration dataset needed.
  - W8A16 preset: INT8 per-channel symmetric weights, BF16 activations.

  For full VLM checkpoints (model_type=qwen3_5), use qwen3_5-W8A16_PTQ.py instead.
"""
import argparse

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

# =========================
# Parse Command-Line Arguments
# =========================
parser = argparse.ArgumentParser(
    description="Run W8A16 PTQ on text-only Qwen3.5 fine-tunes. No calibration data needed."
)
parser.add_argument("model_path", type=str, help="Path to the source model directory.")
parser.add_argument("output_path", type=str, help="Path to save quantized model.")

args = parser.parse_args()

# =========================
# Model
# =========================
MODEL_ID = args.model_path

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
print(f"Loaded model: {MODEL_ID}")

# =========================
# Quantization recipe — text-only (no vision ignores needed)
# =========================
recipe = QuantizationModifier(
    targets="Linear",
    scheme="W8A16",
    ignore=[
        "lm_head",
        "re:.*linear_attn.*",
        "re:.*mtp.*",
    ],
)

# =========================
# Apply quantization (no calibration data needed for W8A16 PTQ)
# =========================
print("\n=== Running W8A16 PTQ ===")
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
