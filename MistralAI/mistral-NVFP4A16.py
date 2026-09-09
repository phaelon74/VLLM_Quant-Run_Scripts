"""
NVFP4A16 Quantization for Mistral-Large-Instruct-2411 and fine-tunes (e.g. Behemoth-R1-123B-v2).

Targets: TheDrummer/Behemoth-R1-123B-v2 (Mistral-Large-Instruct-2411 fine-tune).
NVFP4A16: FP4 weights per-group-16, FP16/BF16 activations. No calibration dataset required.
Optimized for NVIDIA Blackwell GPUs (SM 9.0+). Uses sequential onloading for 123B models.

Usage:
    python MistralAI/mistral-NVFP4A16.py <model_path> <output_path>
"""

import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# =========================
# Parse CLI Arguments
# =========================
parser = argparse.ArgumentParser(
    description="NVFP4A16 quantization for Mistral-Large-Instruct-2411 and fine-tunes (e.g. Behemoth-R1-123B-v2)."
)
parser.add_argument(
    "model_path",
    type=str,
    help="Source model: Hugging Face repo ID or local path.",
)
parser.add_argument(
    "output_path",
    type=str,
    help="Destination directory for quantized model.",
)
args = parser.parse_args()

model_path = args.model_path
output_path = args.output_path

# =========================
# Load Model and Tokenizer
# =========================
print(f"Loading model from: {model_path}")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map=None,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# =========================
# Quantization Recipe
# =========================
# NVFP4A16: FP4 weights per-group-16, activations stay FP16/BF16.
# Mistral-Large has no visual encoder, linear_attn, or MTP -- only lm_head is skipped.
recipe = QuantizationModifier(
    targets="Linear",
    scheme="NVFP4A16",
    ignore=["lm_head"],
)

# =========================
# Apply Quantization (Sequential Onloading for 123B)
# =========================
print("\nStarting NVFP4A16 quantization (sequential onloading)...")
oneshot(
    model=model,
    recipe=recipe,
    sequential_targets=["MistralDecoderLayer"],
)

# =========================
# Save Compressed Model
# =========================
model.save_pretrained(output_path, save_compressed=True)
tokenizer.save_pretrained(output_path)

print(f"\nQuantization complete. Saved to: {output_path}")
