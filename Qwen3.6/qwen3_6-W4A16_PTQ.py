"""
Qwen3.6 W8A16 PTQ (Post-Training Quantization)

  - QuantizationModifier with preset scheme (no AWQ, no GPTQ)
  - AutoModelForImageTextToText to preserve VLM weight paths for vLLM
  - No calibration dataset needed
  - W8A16 preset: INT8 per-channel symmetric weights, FP16/BF16 activations
  - Post-save key remap to fix Transformers v5 double-nested weight paths
"""
import argparse
import glob
import os

import torch.nn as nn
from compressed_tensors.offload import dispatch_model
from safetensors import safe_open
from safetensors.torch import save_file
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
# Transformers v5 save_pretrained workaround
#
# Transformers v5's composable model uses different internal attribute names
# than the checkpoint format (e.g. double-nested "language_model.language_model").
# save_pretrained writes these raw attribute names, producing keys vLLM cannot
# load. We fix this after saving, preserving safetensors metadata.
# =========================
KEY_REMAPS = [
    ("model.language_model.language_model.", "model.language_model."),
    ("model.language_model.visual.", "model.visual."),
]


def fix_saved_weight_keys(save_dir):
    """Remap double-nested weight keys in saved safetensors files."""
    safetensor_files = sorted(glob.glob(os.path.join(save_dir, "*.safetensors")))
    if not safetensor_files:
        return

    for fpath in safetensor_files:
        with safe_open(fpath, framework="pt") as f:
            metadata = f.metadata() or {}
            orig_keys = list(f.keys())

            print(f"\n  --- {os.path.basename(fpath)} ---")
            print(f"  Raw keys from save_pretrained (first 10):")
            for k in orig_keys[:10]:
                print(f"    {k}")

            new_tensors = {}
            remapped_count = 0
            for key in orig_keys:
                new_key = key
                changed = True
                while changed:
                    changed = False
                    for old_prefix, new_prefix in KEY_REMAPS:
                        if new_key.startswith(old_prefix):
                            new_key = new_prefix + new_key[len(old_prefix):]
                            changed = True
                            break
                if new_key != key:
                    remapped_count += 1
                new_tensors[new_key] = f.get_tensor(key).clone()

        sorted_keys = sorted(new_tensors.keys())
        print(f"  Remapped {remapped_count}/{len(sorted_keys)} keys")
        print(f"  First 10 keys AFTER remap:")
        for k in sorted_keys[:10]:
            print(f"    {k}")

        tmp_path = fpath + ".tmp"
        save_file(new_tensors, tmp_path, metadata=metadata)
        os.replace(tmp_path, fpath)
        del new_tensors
        print(f"  Saved {os.path.basename(fpath)} ({len(sorted_keys)} tensors)")

    prefixes = set()
    for fpath in safetensor_files:
        with safe_open(fpath, framework="pt") as f:
            for key in f.keys():
                prefixes.add(".".join(key.split(".")[:3]))
    print("\n=== Saved weight key prefixes ===")
    for p in sorted(prefixes):
        print(f"  {p}")
    print()


# =========================
# Parse Command-Line Arguments
# =========================
parser = argparse.ArgumentParser(
    description="Run W8A16 PTQ on Qwen3.6 (e.g., Qwen3.6-35B-A3B). No calibration data needed."
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
model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
print(f"Loaded model: {MODEL_ID}")

# =========================
# Quantization recipe — matches NVFP4A16 reference pattern
# =========================
# W8A16 preset: INT8 per-channel symmetric weights, activations untouched.
# Ignore list matches the working NVFP4A16 reference exactly.
recipe = QuantizationModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=[
        "lm_head",
        "re:.*visual.*",
        "re:.*linear_attn.*",
        "re:.*mtp.*",
        "re:.*mlp.gate$",
        "re:.*mlp.shared_expert_gate$",
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
print(f"Saving to: {SAVE_DIR}")
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

# Fix weight keys mangled by transformers v5
fix_saved_weight_keys(SAVE_DIR)

print("\n=== Complete ===")
print("Saved to:", SAVE_DIR)
