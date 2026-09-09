"""
GLM-4.7-Flash NVFP4A16 Quantization Script

NVFP4A16: FP4 weights per-group-16, FP16/BF16 activations. PTQ (no calibration dataset).
Exposes all routed experts via CalibrationGlm4MoeLiteMoE for full MoE coverage.

Usage:
    python GLM4_7-Flash_NVFP4A16.py <model_path> <output_path>
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

from glm4_moe_lite_v2 import CalibrationGlm4MoeLiteMoE  # noqa: F401

# =========================
# Monkey-patch: fix fused global scales for MLA attention
# =========================
# NVFP4 uses update_fused_layer_weight_global_scales() which assumes q_proj/k_proj/v_proj.
# GLM-4.7-Flash uses MLA: q_a_proj, q_b_proj, kv_a_proj_with_mqa, kv_b_proj.
import llmcompressor.modifiers.utils.helpers as _awq_helpers
import llmcompressor.modifiers.awq.base as _awq_base

def _safe_update_fused_layer_weight_global_scales(submodule):
    """Skip MLA attention modules lacking q_proj/k_proj/v_proj."""
    has_standard_qkv = (
        hasattr(submodule, 'q_proj')
        and hasattr(submodule, 'k_proj')
        and hasattr(submodule, 'v_proj')
    )
    has_fused_qkv = hasattr(submodule, 'qkv_proj')
    if not has_standard_qkv and not has_fused_qkv:
        return
    try:
        _orig_update_fused(submodule)
    except AttributeError:
        pass

_orig_update_fused = _awq_helpers.update_fused_layer_weight_global_scales
_awq_helpers.update_fused_layer_weight_global_scales = _safe_update_fused_layer_weight_global_scales
_awq_base.update_fused_layer_weight_global_scales = _safe_update_fused_layer_weight_global_scales

# =========================
# Parse Command-Line Arguments
# =========================
parser = argparse.ArgumentParser(
    description="Run NVFP4A16 PTQ on GLM-4.7-Flash. No calibration data needed."
)
parser.add_argument("model_path", type=str, help="Path to the source model directory.")
parser.add_argument("output_path", type=str, help="Path to save quantized model.")

args = parser.parse_args()

# =========================
# MoE module replacement (expose routed experts for quantization)
# =========================
def replace_moe_modules_for_calibration(model):
    """Replace Glm4MoeLiteMoE with CalibrationGlm4MoeLiteMoE so routed experts are exposed as Linear modules."""
    replaced_count = 0
    for name, module in list(model.named_modules()):
        if type(module).__name__ == "Glm4MoeLiteMoE":
            parts = name.split(".")
            attr_name = parts[-1]
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            calibration_module = CalibrationGlm4MoeLiteMoE(
                original=module,
                config=model.config,
                calibrate_all_experts=True,
            )
            setattr(parent, attr_name, calibration_module)
            replaced_count += 1
    print(f"Replaced {replaced_count} MoE modules with calibration versions (all experts exposed)")
    return model

# =========================
# Load model
# =========================
MODEL_ID = args.model_path
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype="auto", trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
print(f"Loaded model: {MODEL_ID}")

# Replace MoE modules so routed experts are exposed as Linear layers for quantization
model = replace_moe_modules_for_calibration(model)

# Configure the quantization algorithm and scheme.
# NVFP4A16: FP4 weights per-group-16, FP16/BF16 activations. PTQ.
# Skip: lm_head, dense layer 0, router gates. shared_experts and routed experts are quantized.
recipe = QuantizationModifier(
    targets="Linear",
    scheme="NVFP4A16",
    ignore=[
        "lm_head",
        # Layer 0: Dense layer - keep in higher precision
        "model.layers.0.self_attn.q_a_proj",
        "model.layers.0.self_attn.q_b_proj",
        "model.layers.0.self_attn.kv_a_proj_with_mqa",
        "model.layers.0.self_attn.kv_b_proj",
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.mlp.up_proj",
        "model.layers.0.mlp.down_proj",
        # Router gates - MoE routing sensitive to quantization
        "re:.*mlp\\.gate$",
    ],
)

# Apply quantization (sequential decoder layers for memory efficiency)
oneshot(
    model=model,
    recipe=recipe,
    sequential_targets=["Glm4MoeLiteDecoderLayer"],
)

# Post-quantization cleanup
for name, module in model.named_modules():
    if type(module).__name__ == "CalibrationGlm4MoeLiteMoE":
        if hasattr(module, "_original_experts"):
            delattr(module, "_original_experts")

# Save to disk in compressed-tensors format.
SAVE_DIR = args.output_path
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

print("\n=== Complete ===")
print("Saved to:", SAVE_DIR)

print("\n\n========== SAMPLE GENERATION ==============")
dispatch_model(model)
inputs = tokenizer("Hello my name is", return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}
output = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(output[0], skip_special_tokens=True))
print("==========================================\n\n")

