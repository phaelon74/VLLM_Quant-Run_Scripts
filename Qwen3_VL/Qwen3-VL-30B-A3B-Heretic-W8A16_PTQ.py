"""
Qwen3-VL-30B-A3B-Instruct-Heretic W8A16 PTQ (Post-Training Quantization)

MoE VLM — catplusplus/Qwen3-VL-30B-A3B-Instruct-Heretic
(base: Qwen/Qwen3-VL-30B-A3B-Instruct)

  - QuantizationModifier with W8A16 preset (no AWQ, no GPTQ, no calibration data)
  - Qwen3VLMoeForConditionalGeneration + AutoProcessor (official MoE VL pattern)
  - MoE load helper probed across llm-compressor nightlies (load_context /
    load_quantizable_moe / replace_modules_for_calibration / plain load)
  - Vision tower left in BF16; MoE experts ARE quantized; router (mlp.gate) skipped
  - Saves processor so vision stays usable in vLLM / Transformers

  Example:
    python Qwen3-VL-30B-A3B-Heretic-W8A16_PTQ.py /path/to/model /path/to/out-W8A16
"""
import argparse
import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from compressed_tensors.offload import dispatch_model
from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration

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


def _ensure_moe_block_top_k(model) -> int:
    """
    llm-compressor's Qwen3-VL-MoE calibration shim reads `original.top_k`, but some
    Transformers builds only keep `num_experts_per_tok` on the text config / block.
    Set `.top_k` so MoE modules can be replaced and fused experts unpacked.
    """
    text_cfg = (
        model.config.get_text_config()
        if hasattr(model.config, "get_text_config")
        else model.config
    )
    fallback_top_k = getattr(text_cfg, "num_experts_per_tok", None)
    patched = 0
    for module in model.modules():
        if module.__class__.__name__ != "Qwen3VLMoeTextSparseMoeBlock":
            continue
        if hasattr(module, "top_k"):
            continue
        top_k = getattr(module, "num_experts_per_tok", None) or fallback_top_k
        if top_k is None:
            raise AttributeError(
                "Qwen3VLMoeTextSparseMoeBlock has no top_k/num_experts_per_tok; "
                "cannot prepare MoE calibration replacement."
            )
        module.top_k = int(top_k)
        patched += 1
    return patched


def _patch_qwen3_vl_moe_calib_for_topk_router() -> bool:
    """
    Transformers v5 Qwen3-VL-MoE uses Qwen3VLMoeTextTopKRouter (returns a tuple)
    and stores experts as [E, 2I, H] / [E, H, I]. Older llm-compressor calib
    modules assume Linear gate + transposed [H, 2I] expert storage.

    Monkeypatch the registered calib class to the Qwen3.5-style Linear-gate path.
    """
    try:
        import llmcompressor.modeling.qwen3_vl_moe as qvl
    except ImportError:
        print("WARNING: llmcompressor.modeling.qwen3_vl_moe not found; skip MoE patch")
        return False

    calib_cls = getattr(qvl, "CalibrateQwen3VLMoeTextSparseMoeBlock", None)
    seq_cls = getattr(qvl, "SequentialQwen3VLMoeTextExperts", None)
    if calib_cls is None or seq_cls is None:
        print("WARNING: Qwen3-VL MoE calib classes missing; skip MoE patch")
        return False

    try:
        from llmcompressor.utils.dev import skip_weights_initialize
    except ImportError:
        from contextlib import nullcontext as skip_weights_initialize

    def sequential_init(self, config, original):
        from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
            Qwen3VLMoeTextMLP,
        )

        text_config = getattr(config, "text_config", config)
        if hasattr(config, "get_text_config"):
            text_config = config.get_text_config()

        num_experts = getattr(text_config, "num_experts", original.gate_up_proj.shape[0])
        intermediate_size = getattr(
            text_config, "moe_intermediate_size", original.down_proj.shape[-1]
        )
        self.num_experts = num_experts

        with skip_weights_initialize():
            nn.ModuleList.__init__(
                self,
                [
                    Qwen3VLMoeTextMLP(text_config, intermediate_size=intermediate_size)
                    for _ in range(num_experts)
                ],
            )

        gate_up_data = original.gate_up_proj.data
        down_data = original.down_proj.data
        for i in range(num_experts):
            gate_up = gate_up_data[i]
            down = down_data[i]
            # New HF: [2I, H] already Linear layout. Old HF: [H, 2I] needs .t().
            if gate_up.shape[0] == 2 * intermediate_size:
                self[i].gate_proj.weight.data = (
                    gate_up[:intermediate_size].detach().clone().contiguous()
                )
                self[i].up_proj.weight.data = (
                    gate_up[intermediate_size:].detach().clone().contiguous()
                )
                self[i].down_proj.weight.data = down.detach().clone().contiguous()
            else:
                self[i].gate_proj.weight.data = (
                    gate_up[:, :intermediate_size].t().detach().clone().contiguous()
                )
                self[i].up_proj.weight.data = (
                    gate_up[:, intermediate_size:].t().detach().clone().contiguous()
                )
                self[i].down_proj.weight.data = (
                    down.t().detach().clone().contiguous()
                    if down.shape[0] == intermediate_size
                    else down.detach().clone().contiguous()
                )

    def calib_init(self, original, config, calibrate_all_experts: bool = True):
        nn.Module.__init__(self)
        text_config = getattr(config, "text_config", config)
        if hasattr(config, "get_text_config"):
            text_config = config.get_text_config()

        self.calibrate_all_experts = calibrate_all_experts
        self.top_k = int(
            getattr(original, "top_k", None)
            or getattr(text_config, "num_experts_per_tok")
        )
        self.num_experts = int(text_config.num_experts)
        self.hidden_size = int(text_config.hidden_size)
        self.hidden_dim = self.hidden_size

        # TopKRouter.weight or Linear.weight -> plain Linear (ignore-list friendly)
        gate = original.gate
        if hasattr(gate, "weight"):
            gate_w = gate.weight.data
        else:
            raise AttributeError(f"Unsupported MoE gate type: {type(gate)}")
        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        self.gate.weight.data = self.gate.weight.data.to(dtype=gate_w.dtype)
        self.gate.weight.data.copy_(gate_w)

        self.experts = seq_cls(config, original.experts)

    def calib_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.reshape(-1, hidden_dim)

        router_logits = F.linear(hidden_states_flat, self.gate.weight)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, router_indices = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states_flat.dtype)

        expert_mask = F.one_hot(router_indices, num_classes=self.num_experts).permute(
            2, 1, 0
        )
        next_states = torch.zeros_like(hidden_states_flat)

        for expert_idx, expert_layer in enumerate(self.experts):
            idx, token_idx = torch.where(expert_mask[expert_idx])
            if self.calibrate_all_experts:
                expert_out = expert_layer(hidden_states_flat)[token_idx]
            else:
                expert_out = expert_layer(hidden_states_flat[token_idx])
            if len(token_idx) > 0:
                weighted = expert_out * routing_weights[token_idx, idx, None]
                next_states.index_add_(
                    0, token_idx, weighted.to(hidden_states_flat.dtype)
                )

        return next_states.reshape(batch_size, sequence_length, hidden_dim)

    seq_cls.__init__ = sequential_init
    calib_cls.__init__ = calib_init
    calib_cls.forward = calib_forward
    # Permanent replace must stay for expert Linear targeting / save.
    calib_cls.is_permanent = True
    print(
        "Patched CalibrateQwen3VLMoeTextSparseMoeBlock for Transformers "
        "TopKRouter + [2I,H] expert layout"
    )
    return True


def _resolve_moe_load_context(model_cls):
    """
    Nightly llm-compressor MoE load APIs have moved around:
      newest  -> llmcompressor.utils.load_context
      prior   -> llmcompressor.modeling.moe.linearize.load_quantizable_moe
      oldest  -> replace_modules_for_calibration (post-load, not a CM)
    Returns (context_manager_or_nullcontext, post_load_fn_or_None, label).
    """
    try:
        from llmcompressor.utils import load_context

        return load_context(model_cls), None, "load_context"
    except ImportError:
        pass

    for import_path in (
        "llmcompressor.modeling.moe.linearize",
        "llmcompressor.modeling.moe",
        "llmcompressor.modeling.linearize",
    ):
        try:
            mod = __import__(import_path, fromlist=["load_quantizable_moe"])
            fn = getattr(mod, "load_quantizable_moe", None)
            if fn is not None:
                return fn(model_cls), None, f"{import_path}.load_quantizable_moe"
        except ImportError:
            continue

    for import_path in (
        "llmcompressor.modeling",
        "llmcompressor.modeling.prepare",
    ):
        try:
            mod = __import__(import_path, fromlist=["replace_modules_for_calibration"])
            fn = getattr(mod, "replace_modules_for_calibration", None)
            if fn is not None:
                return (
                    contextlib.nullcontext(),
                    fn,
                    f"{import_path}.replace_modules_for_calibration",
                )
        except ImportError:
            continue

    return (
        contextlib.nullcontext(),
        None,
        "plain from_pretrained (no MoE load helper in this llm-compressor)",
    )

# =========================
# CLI
# =========================
parser = argparse.ArgumentParser(
    description=(
        "Run W8A16 PTQ on catplusplus/Qwen3-VL-30B-A3B-Instruct-Heretic "
        "(Qwen3-VL MoE VLM). No calibration data needed. Vision preserved."
    )
)
parser.add_argument("model_path", type=str, help="Path to the source model directory.")
parser.add_argument("output_path", type=str, help="Path to save quantized model.")

args = parser.parse_args()
MODEL_ID = args.model_path

# =========================
# Model + processor
# =========================
# NOTE: Qwen3-VL-MoE needs transformers>=4.57 (or install from source).
load_cm, post_load_fn, load_label = _resolve_moe_load_context(
    Qwen3VLMoeForConditionalGeneration
)
print(f"MoE load path: {load_label}")
with load_cm:
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype="auto", trust_remote_code=True
    )
if post_load_fn is not None:
    model = post_load_fn(model)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
n_top_k = _ensure_moe_block_top_k(model)
if n_top_k:
    print(f"Patched top_k on {n_top_k} Qwen3VLMoeTextSparseMoeBlock modules")
print(f"Loaded model: {MODEL_ID}")

# =========================
# Quantization recipe
# =========================
# W8A16: INT8 per-channel symmetric weights, activations untouched.
# Ignore:
#   - lm_head: output head stays BF16
#   - visual.*: vision tower / projector stay BF16 (vision preserved)
#   - mlp.gate: MoE router stays BF16 (vLLM / compressed-tensors convention)
# Experts (gate_proj / up_proj / down_proj) ARE quantized — same as other MoE scripts.
recipe = QuantizationModifier(
    targets="Linear",
    scheme="W8A16",
    ignore=[
        "re:.*lm_head",
        "re:model.visual.*",
        "re:.*mlp.gate$",
    ],
)

# =========================
# Apply quantization (datafree — no calibration dataset)
# =========================
print("\n=== Running W8A16 PTQ (datafree, vision preserved) ===")
_patch_qwen3_vl_moe_calib_for_topk_router()
oneshot(model=model, recipe=recipe)

# =========================
# Quick sanity generation (text-only)
# =========================
print("\n\n========== SAMPLE GENERATION ==============")
dispatch_model(model)

SAMPLE_PROMPT = "Hello my name is"
messages = [{"role": "user", "content": SAMPLE_PROMPT}]
prompt_text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
inputs = processor(text=[prompt_text], return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items() if hasattr(v, "to")}
input_len = inputs["input_ids"].shape[-1]

output = model.generate(**inputs, max_new_tokens=100)
print(processor.decode(output[0][input_len:], skip_special_tokens=True))
print("==========================================\n\n")

# =========================
# Save compressed model + processor (vision usable downstream)
# =========================
SAVE_DIR = args.output_path
print(f"Saving to: {SAVE_DIR}")
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)

print("\n=== Complete ===")
print("Saved to:", SAVE_DIR)
