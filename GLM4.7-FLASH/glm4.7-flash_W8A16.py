"""
GLM-4.7-Flash W8A16 Quantization Script - vLLM Compatible Version 3

Uses argparse + YAML recipe for dataset configuration (consistent with llama-W4A16.py).
Uses local glm4_moe_lite_v2.py for MoE calibration (llm-compressor has no built-in
support for Glm4MoeLiteMoE - only Glm4MoeMoE is supported upstream).
DOES NOT ignore shared_experts so AWQ can track and smooth them properly.
"""

import argparse
import json
import yaml
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# Transformers v5 compatibility: TORCH_INIT_FUNCTIONS was removed in v5
# llm-compressor still imports it; inject if missing (see vllm-project/llm-compressor#2328)
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
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.awq.mappings import AWQMapping
from compressed_tensors.quantization import QuantizationScheme, QuantizationArgs

from glm4_moe_lite_v2 import CalibrationGlm4MoeLiteMoE  # noqa: F401

print("Imported CalibrationGlm4MoeLiteMoE v2 - produces vLLM-compatible expert names")


# =========================
# Monkey-patch: fix fused global scales for MLA attention
# =========================
# The upstream function blindly accesses submodule.q_proj/k_proj/v_proj
# which don't exist in MLA attention (GLM-4.7-Flash uses q_a_proj, q_b_proj,
# kv_a_proj_with_mqa, kv_b_proj). This is an NVFP4-specific optimization
# that doesn't apply to our INT8 quantization anyway.
import llmcompressor.modifiers.utils.helpers as _awq_helpers
import llmcompressor.modifiers.awq.base as _awq_base

def _safe_update_fused_layer_weight_global_scales(submodule):
    """Wrapper that skips MLA attention modules lacking q_proj/k_proj/v_proj.

    The upstream function blindly does [submodule.q_proj, submodule.v_proj, submodule.k_proj]
    which crashes on MLA attention (GLM-4.7-Flash uses q_a_proj, q_b_proj,
    kv_a_proj_with_mqa, kv_b_proj instead).
    This fusing step is only relevant for NVFP4 TENSOR_GROUP quantization, not our INT8.
    """
    # Skip entirely for MLA attention that lacks standard QKV projections
    has_standard_qkv = (
        hasattr(submodule, 'q_proj')
        and hasattr(submodule, 'k_proj')
        and hasattr(submodule, 'v_proj')
    )
    has_fused_qkv = hasattr(submodule, 'qkv_proj')

    if not has_standard_qkv and not has_fused_qkv:
        return  # MLA attention — nothing to fuse
    # For standard attention or MLP modules, delegate to original
    try:
        _orig_update_fused(submodule)
    except AttributeError:
        pass  # Safety net

_orig_update_fused = _awq_helpers.update_fused_layer_weight_global_scales

# Patch in BOTH places: the helpers module AND the awq.base module's local reference
_awq_helpers.update_fused_layer_weight_global_scales = _safe_update_fused_layer_weight_global_scales
_awq_base.update_fused_layer_weight_global_scales = _safe_update_fused_layer_weight_global_scales
print("Patched update_fused_layer_weight_global_scales for MLA attention compatibility")


# =========================
# Parse Command-Line Arguments
# =========================
parser = argparse.ArgumentParser(
    description="Run W8A16 AWQ quantization on GLM-4.7-Flash (glm4_moe_lite) model."
)
parser.add_argument(
    "source_folder",
    type=str,
    help="Path to the source model directory."
)
parser.add_argument(
    "destination_folder",
    type=str,
    help="Path to the destination directory for saving quantized model."
)
parser.add_argument(
    "dataset",
    type=str,
    help="Path to the dataset recipe YAML file (contains max_seq_length and dataset config)."
)
parser.add_argument(
    "group_size",
    type=int,
    help="Group size for W8A16 quantization (e.g., 32, 64, 128)."
)

args = parser.parse_args()
model_path = args.source_folder
output_path = args.destination_folder
recipe_yaml_path = args.dataset
group_size = args.group_size


# =========================
# Load Recipe YAML and extract config
# =========================
with open(recipe_yaml_path, 'r') as f:
    recipe_config = yaml.safe_load(f)

calibration_config = recipe_config.get('calibration_set', {})
MAX_SEQUENCE_LENGTH = calibration_config['max_seq_length']  # Required - fail if missing
SHUFFLE = calibration_config.get('shuffle', True)
SEED = calibration_config.get('seed', 42)
datasets_config = calibration_config.get('datasets', [])

print(f"Loaded recipe from: {recipe_yaml_path}")
print(f"  - max_seq_length: {MAX_SEQUENCE_LENGTH}")
print(f"  - shuffle: {SHUFFLE}")
print(f"  - seed: {SEED}")
print(f"  - group_size: {group_size}")
print(f"  - datasets to load: {len(datasets_config)}")


# =========================
# Manual MoE Module Replacement
# =========================
_original_moe_modules = {}

def replace_moe_modules_for_calibration(model):
    """
    Replace all Glm4MoeLiteMoE modules with CalibrationGlm4MoeLiteMoE v2.
    
    Required because llm-compressor has NO built-in support for Glm4MoeLiteMoE.
    Upstream glm4_moe.py only supports Glm4MoeMoE (full model, not lite).
    """
    global _original_moe_modules
    _original_moe_modules.clear()
    
    replaced_count = 0
    for name, module in list(model.named_modules()):
        if type(module).__name__ == "Glm4MoeLiteMoE":
            parts = name.split(".")
            attr_name = parts[-1]
            
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            
            _original_moe_modules[name] = module
            
            calibration_module = CalibrationGlm4MoeLiteMoE(
                original=module,
                config=model.config,
                calibrate_all_experts=True
            )
            setattr(parent, attr_name, calibration_module)
            replaced_count += 1
    
    print(f"Replaced {replaced_count} MoE modules with calibration versions (v2)")
    return model

# =========================
# Model
# =========================
MODEL_ID = model_path

# Optional: log model config (non-blocking if transformers version mismatch)
try:
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    print(f"Model config type: {type(config).__name__}")
    print(f"Model type: {getattr(config, 'model_type', 'unknown')}")
except (ValueError, KeyError) as e:
    print(f"Note: AutoConfig could not resolve model type (may need newer transformers): {e}")
    print("Proceeding with model loading anyway...")

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# Replace MoE modules with calibration versions (all experts activated)
model = replace_moe_modules_for_calibration(model)

# =========================
# Dataset Formatters
# =========================
def format_sharegpt(example, columns, tokenizer):
    """Format ShareGPT-style conversations."""
    formatted_messages = []
    
    # Check if first column is system_prompt
    if len(columns) >= 2 and 'system' in columns[0].lower():
        system_prompt = example.get(columns[0], '')
        if system_prompt:
            formatted_messages.append({'role': 'system', 'content': str(system_prompt)})
        conv_column = columns[1]
    else:
        conv_column = columns[0]
    
    messages = example.get(conv_column, [])
    
    if isinstance(messages, str):
        try:
            import json
            messages = json.loads(messages)
        except Exception:
            formatted_messages.append({'role': 'user', 'content': messages})
            if formatted_messages:
                text = tokenizer.apply_chat_template(formatted_messages, tokenize=False)
                return {'text': text}
            return {'text': ''}
    
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get('role', msg.get('from', 'user'))
                content = msg.get('content', msg.get('value', ''))
                if role in ['human', 'user']:
                    role = 'user'
                elif role in ['gpt', 'assistant', 'bot']:
                    role = 'assistant'
                elif role == 'system':
                    role = 'system'
                if content:
                    formatted_messages.append({'role': role, 'content': str(content)})
            elif isinstance(msg, str):
                idx = len([m for m in formatted_messages if m['role'] != 'system'])
                role = 'user' if idx % 2 == 0 else 'assistant'
                formatted_messages.append({'role': role, 'content': str(msg)})
    
    if not formatted_messages:
        return {'text': ''}
    
    try:
        text = tokenizer.apply_chat_template(formatted_messages, tokenize=False)
        return {'text': text}
    except Exception:
        return {'text': ''}


def format_prompt_answer(example, columns, tokenizer):
    """Format prompt/answer pairs (e.g., instruction/response)."""
    prompt_col = columns[0]
    answer_col = columns[1] if len(columns) > 1 else columns[0]
    
    prompt = example.get(prompt_col, '')
    answer = example.get(answer_col, '')
    
    messages = [
        {'role': 'user', 'content': str(prompt)},
        {'role': 'assistant', 'content': str(answer)}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {'text': text}


def format_chat_completion(example, columns, tokenizer):
    """Format chat completion style data."""
    for col in columns:
        if col in example:
            data = example[col]
            if isinstance(data, list) and len(data) > 0:
                if isinstance(data[0], dict):
                    text = tokenizer.apply_chat_template(data, tokenize=False)
                    return {'text': text}
                elif isinstance(data[0], str):
                    messages = []
                    for i, item in enumerate(data):
                        role = 'user' if i % 2 == 0 else 'assistant'
                        messages.append({'role': role, 'content': str(item)})
                    text = tokenizer.apply_chat_template(messages, tokenize=False)
                    return {'text': text}
            elif isinstance(data, str):
                return {'text': str(data)}
    
    text = ' '.join(str(example.get(col, '')) for col in columns)
    return {'text': text}


def format_raw_text(example, columns, tokenizer):
    """Format raw text data."""
    texts = []
    for col in columns:
        if col in example and example[col]:
            texts.append(str(example[col]))
    return {'text': ' '.join(texts)}


FORMATTERS = {
    'sharegpt': format_sharegpt,
    'prompt_answer': format_prompt_answer,
    'chat_completion': format_chat_completion,
    'raw_text': format_raw_text,
}


# =========================
# Load datasets from YAML recipe
# =========================
print("\n=== Loading datasets from recipe ===")
all_datasets = []
total_samples = 0

for ds_config in datasets_config:
    dataset_name = ds_config['dataset']
    split = ds_config.get('split', 'train')
    columns = ds_config.get('columns', [])
    formatter_name = ds_config.get('formatter', 'raw_text')
    num_samples = ds_config.get('num_samples', 10)
    streaming = ds_config.get('streaming', False)
    
    print(f"  Loading: {dataset_name} (split={split}, samples={num_samples}, formatter={formatter_name})")
    
    try:
        if streaming:
            ds = load_dataset(dataset_name, split=split, streaming=True)
            ds = ds.take(num_samples)
            ds = list(ds)
            from datasets import Dataset
            ds = Dataset.from_list(ds)
        else:
            ds = load_dataset(dataset_name, split=split)
            n = min(num_samples, len(ds))
            ds = ds.shuffle(seed=SEED).select(range(n))
        
        formatter_fn = FORMATTERS.get(formatter_name, format_raw_text)
        
        ds = ds.map(
            lambda x: formatter_fn(x, columns, tokenizer),
            remove_columns=ds.column_names,
            num_proc=1,
        )
        
        ds = ds.filter(lambda x: len(x.get('text', '')) > 0)
        
        all_datasets.append(ds)
        total_samples += len(ds)
        print(f"    -> Loaded {len(ds)} samples")
        
    except Exception as e:
        print(f"    -> WARNING: Failed to load {dataset_name}: {e}")
        continue

if not all_datasets:
    raise ValueError("No datasets were successfully loaded from the recipe!")

ds = concatenate_datasets(all_datasets)
print(f"\n=== Total samples loaded: {total_samples} ===")

if SHUFFLE:
    ds = ds.shuffle(seed=SEED)


# =========================
# Tokenize in batches
# =========================
print("\n=== Tokenizing dataset ===")
ds = ds.map(
    lambda batch: tokenizer(
        batch["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    ),
    batched=True,
    remove_columns=ds.column_names,
    num_proc=4,
)

NUM_CALIBRATION_SAMPLES = len(ds)
print(f"Tokenized {NUM_CALIBRATION_SAMPLES} samples (max_seq_length={MAX_SEQUENCE_LENGTH})")

# =========================
# Quantization recipe  (W8A16-SYM, Marlin-friendly)
# =========================
# Ignore list for GLM-4.7-Flash (glm4_moe_lite):
#   - lm_head: output projection (always ignore)
#   - Layer 0: dense layer (first_k_dense_replace=1), keep high precision for stability
#   - Router gates: MoE routing is sensitive to quantization
#   - shared_experts: NOT ignored (AWQ needs to track and smooth them)
ignore_list = [
    # Output head
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
    
    # Router gates - sensitive to quantization
    "re:.*mlp\\.gate$",
]

print(f"Ignore list has {len(ignore_list)} entries (shared_experts NOT ignored)")

weight_args = QuantizationArgs(
    num_bits=8,          # 8-bit weights (W8A16)
    type="int",
    symmetric=True,      # SYMMETRIC (Marlin requirement)
    strategy="group",    # group-wise quantization
    group_size=group_size,  # Dynamic group size from argument
)

quant_scheme = QuantizationScheme(
    targets=["Linear"],
    weights=weight_args,
    input_activations=None,   # A16 (leave activations in FP16/BF16)
    output_activations=None,
)

# AWQ Mappings for GLM-4.7-Flash (MLA + MoE architecture)
# GLM-4.7-Flash uses Multi-head Latent Attention (same as DeepSeek-V3):
#   q_a_proj, q_b_proj, kv_a_proj_with_mqa, kv_b_proj instead of q/k/v_proj
# The default AWQ mappings assume standard q/k/v_proj and will fail.
glm4_moe_lite_mappings = [
    # MLA: input_layernorm feeds into q_a_proj and kv_a_proj_with_mqa
    AWQMapping(
        "re:.*input_layernorm$",
        ["re:.*(q|q_a)_proj$", "re:.*kv_a_proj_with_mqa$"],
    ),
    # MLA: q_a_layernorm feeds into q_b_proj
    AWQMapping("re:.*q_a_layernorm$", ["re:.*q_b_proj$"]),
    # MLA: kv_a_layernorm feeds into kv_b_proj
    AWQMapping("re:.*kv_a_layernorm$", ["re:.*kv_b_proj$"]),
    # MLP: post_attention_layernorm feeds into gate_proj and up_proj
    # (matches dense layer 0, MoE experts, and shared_experts)
    AWQMapping(
        "re:.*post_attention_layernorm$",
        ["re:.*gate_proj$", "re:.*up_proj$"],
    ),
    # MLP: up_proj feeds into down_proj
    AWQMapping("re:.*up_proj$", ["re:.*down_proj$"]),
]

recipe = [
    AWQModifier(
        ignore=ignore_list,
        mappings=glm4_moe_lite_mappings,
        config_groups={"group_0": quant_scheme},
    ),
]

# =========================
# Run one-shot compression
# =========================
print("\n=== Running one-shot compression ===")
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    tokenizer=tokenizer,
    # Explicit sequential targets to bypass _get_no_split_modules() which
    # was renamed to _no_split_modules in transformers 5.x
    sequential_targets=["Glm4MoeLiteDecoderLayer"],
)

# =========================
# Post-quantization cleanup
# =========================
for name, module in model.named_modules():
    if type(module).__name__ == "CalibrationGlm4MoeLiteMoE":
        if hasattr(module, '_original_experts'):
            delattr(module, '_original_experts')

# Count quantized modules
linear_count = 0
quantized_count = 0
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        linear_count += 1
        if hasattr(module, 'quantization_scheme'):
            quantized_count += 1
print(f"Total Linear modules: {linear_count}, Quantized: {quantized_count}")

# Fix generation config
if hasattr(model, 'generation_config') and model.generation_config is not None:
    if hasattr(model.generation_config, 'temperature') and model.generation_config.temperature is not None:
        if not getattr(model.generation_config, 'do_sample', False):
            model.generation_config.do_sample = True

# =========================
# Save compressed model
# =========================
SAVE_DIR = output_path
print(f"\nSaving to: {SAVE_DIR}")
model.save_pretrained(SAVE_DIR, save_compressed=True, max_shard_size="5GB")
tokenizer.save_pretrained(SAVE_DIR)

# Post-save fix: Remove auto_map from config.json
save_path = Path(SAVE_DIR)
config_path = save_path / "config.json"
if config_path.exists():
    with open(config_path, 'r') as f:
        saved_config = json.load(f)
    if 'auto_map' in saved_config:
        print("Removing auto_map from config.json")
        del saved_config['auto_map']
        with open(config_path, 'w') as f:
            json.dump(saved_config, f, indent=2)

print("\n" + "=" * 60)
print("QUANTIZATION COMPLETE!")
print("=" * 60)
print(f"\nModel saved to: {SAVE_DIR}")
print(f"\nW8A16: 8-bit integer weights, 16-bit float activations")
print(f"Group size: {group_size}")
print(f"\nvLLM command:")
print(f'  vllm serve "{SAVE_DIR}" \\')
print(f'       --tensor-parallel-size 2 \\')
print(f'       --tool-call-parser glm47 \\')
print(f'       --reasoning-parser glm45 \\')
print(f'       --enable-auto-tool-choice \\')
print(f'       --trust-remote-code')
