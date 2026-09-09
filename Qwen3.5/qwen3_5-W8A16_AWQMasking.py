import argparse
import json
import yaml

import torch.nn as nn
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)

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

# =========================
# Parse Command-Line Arguments
# =========================
parser = argparse.ArgumentParser(
    description="Run W8A16 AWQ quantization on Qwen3.5 VLM (e.g., Qwen3.5-27B)."
)
parser.add_argument(
    "model_path",
    type=str,
    help="Path to the source model directory."
)
parser.add_argument(
    "output_path",
    type=str,
    help="Path to the destination directory for saving quantized model."
)
parser.add_argument(
    "recipe_yaml",
    type=str,
    help="Path to the dataset recipe YAML file (contains max_seq_length and dataset config)."
)
parser.add_argument(
    "group_size",
    type=int,
    help="Group size for W8A16 quantization (e.g., 32, 64, 128)."
)
parser.add_argument(
    "--use-loss-mask",
    action="store_true",
    default=False,
    help="Enable AWQ loss masking (only completion tokens contribute to calibration)."
)

args = parser.parse_args()
model_path = args.model_path
output_path = args.output_path
recipe_yaml_path = args.recipe_yaml
group_size = args.group_size
use_loss_mask = args.use_loss_mask

# =========================
# Load Recipe YAML and extract config
# =========================
with open(recipe_yaml_path, 'r') as f:
    recipe_config = yaml.safe_load(f)

# Extract config from calibration_set section
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
print(f"  - use_loss_mask: {use_loss_mask}")
print(f"  - datasets to load: {len(datasets_config)}")

# =========================
# Model
# =========================
MODEL_ID = model_path

# Load with AutoModelForCausalLM — matching the working NVFP4A16 reference.
# AutoModelForImageTextToText wraps the LM under .language_model, which saves
# weights with VLM paths that vLLM doesn't recognize -> garbage output.
config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
model_type = getattr(config, "model_type", "")
print(f"Model config type: {type(config).__name__}, model_type: {model_type}")

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto", trust_remote_code=True)
print(f"Loaded model with AutoModelForCausalLM (matching NVFP4A16 reference)")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# =========================
# Ignore list — aligned with qwen3_5-NVFP4A16.py (working reference)
# Skip ALL linear_attn (Gated DeltaNet fused projections are incompatible with
# quantization per reference). Also skip vision, MTP, lm_head.
# =========================
vision_keywords = (
    "vision", "visual", "vision_tower", "image", "pixel", "clip",
    "vit", "resampler", "mm_projector", "multimodal_projector",
    "projector", "logit_scale", "merger",
)

ignore_list = ["lm_head"]

for mod_name, mod in model.named_modules():
    if not isinstance(mod, nn.Linear):
        continue

    name_lower = mod_name.lower()

    # Vision / projector components — keep entirely in BF16
    if any(k in name_lower for k in vision_keywords):
        ignore_list.append(mod_name)
        continue

    # MTP (multi-token prediction) head
    if "mtp" in name_lower:
        ignore_list.append(mod_name)
        continue

    # Final output head
    if name_lower.endswith("lm_head") or name_lower == "lm_head":
        ignore_list.append(mod_name)
        continue

    # ALL linear_attn (Gated DeltaNet) — incompatible with quantization per NVFP4A16 reference
    if "linear_attn" in name_lower:
        ignore_list.append(mod_name)
        continue

    # Catch any remaining layers whose dimensions are incompatible with group_size
    if mod.out_features % group_size != 0:
        ignore_list.append(mod_name)

ignore_list = sorted(set(ignore_list))

linear_count = sum(1 for _, mod in model.named_modules() if isinstance(mod, nn.Linear))
print(f"Linear modules: {linear_count}, ignored: {len(ignore_list)}")
for m in ignore_list:
    print(f"  - {m}")

# =========================
# Dataset Formatters
# =========================
def format_sharegpt(example, columns, tokenizer):
    """Format ShareGPT-style conversations."""
    formatted_messages = []
    
    # Check if first column is system_prompt (for datasets like Gryphe/Sonnet3.5-Charcard-Roleplay)
    if len(columns) >= 2 and 'system' in columns[0].lower():
        system_prompt = example.get(columns[0], '')
        if system_prompt:
            formatted_messages.append({'role': 'system', 'content': str(system_prompt)})
        conv_column = columns[1]
    else:
        conv_column = columns[0]
    
    # Get conversation data
    messages = example.get(conv_column, [])
    
    # Handle case where messages is a string (some datasets store JSON strings)
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except:
            # Not JSON, treat as raw text
            formatted_messages.append({'role': 'user', 'content': messages})
            if formatted_messages:
                text = tokenizer.apply_chat_template(formatted_messages, tokenize=False)
                return {'text': text, 'messages': json.dumps(formatted_messages)}
            return {'text': '', 'messages': ''}
    
    # Convert to standard format if needed
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get('role', msg.get('from', 'user'))
                content = msg.get('content', msg.get('value', ''))
                # Normalize role names
                if role in ['human', 'user']:
                    role = 'user'
                elif role in ['gpt', 'assistant', 'bot']:
                    role = 'assistant'
                elif role == 'system':
                    role = 'system'
                if content:  # Only add if there's content
                    formatted_messages.append({'role': role, 'content': str(content)})
            elif isinstance(msg, str):
                # Alternate user/assistant for string lists
                idx = len([m for m in formatted_messages if m['role'] != 'system'])
                role = 'user' if idx % 2 == 0 else 'assistant'
                formatted_messages.append({'role': role, 'content': str(msg)})
    
    if not formatted_messages:
        return {'text': '', 'messages': ''}
    
    try:
        text = tokenizer.apply_chat_template(formatted_messages, tokenize=False)
        return {'text': text, 'messages': json.dumps(formatted_messages)}
    except Exception as e:
        # If chat template fails, return empty
        return {'text': '', 'messages': ''}


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
    return {'text': text, 'messages': json.dumps(messages)}


def format_chat_completion(example, columns, tokenizer):
    """Format chat completion style data."""
    # Try to find messages-like column
    for col in columns:
        if col in example:
            data = example[col]
            if isinstance(data, list) and len(data) > 0:
                if isinstance(data[0], dict):
                    # Already in messages format
                    text = tokenizer.apply_chat_template(data, tokenize=False)
                    return {'text': text, 'messages': json.dumps(data)}
                elif isinstance(data[0], str):
                    # List of strings - alternate user/assistant
                    messages = []
                    for i, item in enumerate(data):
                        role = 'user' if i % 2 == 0 else 'assistant'
                        messages.append({'role': role, 'content': str(item)})
                    text = tokenizer.apply_chat_template(messages, tokenize=False)
                    return {'text': text, 'messages': json.dumps(messages)}
            elif isinstance(data, str):
                # Single text field
                return {'text': str(data), 'messages': ''}
    
    # Fallback: concatenate all columns
    text = ' '.join(str(example.get(col, '')) for col in columns)
    return {'text': text, 'messages': ''}


def format_raw_text(example, columns, tokenizer):
    """Format raw text data."""
    texts = []
    for col in columns:
        if col in example and example[col]:
            texts.append(str(example[col]))
    return {'text': ' '.join(texts), 'messages': ''}


FORMATTERS = {
    'sharegpt': format_sharegpt,
    'prompt_answer': format_prompt_answer,
    'chat_completion': format_chat_completion,
    'raw_text': format_raw_text,
}


def generate_assistant_mask(messages, tokenizer, max_seq_length):
    """
    Generate loss_mask: 1 for assistant/completion tokens, 0 for prompt tokens.
    Tries apply_chat_template(return_assistant_tokens_mask=True) first;
    falls back to heuristic (prompt-only tokenization) if unsupported.
    """
    try:
        result = tokenizer.apply_chat_template(
            messages,
            return_assistant_tokens_mask=True,
            return_dict=True,
            add_special_tokens=False,
            max_length=max_seq_length,
            truncation=True,
        )
        mask = result.get("assistant_tokens_mask")
        if mask is not None:
            return mask[:max_seq_length]
    except (TypeError, ValueError, KeyError):
        pass

    # Fallback: tokenize prompt-only to find boundary
    try:
        # Build prompt: all messages except last assistant content
        if not messages:
            return []
        last_role = messages[-1].get("role", "user")
        if last_role != "assistant":
            return [0] * max_seq_length  # No completion tokens

        prompt_messages = messages[:-1] + [{"role": "assistant", "content": ""}]
        prompt_ids = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=False,
            add_special_tokens=False,
        )
        full_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_special_tokens=False,
            max_length=max_seq_length,
            truncation=True,
        )
        prompt_len = min(len(prompt_ids), len(full_ids), max_seq_length)
        seq_len = min(len(full_ids), max_seq_length)
        return [0] * prompt_len + [1] * (seq_len - prompt_len)
    except Exception:
        return [1] * max_seq_length  # Fallback: treat all as completion


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
        # Load dataset
        if streaming:
            ds = load_dataset(dataset_name, split=split, streaming=True)
            # Take samples from streaming dataset
            ds = ds.take(num_samples)
            # Convert to regular dataset
            ds = list(ds)
            from datasets import Dataset
            ds = Dataset.from_list(ds)
        else:
            ds = load_dataset(dataset_name, split=split)
            # Sample from dataset
            n = min(num_samples, len(ds))
            ds = ds.shuffle(seed=SEED).select(range(n))
        
        # Get formatter function
        formatter_fn = FORMATTERS.get(formatter_name, format_raw_text)
        
        # Apply formatter
        ds = ds.map(
            lambda x: formatter_fn(x, columns, tokenizer),
            remove_columns=ds.column_names,
            num_proc=1,  # Use single proc to avoid tokenizer issues
        )
        
        # Filter out empty texts
        ds = ds.filter(lambda x: len(x.get('text', '')) > 0)
        
        all_datasets.append(ds)
        total_samples += len(ds)
        print(f"    -> Loaded {len(ds)} samples")
        
    except Exception as e:
        print(f"    -> WARNING: Failed to load {dataset_name}: {e}")
        continue

# Concatenate all datasets
if not all_datasets:
    raise ValueError("No datasets were successfully loaded from the recipe!")

ds = concatenate_datasets(all_datasets)
print(f"\n=== Total samples loaded: {total_samples} ===")

# Shuffle combined dataset if requested
if SHUFFLE:
    ds = ds.shuffle(seed=SEED)


# =========================
# Tokenize in batches
# =========================
print("\n=== Tokenizing dataset ===")


def tokenize_with_mask(batch):
    result = tokenizer(
        batch["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )
    if use_loss_mask:
        loss_masks = []
        for i, messages_json in enumerate(batch["messages"]):
            input_ids = result["input_ids"][i]
            seq_len = len(input_ids)
            if messages_json is None or messages_json == "":
                loss_masks.append([1] * seq_len)  # raw_text: all completion
            else:
                messages = json.loads(messages_json)
                mask = generate_assistant_mask(messages, tokenizer, MAX_SEQUENCE_LENGTH)
                loss_masks.append(mask[:seq_len])
        result["loss_mask"] = loss_masks
    return result


ds = ds.map(
    tokenize_with_mask,
    batched=True,
    remove_columns=ds.column_names,
    num_proc=1 if use_loss_mask else 4,  # Single proc when using messages for mask
)

NUM_CALIBRATION_SAMPLES = len(ds)
print(f"Tokenized {NUM_CALIBRATION_SAMPLES} samples (max_seq_length={MAX_SEQUENCE_LENGTH}, use_loss_mask={use_loss_mask})")


# =========================
# Quantization recipe  (W8A16-SYM, Marlin-friendly)
# =========================
from compressed_tensors.quantization import QuantizationScheme, QuantizationArgs

weight_args = QuantizationArgs(
    num_bits=8,          # 8-bit weights
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

# =========================
# Build explicit per-layer AWQ mappings
# =========================
# Qwen3.5 VLM is not in AWQ's known architectures, so default regex mappings
# fail (they match ALL layers at once instead of one-per-mapping).
# Hybrid attention (self_attn vs linear_attn) also means not all layers have
# q_proj/k_proj/v_proj. We must build per-layer mappings explicitly.

import re as _re
ignore_set = set(ignore_list)

# Derive layer prefix from actual module names
_layer_prefix = None
for mod_name, mod in model.named_modules():
    if isinstance(mod, nn.Linear) and "mlp.gate_proj" in mod_name and "layers." in mod_name:
        _layer_prefix = mod_name.rsplit(".mlp.gate_proj", 1)[0]
        break
if _layer_prefix is None:
    raise AttributeError("Could not find mlp.gate_proj — cannot derive layer prefix")
_m = _re.match(r"^(.+\.layers)\.\d+$", _layer_prefix)
layer_prefix = _m.group(1) if _m else _layer_prefix.rsplit(".", 1)[0]

if hasattr(model.model, "language_model") and hasattr(model.model.language_model, "layers"):
    text_model = model.model.language_model
elif hasattr(model.model, "layers"):
    text_model = model.model
else:
    raise AttributeError("Could not locate layers container in model")
num_layers = len(text_model.layers)

print(f"Layer prefix: {layer_prefix} ({num_layers} layers)")

awq_mappings = []
for i in range(num_layers):
    pfx = f"{layer_prefix}.{i}"
    layer = text_model.layers[i]

    # Self-attention layers: input_layernorm -> q/k/v_proj, v_proj -> o_proj
    if hasattr(layer, 'self_attn'):
        qkv = [f"{pfx}.self_attn.{p}" for p in ("q_proj", "k_proj", "v_proj")
               if f"{pfx}.self_attn.{p}" not in ignore_set]
        if qkv:
            awq_mappings.append(AWQMapping(
                smooth_layer=f"{pfx}.input_layernorm",
                balance_layers=qkv,
            ))
        if f"{pfx}.self_attn.o_proj" not in ignore_set and f"{pfx}.self_attn.v_proj" not in ignore_set:
            awq_mappings.append(AWQMapping(
                smooth_layer=f"{pfx}.self_attn.v_proj",
                balance_layers=[f"{pfx}.self_attn.o_proj"],
            ))

    # linear_attn layers: all ignored, no attention AWQ mappings

    # MLP mappings (all layers have MLP)
    gate_up = [f"{pfx}.mlp.{p}" for p in ("gate_proj", "up_proj")
               if f"{pfx}.mlp.{p}" not in ignore_set]
    if gate_up:
        awq_mappings.append(AWQMapping(
            smooth_layer=f"{pfx}.post_attention_layernorm",
            balance_layers=gate_up,
        ))
    if f"{pfx}.mlp.down_proj" not in ignore_set and f"{pfx}.mlp.up_proj" not in ignore_set:
        awq_mappings.append(AWQMapping(
            smooth_layer=f"{pfx}.mlp.up_proj",
            balance_layers=[f"{pfx}.mlp.down_proj"],
        ))

print(f"Built {len(awq_mappings)} AWQ mappings")

recipe = [
    AWQModifier(
        ignore=ignore_list,
        config_groups={"group_0": quant_scheme},
        mappings=awq_mappings,
    ),
]

# =========================
# Run one-shot compression
# =========================
print("\n=== Running one-shot compression ===")
oneshot_kwargs = dict(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    tokenizer=tokenizer,
    use_loss_mask=use_loss_mask,
    # Explicit sequential targets to bypass _get_no_split_modules() which
    # was removed in transformers 5.x (replaced by _no_split_modules attribute)
    sequential_targets=["Qwen3_5DecoderLayer"],
)
if use_loss_mask:
    oneshot_kwargs["pipeline"] = "sequential"  # Required for AWQ masking
oneshot(**oneshot_kwargs)

# =========================
# Quick sanity generation (verify quantization before saving)
# =========================
print("\n\n========== SAMPLE GENERATION ==============")
from compressed_tensors.offload import dispatch_model
dispatch_model(model)
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(model.device)
output = model.generate(input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0], skip_special_tokens=True))
print("==========================================\n\n")

# =========================
# Save compressed model
# =========================
SAVE_DIR = output_path
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

print("\n=== Complete ===")
print("Saved to:", SAVE_DIR)
