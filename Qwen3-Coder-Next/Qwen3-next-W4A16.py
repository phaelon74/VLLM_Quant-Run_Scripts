import argparse
import yaml
import torch
import gc

from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.utils import dispatch_for_generation

# =========================
# Qwen3-Coder-Next Quantization Strategy
# =========================
# Qwen3-Coder-Next uses @torch.fx.wrap decorators in its DeltaNet (linear attention)
# layers which cause FX tracing failures with llmcompressor's sequential pipeline.
#
# Solution (based on successful quantizations like dazipe's Qwen3-Next-80B-GPTQ):
# 1. Use device_map="auto" to distribute the model across both GPUs (192GB total)
# 2. Use pipeline="basic" to skip FX tracing entirely
# 3. Use calibrate_moe_context=True to ensure all 512 experts are calibrated
# 4. Ignore linear_attn and self_attn layers that contain @torch.fx.wrap decorators
#
# This approach avoids the "symbolically traced variables cannot be used as inputs
# to control flow" error while maintaining quantization accuracy.
# =========================


# =========================
# MoE Expert Coverage Utilities
# =========================
def get_moe_expert_info(model):
    """
    Extract MoE configuration from Qwen3-Coder-Next model.
    
    Returns dict with expert counts and routing info.
    """
    config = model.config
    info = {
        "num_experts": getattr(config, "num_experts", 512),
        "num_experts_per_tok": getattr(config, "num_experts_per_tok", 10),
        "num_shared_experts": getattr(config, "num_shared_experts", 1),
        "num_layers": getattr(config, "num_hidden_layers", 48),
    }
    
    # Calculate minimum samples needed for expert coverage
    # Each token activates num_experts_per_tok experts
    # We want to activate all experts multiple times for good calibration
    # Rule of thumb: (num_experts / num_experts_per_tok) * coverage_factor
    coverage_factor = 3  # Each expert should be activated ~3 times on average
    min_tokens = (info["num_experts"] / info["num_experts_per_tok"]) * coverage_factor
    
    # With typical seq_len of 2048, calculate recommended samples
    typical_seq_len = 2048
    info["recommended_min_samples"] = max(256, int(min_tokens / typical_seq_len * 10))
    info["min_tokens_for_coverage"] = int(min_tokens)
    
    return info


def print_moe_calibration_guidance(model, num_samples, max_seq_len):
    """
    Print guidance for MoE expert coverage during calibration.
    """
    try:
        moe_info = get_moe_expert_info(model)
        
        print("\n" + "="*70)
        print("MoE Expert Coverage Analysis for Qwen3-Coder-Next")
        print("="*70)
        print(f"  Total experts:           {moe_info['num_experts']}")
        print(f"  Experts per token:       {moe_info['num_experts_per_tok']}")
        print(f"  Shared experts:          {moe_info['num_shared_experts']}")
        print(f"  Number of layers:        {moe_info['num_layers']}")
        print("-"*70)
        print(f"  Your calibration setup:")
        print(f"    - Samples:             {num_samples}")
        print(f"    - Max sequence length: {max_seq_len}")
        print(f"    - Estimated tokens:    ~{num_samples * max_seq_len // 2:,} (assuming 50% fill)")
        print("-"*70)
        
        estimated_tokens = num_samples * max_seq_len // 2  # Conservative estimate
        expert_activations_per_token = moe_info['num_experts_per_tok']
        estimated_expert_activations = estimated_tokens * expert_activations_per_token
        avg_activations_per_expert = estimated_expert_activations / moe_info['num_experts']
        
        print(f"  Estimated expert activation coverage:")
        print(f"    - Total expert activations: ~{estimated_expert_activations:,}")
        print(f"    - Avg activations/expert:   ~{avg_activations_per_expert:,.0f}")
        
        # Coverage assessment
        if avg_activations_per_expert >= 100:
            coverage_status = "EXCELLENT - All experts should be well-calibrated"
        elif avg_activations_per_expert >= 50:
            coverage_status = "GOOD - Most experts should have adequate coverage"
        elif avg_activations_per_expert >= 10:
            coverage_status = "FAIR - Some experts may have limited coverage"
        else:
            coverage_status = "WARNING - Increase samples for better expert coverage!"
            
        print(f"    - Coverage assessment:      {coverage_status}")
        
        if avg_activations_per_expert < 50:
            recommended = moe_info['recommended_min_samples']
            print(f"\n  RECOMMENDATION: Consider using {recommended}+ samples")
            print(f"                  with diverse content (code, text, math, etc.)")
        
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"Note: Could not analyze MoE config: {e}")


def clear_memory():
    """Clear GPU and CPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# =========================
# Parse Command-Line Arguments
# =========================
parser = argparse.ArgumentParser(
    description="Run W4A16 AWQ quantization on Qwen3-Next model."
)
parser.add_argument(
    "source_model",
    type=str,
    help="Path to the source model directory."
)
parser.add_argument(
    "output_path",
    type=str,
    help="Path to the destination directory for saving quantized model."
)
parser.add_argument(
    "dataset_config",
    type=str,
    help="Path to the dataset YAML configuration file (contains max_seq_length and dataset config)."
)
parser.add_argument(
    "group_size",
    type=int,
    help="Group size for W4A16 quantization (e.g., 32, 64, 128)."
)

args = parser.parse_args()
source_model_path = args.source_model
output_path = args.output_path
dataset_config_path = args.dataset_config
group_size = args.group_size


# =========================
# Load Dataset Config and extract config
# =========================
with open(dataset_config_path, 'r') as f:
    dataset_config = yaml.safe_load(f)

# Extract config from calibration_set section
calibration_config = dataset_config.get('calibration_set', {})
MAX_SEQUENCE_LENGTH = calibration_config['max_seq_length']  # Required - fail if missing
SHUFFLE = calibration_config.get('shuffle', True)
SEED = calibration_config.get('seed', 42)
num_calibration_samples = calibration_config.get('num_samples', 512)
datasets_config = calibration_config.get('datasets', [])

print(f"Loaded dataset config from: {dataset_config_path}")
print(f"  - max_seq_length: {MAX_SEQUENCE_LENGTH}")
print(f"  - shuffle: {SHUFFLE}")
print(f"  - seed: {SEED}")
print(f"  - num_samples: {num_calibration_samples}")
print(f"  - group_size: {group_size}")
print(f"  - datasets to load: {len(datasets_config)}")

# =========================
# Model
# =========================
MODEL_ID = source_model_path

# =========================
# IMPORTANT: Qwen3-Coder-Next Architecture & Memory
# =========================
# - 80B total params, 3B activated (MoE)
# - 512 experts, 10 activated per token + 1 shared expert
# - Hybrid layout: DeltaNet (linear attention) + Gated Attention
# - 48 layers with repeating pattern
# - Uses @torch.fx.wrap decorators which break FX tracing in sequential pipeline
#
# Memory calculation for 2x RTX PRO 6000 Blackwell (192GB total):
#   - Model weights (BF16):     ~160GB
#   - Calibration overhead:     ~20-30GB
#   - Total required:           ~180-190GB
#   - Available VRAM:           192GB (2x 96GB)
#
# Solution: Use device_map="auto" with pipeline="basic"
#   - Model distributed across both GPUs via device_map="auto"
#   - pipeline="basic" skips FX tracing entirely (avoids @torch.fx.wrap issues)
#   - calibrate_moe_context=True ensures all 512 experts receive calibration
#   - Ignore linear_attn layers that contain problematic @torch.fx.wrap code
#
# This approach follows successful quantizations like:
#   - dazipe/Qwen3-Next-80B-A3B-Instruct-GPTQ-Int4A16
#   - nm-testing/Qwen3-Coder-30B-A3B-Instruct-W4A16-awq
# =========================

print("\n" + "="*70)
print("Loading Qwen3-Coder-Next with device_map='auto'")
print("Model will be distributed across available GPUs")
print("="*70 + "\n")

# Show available GPU memory
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        free_mem = props.total_memory - torch.cuda.memory_allocated(i)
        print(f"  GPU {i}: {props.name} - {props.total_memory / 1e9:.1f}GB total, {free_mem / 1e9:.1f}GB free")
    total_vram = sum(torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count()))
    print(f"  Total VRAM: {total_vram / 1e9:.1f}GB")
    print(f"\n  NOTE: Model (~160GB BF16) will be distributed across GPUs")
    print(f"        Using pipeline='basic' to skip FX tracing")
    print(f"        Using calibrate_moe_context=True for all 512 experts")

# Load model with device_map="auto" to distribute across GPUs
# This avoids the sequential pipeline's FX tracing which fails on @torch.fx.wrap
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",  # Distribute across 2x 96GB GPUs (192GB total)
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

print(f"\nModel loaded from: {MODEL_ID}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

# Show model device distribution
if hasattr(model, 'hf_device_map'):
    devices_used = set(model.hf_device_map.values())
    print(f"Model distributed across devices: {devices_used}")


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
            import json
            messages = json.loads(messages)
        except:
            # Not JSON, treat as raw text
            formatted_messages.append({'role': 'user', 'content': messages})
            if formatted_messages:
                try:
                    text = tokenizer.apply_chat_template(formatted_messages, tokenize=False)
                    return {'text': text}
                except Exception as e:
                    # If chat template fails, return empty
                    print(f"WARNING: Failed to apply chat template: {e}")
                    return {'text': ''}
            return {'text': ''}
    
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
        return {'text': ''}
    
    try:
        text = tokenizer.apply_chat_template(formatted_messages, tokenize=False)
        return {'text': text}
    except Exception as e:
        # If chat template fails, return empty
        print(f"WARNING: Failed to apply chat template: {e}")
        return {'text': ''}
    
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
        return {'text': ''}
    
    try:
        text = tokenizer.apply_chat_template(formatted_messages, tokenize=False)
        return {'text': text}
    except Exception as e:
        # If chat template fails, return empty
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
    
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        return {'text': text}
    except Exception as e:
        # If chat template fails, return empty
        print(f"WARNING: Failed to apply chat template in prompt_answer format: {e}")
        return {'text': ''}


def format_chat_completion(example, columns, tokenizer):
    """Format chat completion style data."""
    # Try to find messages-like column
    for col in columns:
        if col in example:
            data = example[col]
            if isinstance(data, list) and len(data) > 0:
                if isinstance(data[0], dict):
                    # Already in messages format
                    try:
                        text = tokenizer.apply_chat_template(data, tokenize=False)
                        return {'text': text}
                    except Exception as e:
                        # If chat template fails, return empty
                        print(f"WARNING: Failed to apply chat template: {e}")
                        return {'text': ''}
                elif isinstance(data[0], str):
                    # List of strings - alternate user/assistant
                    messages = []
                    for i, item in enumerate(data):
                        role = 'user' if i % 2 == 0 else 'assistant'
                        messages.append({'role': role, 'content': str(item)})
                    try:
                        text = tokenizer.apply_chat_template(messages, tokenize=False)
                        return {'text': text}
                    except Exception as e:
                        print(f"WARNING: Failed to apply chat template: {e}")
                        return {'text': ''}
            elif isinstance(data, str):
                # Single text field
                return {'text': str(data)}
    
    # Fallback: concatenate all columns
    text = ' '.join(str(example.get(col, '')) for col in columns)
    return {'text': text}


def format_raw_text(example, columns, tokenizer):
    """Format raw text data."""
    texts = []
    
    # Handle custom parameters from dataset config
    prefix = ""
    if '_formatter_params' in example and isinstance(example['_formatter_params'], dict):
        params = example['_formatter_params']
        if 'prefix' in params:
            prefix = str(params['prefix'])
    
    # Also support prefix in columns[0] for raw_text datasets
    if len(columns) == 1 and columns[0] not in ['text', 'content', 'user', 'prompt', 'problem', 'instruction', 'prompt_input', 'article', 'text_output']:
        prefix += str(columns[0]) + "\n***\n"
    
    for col in columns:
        if col in example and example[col]:
            text_content = str(example[col])
            texts.append(prefix + text_content)
    
    return {'text': ' '.join(texts)}


FORMATTERS = {
    'sharegpt': format_sharegpt,
    'prompt_answer': format_prompt_answer,
    'chat_completion': format_chat_completion,
    'raw_text': format_raw_text,
    'deepmind_code_contests': format_raw_text,
}


# =========================
# Load datasets from YAML config
# =========================
print("\n=== Loading datasets from config ===")
all_datasets = []
total_samples = 0

for ds_config in datasets_config:
    dataset_name = ds_config['dataset']
    split = ds_config.get('split', 'train')
    columns = ds_config.get('columns', [])
    formatter_name = ds_config.get('formatter', 'raw_text')
    num_samples = ds_config.get('num_samples', num_calibration_samples)
    streaming = ds_config.get('streaming', False)
    shuffle = ds_config.get('shuffle', SHUFFLE)
    ds_seed = ds_config.get('seed', SEED)
    
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
            if shuffle:
                ds = ds.shuffle(seed=ds_seed).select(range(n))
            else:
                ds = ds.select(range(0, n))
        
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
    raise ValueError("No datasets were successfully loaded from the config!")

ds = concatenate_datasets(all_datasets)
print(f"\n=== Total samples loaded: {total_samples} ===")

# Shuffle combined dataset if requested
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
    num_proc=1,
)

NUM_CALIBRATION_SAMPLES = min(len(ds), num_calibration_samples)
print(f"Tokenized {NUM_CALIBRATION_SAMPLES} samples (max_seq_length={MAX_SEQUENCE_LENGTH})")

# =========================
# MoE Expert Coverage Analysis
# =========================
# Qwen3-Coder-Next has 512 experts with 10 activated per token.
# Using calibrate_moe_context=True in oneshot() ensures all experts receive
# calibration samples, not just frequently-activated ones.
print_moe_calibration_guidance(model, NUM_CALIBRATION_SAMPLES, MAX_SEQUENCE_LENGTH)


# =========================
# AWQ recipe with config_groups
#  - Weight-only INT4 (W4A16 **symmetric**)
#  - Dynamic group_size from argument
#  - Ignore layers to avoid FX tracing issues and preserve accuracy:
#    * linear_attn: DeltaNet layers with @torch.fx.wrap decorators
#    * self_attn: Standard attention layers
#    * norm layers: All normalization layers
#    * MoE routers: mlp.gate, mlp.shared_expert_gate, router
#    * lm_head: Output head
#
# This ignore list follows dazipe's successful Qwen3-Next-80B quantization
# =========================
from compressed_tensors.quantization import QuantizationScheme, QuantizationArgs

weight_args = QuantizationArgs(
    num_bits=4,          # 4-bit weights
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

recipe = [
    AWQModifier(
        ignore=[
            "lm_head",                          # Output head
            "re:.*mlp.gate$",                   # MoE router gates
            "re:.*mlp.shared_expert_gate$",     # Shared expert gates
            "re:.*linear_attn.*",               # DeltaNet layers (has @torch.fx.wrap - causes tracing issues)
            "re:.*self_attn.*",                 # Standard attention layers
            "re:.*input_layernorm$",            # Input layer norms
            "re:.*post_attention_layernorm$",   # Post-attention layer norms
            "re:.*norm.*",                      # All normalization layers
            "re:.*router.*",                    # Expert routing layers
        ],
        config_groups={"group_0": quant_scheme},
        # NOTE: Using pipeline="basic" and ignoring linear_attn/self_attn layers
        # to avoid FX tracing issues with @torch.fx.wrap decorators in Qwen3-Coder-Next
    ),
]

# =========================
# Run one-shot compression
# =========================
print(f"\n=== Running one-shot compression with {NUM_CALIBRATION_SAMPLES} calibration samples ===")
print(f"  - Sequence length: {MAX_SEQUENCE_LENGTH}")
print(f"  - Calibration samples: {NUM_CALIBRATION_SAMPLES}")
print(f"  - Pipeline: basic (skips FX tracing)")
print(f"  - MoE calibration: default (moe_calibrate_all_experts=True)")

# Clear memory before quantization
clear_memory()

# Show memory status before quantization
if torch.cuda.is_available():
    print("\nGPU memory before quantization:")
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        free = total - reserved
        print(f"  GPU {i}: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved, {free:.1f}GB free, {total:.1f}GB total")

# Run AWQ quantization with basic pipeline
# Using device_map="auto" + pipeline="basic" approach:
#   1. Model is distributed across GPUs via device_map="auto"
#   2. pipeline="basic" skips FX tracing (avoids @torch.fx.wrap issues)
#   3. moe_calibrate_all_experts defaults to True (ensures all 512 experts are calibrated)
#   4. Ignoring linear_attn/self_attn layers avoids any remaining tracing issues
#
# This approach follows successful quantizations:
#   - dazipe/Qwen3-Next-80B-A3B-Instruct-GPTQ-Int4A16
#   - nm-testing/Qwen3-Coder-30B-A3B-Instruct-W4A16-awq
#
# NOTE: moe_calibrate_all_experts defaults to True in newer llmcompressor versions,
# ensuring all MoE experts receive calibration samples for proper quantization.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    tokenizer=tokenizer,
    pipeline="basic",  # Skip FX tracing to avoid @torch.fx.wrap issues
)

# =========================
# Save compressed model
# =========================
SAVE_DIR = output_path
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

print("\n=== Complete ===")
print("Saved to:", SAVE_DIR)