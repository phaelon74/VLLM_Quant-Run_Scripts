"""
NVFP4 Quantization Script

Supports Llama, Mistral, and other decoder-based architectures.
NVFP4: FP4 weights + FP4 activations, per-group-16 (fixed), optimized for Blackwell GPUs.
"""

import argparse
import yaml
import torch

from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.utils import update_fused_layer_weight_global_scales

# =========================
# Parse Command-Line Arguments
# =========================
parser = argparse.ArgumentParser(
    description="Run NVFP4 quantization on Llama model (3, 3.1, 3.3, etc.)."
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

args = parser.parse_args()
model_path = args.model_path
output_path = args.output_path
recipe_yaml_path = args.recipe_yaml

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
num_calibration_samples = calibration_config.get('num_samples', None)  # Optional cap
datasets_config = calibration_config.get('datasets', [])

print(f"Loaded recipe from: {recipe_yaml_path}")
print(f"  - max_seq_length: {MAX_SEQUENCE_LENGTH}")
print(f"  - shuffle: {SHUFFLE}")
print(f"  - seed: {SEED}")
print(f"  - datasets to load: {len(datasets_config)}")

# =========================
# Model
# =========================
MODEL_ID = model_path

# Load config to check model type
config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
print(f"Model config type: {type(config).__name__}")

# Load in BF16 - NVFP4 calibration prefers BF16/FP16
# device_map=None enables sequential processing for large models
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=None,
    )
except ValueError as e:
    print(f"AutoModelForCausalLM failed (likely custom model): {e}")
    print("Attempting AutoModel with trust_remote_code=True...")
    from transformers import AutoModel
    model = AutoModel.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=None,
    )
    print("Successfully loaded model with AutoModel")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)


def detect_sequential_targets(model):
    """Detect decoder layer type for memory-efficient calibration."""
    candidates = [
        "LlamaDecoderLayer",
        "MistralDecoderLayer",
        "Qwen2DecoderLayer",
        "Phi3DecoderLayer",
        "GemmaDecoderLayer",
        "Gemma2DecoderLayer",
        "CohereDecoderLayer",
        "Starcoder2DecoderLayer",
    ]
    for name, module in model.named_modules():
        t = type(module).__name__
        if t in candidates:
            return [t]
    # Fallback: match any module whose name ends with "DecoderLayer"
    for name, module in model.named_modules():
        t = type(module).__name__
        if t.endswith("DecoderLayer"):
            return [t]
    return None


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
        except Exception:
            # Not JSON, treat as raw text
            formatted_messages.append({'role': 'user', 'content': messages})
            if formatted_messages:
                text = tokenizer.apply_chat_template(formatted_messages, tokenize=False)
                return {'text': text}
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
    except Exception:
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
    
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {'text': text}


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
                    return {'text': text}
                elif isinstance(data[0], str):
                    # List of strings - alternate user/assistant
                    messages = []
                    for i, item in enumerate(data):
                        role = 'user' if i % 2 == 0 else 'assistant'
                        messages.append({'role': role, 'content': str(item)})
                    text = tokenizer.apply_chat_template(messages, tokenize=False)
                    return {'text': text}
            elif isinstance(data, str):
                # Single text field
                return {'text': str(data)}
    
    # Fallback: concatenate all columns
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

NUM_CALIBRATION_SAMPLES = min(len(ds), num_calibration_samples) if num_calibration_samples else len(ds)
print(f"Tokenized {NUM_CALIBRATION_SAMPLES} samples (max_seq_length={MAX_SEQUENCE_LENGTH})")


# =========================
# NVFP4 recipe
# =========================
# NVFP4: FP4 weights + FP4 activations, per-group-16 (fixed), optimized for Blackwell.
# Quantizes all Linear layers except lm_head.
recipe = QuantizationModifier(
    targets="Linear",
    scheme="NVFP4",
    ignore=["lm_head"],
)

# =========================
# Run one-shot compression
# =========================
print("\n=== Running one-shot compression ===")
print(f"  - Scheme: NVFP4 (FP4 weights + FP4 activations, per-group-16)")
print(f"  - Calibration samples: {NUM_CALIBRATION_SAMPLES}")

oneshot_kwargs = {
    "model": model,
    "dataset": ds,
    "recipe": recipe,
    "max_seq_length": MAX_SEQUENCE_LENGTH,
    "num_calibration_samples": NUM_CALIBRATION_SAMPLES,
    "tokenizer": tokenizer,
    "tracing_ignore": [
        "_update_causal_mask",
        "create_causal_mask",
        "create_sliding_window_causal_mask",
        "make_causal_mask",
        "get_causal_mask",
        "mask_interface",
        "mask_function",
        "_prepare_4d_causal_attention_mask",
        "_prepare_4d_causal_attention_mask_with_cache_position",
    ],
}

sequential_targets = detect_sequential_targets(model)
if sequential_targets:
    oneshot_kwargs["sequential_targets"] = sequential_targets
    print(f"  - Sequential targets: {sequential_targets} (memory-efficient calibration)")

oneshot(**oneshot_kwargs)

# Unify global scales for fused QKV and gate/up layers (vLLM requirement)
update_fused_layer_weight_global_scales(model)

# =========================
# Save compressed model
# =========================
SAVE_DIR = output_path
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

print("\n=== Complete ===")
print("Saved to:", SAVE_DIR)
print("NVFP4 quantization ready for inference on Blackwell GPUs (SM 9.0+).")
