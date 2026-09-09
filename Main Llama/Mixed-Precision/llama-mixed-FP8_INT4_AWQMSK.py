import argparse
import json
import yaml

from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.utils import dispatch_for_generation

# =========================
# Parse Command-Line Arguments
# =========================
parser = argparse.ArgumentParser(
    description="Run Mixed Precision FP8+INT4 AWQ quantization on Llama model."
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
    help="Group size for W4A16 quantization (e.g., 32, 64, 128)."
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

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)


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
        if not messages:
            return []
        last_role = messages[-1].get("role", "user")
        if last_role != "assistant":
            return [0] * max_seq_length

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
        return [1] * max_seq_length


# =========================
# Load datasets from YAML recipe
# =========================
print("\n=== Loading datasets from recipe ===")
all_datasets = []
total_samples = 0
total_requested = 0
total_lost = 0

for ds_config in datasets_config:
    dataset_name = ds_config['dataset']
    split = ds_config.get('split', 'train')
    columns = ds_config.get('columns', [])
    formatter_name = ds_config.get('formatter', 'raw_text')
    num_samples = ds_config.get('num_samples', 10)
    streaming = ds_config.get('streaming', False)
    
    total_requested += num_samples
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
            samples_available = num_samples  # Can't know for streaming
        else:
            ds = load_dataset(dataset_name, split=split)
            samples_available = len(ds)
            # Sample from dataset
            n = min(num_samples, len(ds))
            if n < num_samples:
                print(f"    [!] Only {samples_available} available, requested {num_samples}")
            ds = ds.shuffle(seed=SEED).select(range(n))
        
        samples_before_format = len(ds)
        
        # Get formatter function
        formatter_fn = FORMATTERS.get(formatter_name, format_raw_text)
        
        # Apply formatter
        ds = ds.map(
            lambda x: formatter_fn(x, columns, tokenizer),
            remove_columns=ds.column_names,
            num_proc=1,  # Use single proc to avoid tokenizer issues
        )
        
        samples_before_filter = len(ds)
        
        # Filter out empty texts
        ds = ds.filter(lambda x: len(x.get('text', '')) > 0)
        
        samples_after_filter = len(ds)
        lost = num_samples - samples_after_filter
        
        if samples_after_filter < samples_before_filter:
            print(f"    [!] Filtered out {samples_before_filter - samples_after_filter} empty samples")
        
        all_datasets.append(ds)
        total_samples += len(ds)
        total_lost += lost
        
        status = "✓" if samples_after_filter == num_samples else "⚠"
        print(f"    -> {status} Loaded {samples_after_filter}/{num_samples} samples")
        
    except Exception as e:
        print(f"    -> ✗ FAILED to load {dataset_name}: {e}")
        total_lost += num_samples
        continue

print(f"\n=== Dataset Loading Summary ===")
print(f"  Requested: {total_requested} samples")
print(f"  Loaded:    {total_samples} samples")
print(f"  Lost:      {total_lost} samples ({100*total_lost/total_requested:.1f}%)")

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
                loss_masks.append([1] * seq_len)
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
    num_proc=1 if use_loss_mask else 4,
)

NUM_CALIBRATION_SAMPLES = len(ds)
print(f"Tokenized {NUM_CALIBRATION_SAMPLES} samples (max_seq_length={MAX_SEQUENCE_LENGTH}, use_loss_mask={use_loss_mask})")


# =========================
# Mixed Precision Quantization Recipe
#   * quantize self_attn layers to FP8_BLOCK (attention: k, q, o, v_proj)
#   * quantize mlp layers to W4A16 with AWQ (MLP: down, gate, up_proj)
# =========================
quant_recipe = f"""
quant_stage:
  quant_modifiers:
    QuantizationModifier:
      targets: ["re:.*(k|q|o|v)_proj$"]
      scheme: FP8_BLOCK
    AWQModifier:
      config_groups:
        group_0:
          targets: ["re:.*(down|gate|up)_proj$"]
          weights:
            num_bits: 4
            type: int
            symmetric: true
            group_size: {group_size}
            strategy: group
            dynamic: false
            observer: minmax

      # Layers to exclude from quantization
      ignore:
        - "lm_head"

      # Scaling options
      duo_scaling: true

      mappings:
        - smooth_layer: re:.*post_attention_layernorm$
          balance_layers: ["re:.*gate_proj$", "re:.*up_proj$"]
        - smooth_layer: re:.*up_proj$
          balance_layers: ["re:.*down_proj$"]
"""

# =========================
# Run one-shot compression
# =========================
print("\n=== Running one-shot compression ===")
oneshot_kwargs = dict(
    model=model,
    dataset=ds,
    recipe=quant_recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    tokenizer=tokenizer,
    use_loss_mask=use_loss_mask,
)
if use_loss_mask:
    oneshot_kwargs["pipeline"] = "sequential"
oneshot(**oneshot_kwargs)

# =========================
# Quick sanity generation
# =========================
#print("\n\n========== SAMPLE GENERATION ==============")
#dispatch_for_generation(model)
#input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to("cuda")
#output = model.generate(input_ids, max_new_tokens=100)
#print(tokenizer.decode(output[0]))
#print("==========================================\n\n")

# =========================
# Save compressed model
# =========================
SAVE_DIR = output_path
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

print("\n=== Complete ===")
print("Saved to:", SAVE_DIR)