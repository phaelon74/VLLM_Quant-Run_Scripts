import os
from pathlib import Path

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier


# =========================
# Load ENV Variables
# =========================
from dotenv import load_dotenv

# Load the .env that sits next to this script (works regardless of where you run it)
load_dotenv(Path(__file__).with_name(".env"))

def require_env(key: str) -> str:
    val = os.getenv(key)
    if not val or not val.strip():
        raise RuntimeError(f"Missing environment variable: {key}")
    return val.strip()

SRC_DIR = require_env("SRC_DIR")
DST_DIR = require_env("DST_DIR")

# =========================
# Model (gpt-oss-20b)
# =========================
MODEL_ID = require_env("SRC_DIR")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# Debug: Print layer names to verify structure and find all quantizable layers
print("\n=== Debug: Checking model layer structure ===")
all_layer_names = []
def get_layer_names(module, prefix=""):
    """Recursively get all layer names"""
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        all_layer_names.append(full_name)
        get_layer_names(child, full_name)

get_layer_names(model)
print(f"Total layers found: {len(all_layer_names)}")

# Check for attention and MLP layers specifically
attention_layers = [n for n in all_layer_names if 'self_attn' in n and ('q_proj' in n or 'k_proj' in n or 'v_proj' in n or 'o_proj' in n)]
mlp_layers = [n for n in all_layer_names if 'mlp' in n]
router_layers = [n for n in all_layer_names if 'router' in n]
expert_layers = [n for n in all_layer_names if 'experts' in n]

print(f"\nFound {len(attention_layers)} attention projection layers")
print(f"Found {len(mlp_layers)} MLP-related layers")
print(f"Found {len(router_layers)} router layers")
print(f"Found {len(expert_layers)} expert-related layers")

# Check expert structure - this is critical for MoE
if expert_layers:
    print("\nExpert structure sample:")
    for name in expert_layers[:10]:
        print(f"  {name}")
    # Check what's inside experts - this is critical for quantization
    if hasattr(model.model.layers[0].mlp, 'experts'):
        experts_module = model.model.layers[0].mlp.experts
        print(f"\nLayer 0 MLP experts structure:")
        print(f"  Type: {type(experts_module).__name__}")
        print(f"  Children:")
        for name, child in experts_module.named_children():
            print(f"    {name}: {type(child).__name__}")
            # Check if this child has Linear layers
            for subname, submodule in child.named_modules():
                if 'Linear' in str(type(submodule)):
                    print(f"      {name}.{subname}: {type(submodule).__name__}")
                    # Get full path
                    full_path = f"model.layers.0.mlp.experts.{name}.{subname}"
                    print(f"        Full path: {full_path}")
                    break

print("=" * 50 + "\n")

# Verify tokenizer has harmony chat template, set if missing
# Harmony format: <|startoftext|><|message|>role: content<|return|>...
if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
    # Set harmony chat template for gpt-oss-20b
    # Format: <|startoftext|><|message|>role: content<|return|>...
    harmony_template = "<|startoftext|>{% for message in messages %}<|message|>{{ message['role'] }}: {{ message['content'] }}<|return|>{% endfor %}"
    tokenizer.chat_template = harmony_template
    print("✓ Set harmony chat template on tokenizer (was missing)")
else:
    print(f"✓ Tokenizer already has chat template: {type(tokenizer.chat_template).__name__}")

# =========================
# Calibration data (Neural Magic LLM Compression Calibration)
# =========================
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

DATASET_ID = "neuralmagic/LLM_compression_calibration"
DATASET_SPLIT = "train"

ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)

n = min(NUM_CALIBRATION_SAMPLES, len(ds))
ds = ds.shuffle(seed=42).select(range(n))

# Render messages to chat-style text (batch)
# The neuralmagic dataset has "messages" field with user/assistant roles
# For gpt-oss-20b, we use harmony format which uses special tokens
def preprocess(batch):
    rendered = []
    for messages in batch["messages"]:
        # Try to use chat template if available
        try:
            if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            else:
                raise AttributeError("No chat template")
        except (AttributeError, ValueError):
            # Fallback: Use harmony-like format for gpt-oss-20b
            # Harmony format uses special tokens: <|startoftext|>, <|message|>, <|return|>, etc.
            # Basic harmony format: <|startoftext|><|message|>role: content<|return|>...
            formatted_parts = ["<|startoftext|>"]
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if content:
                    # Map roles to harmony format (user/assistant are common)
                    # Harmony uses <|message|>role: content<|return|>
                    formatted_parts.append(f"<|message|>{role}: {content}<|return|>")
            text = "".join(formatted_parts)
        rendered.append(text)
    return {"text": rendered}

ds = ds.map(preprocess, batched=True, num_proc=4)

# Tokenize in batches
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

# =========================
# AWQ recipe with config_groups for gpt-oss-20b
#  - Weight-only INT8 (W8A16 **symmetric**)
#  - group_size: 64 (must divide hidden_size=2880, 2880/64=45)
#  - MINIMAL skip: Only routers, embeddings, and output head
#  - QUANTIZE: All attention layers + MLP expert weights (~95% of model)
#  - Expected size: ~10GB (50% reduction from 20B model with W8A16)
#  - Model has 24 layers (num_hidden_layers: 24)
#  - MoE architecture with 32 experts per layer
# =========================

# Build ignore list for gpt-oss-20b
# MINIMAL skip list: Only skip what's absolutely critical
# QUANTIZE: Everything except routers, embeddings, and output head
# This should result in ~50% size reduction (W8A16 on 20B model = ~10GB)
#
# Model has 24 layers (0-23), MoE with 32 experts per layer
# Start minimal - we can add first/last layer skips if quality degrades

# Ignore list for gpt-oss-20b
# Following best practices: skip embeddings, routers, and output head
# QUANTIZE: All attention layers + MLP expert weights (~95% of model)

gpt_oss_ignores = [
    # Embedding layer (always skip - critical for input representation)
    "model.embed_tokens",
    
    # Output head (always skip - critical for output quality)
    "lm_head",
    
    # All MLP router layers (critical for MoE routing, must stay unquantized)
    # Router determines which experts to activate - quantization breaks routing
]

# Add router layers for all 24 layers (routers must stay unquantized)
for layer_idx in range(24):
    gpt_oss_ignores.append(f"model.layers.{layer_idx}.mlp.router")

print(f"\n=== Ignore list ({len(gpt_oss_ignores)} items) ===")
print("Ignoring: embeddings, routers, and output head")
print("First 10 ignored patterns:")
for item in gpt_oss_ignores[:10]:
    print(f"  {item}")
print("=" * 50 + "\n")

# Debug: Check which layers would match the ignore patterns
print("=== Checking ignore pattern matching ===")
import re
matched_layers = []
for ignore_pattern in gpt_oss_ignores:
    # Check if pattern matches any layer names
    pattern_re = ignore_pattern.replace(".", r"\.").replace("*", ".*")
    for layer_name in all_layer_names:
        if re.match(pattern_re, layer_name) or layer_name == ignore_pattern:
            if layer_name not in matched_layers:
                matched_layers.append(layer_name)
print(f"Found {len(matched_layers)} layers matching ignore patterns")
if len(matched_layers) > 0:
    print("Sample matched layers:")
    for layer in matched_layers[:10]:
        print(f"  {layer}")
print("=" * 50 + "\n")

# AWQModifier for gpt-oss-20b
# Issue: Default mappings don't work for MoE architecture with experts
# Try: Don't provide mappings parameter - let AWQModifier handle it differently
# Or: Use a different quantization approach that works with MoE

recipe = [
    AWQModifier(
        ignore=gpt_oss_ignores,
        # Don't provide mappings - let AWQModifier infer or use targets directly
        # The issue might be that mappings are required for AWQ to work
        config_groups={
            "group_0": {
                "targets": ["Linear"],  # Target all Linear layers (attention + expert MLPs)
                "weights": {
                    "num_bits": 8,
                    "type": "int",
                    "symmetric": True,   # W8A16 (symmetric)
                    "strategy": "group",
                    # group_size must divide hidden_size (2880)
                    # Options: 64 (2880/64=45), 96 (2880/96=30), 160 (2880/160=18), 192 (2880/192=15)
                    # Smaller groups = finer quantization = potentially better quality
                    # 64 is a good balance; 96/160/192 are closer to 128 but may have slightly more error
                    "group_size": 64,    # Using 64 for best quality (finer quantization)
                    "dynamic": False,
                },
            },
        },
    ),
]

# Alternative: If AWQModifier doesn't work, we might need to use a different modifier
# or manually quantize layers. But let's try this first.

# =========================
# Quantize + save (writes quantization_config for vLLM)
# =========================
SAVE_DIR = require_env("DST_DIR")

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
#    output_dir=SAVE_DIR,  # Save compressed model via oneshot
)

# Debug: Check if model was actually quantized
print("\n=== Checking quantization status ===")
quantized_layers = 0
total_layers = 0
for name, module in model.named_modules():
    if hasattr(module, 'weight') and module.weight is not None:
        total_layers += 1
        # Check if weight is quantized (quantized weights have different dtype or are in different format)
        if hasattr(module.weight, 'dtype'):
            weight_dtype = str(module.weight.dtype)
            # Quantized weights might be int8, or have quantization metadata
            if 'int' in weight_dtype.lower() or hasattr(module, 'quantization_config'):
                quantized_layers += 1
                if quantized_layers <= 5:  # Show first 5 quantized layers
                    print(f"  Quantized: {name} ({weight_dtype})")

print(f"\nFound {quantized_layers} quantized layers out of {total_layers} total layers with weights")
print("=" * 50 + "\n")

# Fix generation config validation issue before saving
if hasattr(model, 'generation_config') and model.generation_config is not None:
    # If temperature is set but do_sample is False, either enable do_sample or remove temperature
    if hasattr(model.generation_config, 'temperature') and model.generation_config.temperature is not None:
        if not getattr(model.generation_config, 'do_sample', False):
            # Set do_sample=True to make temperature valid, or remove temperature
            model.generation_config.do_sample = True

# Save compressed model
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

print("Saved to:", SAVE_DIR)

