
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM
from huggingface_hub import hf_hub_download
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.utils import dispatch_for_generation

# =========================
# Load ENV Variables
# =========================
from pathlib import Path
import os
import shutil
from dotenv import load_dotenv

# Load the .env that sits next to this script (works regardless of where you run it)
load_dotenv(Path(__file__).with_name(".env"))

def require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Missing environment variable: {name}")
    return val


# =========================
# Model
# =========================
MODEL_ID = require_env("SRC_DIR")

def get_tokenizer(model_id: str):
    # Devstral uses mistral-common tokenizer via tekken.json
    # Check if model_id is a local path (contains path separators or is absolute)
    # Hugging Face repo IDs are in format "namespace/repo_name"
    is_local_path = os.path.sep in model_id or (os.path.altsep and os.path.altsep in model_id) or os.path.isabs(model_id)
    
    if is_local_path:
        # Local path - construct tekken.json path directly
        tekken_path = os.path.join(model_id, "tekken.json")
        if not os.path.exists(tekken_path):
            raise FileNotFoundError(f"tekken.json not found at {tekken_path}")
    else:
        # Hugging Face repo ID - download from hub
        tekken_path = hf_hub_download(model_id, "tekken.json")
    tokenizer = MistralTokenizer.from_file(tekken_path)
    return tokenizer

# Disable gradients for quantization
torch.set_grad_enabled(False)

print("Loading model…")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",  # let accelerate/transformers shard across GPUs
)

tokenizer = get_tokenizer(MODEL_ID)


# =========================
# Calibration data config
# =========================
NUM_CALIBRATION_SAMPLES = 512      # Adjust as needed
MAX_SEQUENCE_LENGTH = 2048

DATASET_ID = "neuralmagic/LLM_compression_calibration"
DATASET_SPLIT = "train"

# =========================
# Load + sample neuralmagic calibration dataset
# =========================
ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)

# Random, reproducible subset of N samples
n = min(NUM_CALIBRATION_SAMPLES, len(ds))
ds = ds.shuffle(seed=42).select(range(n))


# =========================
# Preprocess (batch-aware)
# =========================
def format_messages_to_text(messages):
    # Convert messages list to formatted text string
    # Messages format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    formatted_parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role and content:
            formatted_parts.append(f"{role}: {content}")
    return "\n".join(formatted_parts)

def preprocess(batch):
    # The neuralmagic dataset has a 'messages' field with pre-formatted conversations
    messages_list = batch["messages"]  # list[list[dict]]
    # MistralTokenizer doesn't have apply_chat_template, so we format messages manually
    rendered = [
        format_messages_to_text(messages)
        for messages in messages_list
    ]
    return {"text": rendered}

# Render chat template in batches
ds = ds.map(preprocess, batched=True, num_proc=4)

# =========================
# Tokenize in batches
# =========================
def tokenize_batch(batch):
    # MistralTokenizer doesn't have encode() - use underlying tokenizer
    # Access the underlying Tekkenizer/SentencePieceTokenizer via instruct_tokenizer.tokenizer
    # The base Tokenizer class has encode(s: str, bos: bool, eos: bool) -> list[int]
    # Format: {"input_ids": [[token_ids], [token_ids], ...]}
    underlying_tokenizer = tokenizer.instruct_tokenizer.tokenizer
    tokenized = []
    for text in batch["text"]:
        # Use the underlying tokenizer's encode method with bos=True, eos=True
        ids = underlying_tokenizer.encode(text, bos=True, eos=True)
        # Truncate to MAX_SEQUENCE_LENGTH if needed
        if len(ids) > MAX_SEQUENCE_LENGTH:
            ids = ids[:MAX_SEQUENCE_LENGTH]
        tokenized.append(ids)
    return {"input_ids": tokenized}

ds = ds.map(
    tokenize_batch,
    batched=True,
    remove_columns=ds.column_names,  # drop "text" column
    num_proc=4,
)

# =========================
# Quantization recipe  (W4A16-SYM, Marlin-friendly)
# =========================
from compressed_tensors.quantization import QuantizationScheme, QuantizationArgs

weight_args = QuantizationArgs(
    num_bits=4,          # 4-bit weights
    type="int",
    symmetric=True,      # SYMMETRIC (Marlin requirement)
    strategy="group",    # group-wise quantization
    group_size=32,      # 32 groupsize (High Accuracy)
)

quant_scheme = QuantizationScheme(
    targets=["Linear"],
    weights=weight_args,
    input_activations=None,   # A16 (leave activations in FP16/BF16)
    output_activations=None,
)

recipe = [
    AWQModifier(
        ignore=["lm_head"],
        config_groups={"group_0": quant_scheme},
    ),
]


# =========================
# Run one-shot compression
# =========================
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

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
SAVE_DIR = require_env("DST_DIR")
model.save_pretrained(SAVE_DIR, save_compressed=True)
# MistralTokenizer may not support save_pretrained, so copy tekken.json if needed
try:
    tokenizer.save_pretrained(SAVE_DIR)
except AttributeError:
    # If save_pretrained not available, copy tekken.json manually
    # Check if MODEL_ID is a local path or Hugging Face repo ID
    is_local_path = os.path.sep in MODEL_ID or (os.path.altsep and os.path.altsep in MODEL_ID) or os.path.isabs(MODEL_ID)
    
    if is_local_path:
        # Local path - construct tekken.json path directly
        tekken_path = os.path.join(MODEL_ID, "tekken.json")
        if not os.path.exists(tekken_path):
            raise FileNotFoundError(f"tekken.json not found at {tekken_path}")
    else:
        # Hugging Face repo ID - download from hub
        tekken_path = hf_hub_download(MODEL_ID, "tekken.json")
    shutil.copy(tekken_path, SAVE_DIR)
