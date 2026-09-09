from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.awq.mappings import AWQMapping
from llmcompressor.utils import dispatch_for_generation

# =========================
# Load ENV Variables
# =========================
from pathlib import Path
import os
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

# Load model with trust_remote_code=True for Command-A models (extended vocabulary, custom layers)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

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
def preprocess(batch):
    # The neuralmagic dataset has a 'messages' field with pre-formatted conversations
    messages_list = batch["messages"]  # list[list[dict]]
    rendered = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
        )
        for messages in messages_list
    ]
    return {"text": rendered}

# Render chat template in batches
ds = ds.map(preprocess, batched=True, num_proc=4)

# =========================
# Tokenize in batches
# =========================
ds = ds.map(
    lambda batch: tokenizer(
        batch["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    ),
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
    group_size=32,      # 32 groupsize (Marlin standard)
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
        mappings=[
            # Command-A specific AWQ mappings for optimal quantization
            AWQMapping(
                "re:.*input_layernorm$",
                [
                    "re:.*self_attn.q_proj$",
                    "re:.*self_attn.k_proj$",
                    "re:.*self_attn.v_proj$",
                    "re:.*mlp.gate_proj$",
                    "re:.*mlp.up_proj$",
                ],
            ),
            # Removed redundant v_proj -> o_proj mapping (v_proj already handled in first mapping)
            AWQMapping("re:.*up_proj$", ["re:.*down_proj$"]),
        ],
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
    tokenizer=tokenizer,
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
tokenizer.save_pretrained(SAVE_DIR)
