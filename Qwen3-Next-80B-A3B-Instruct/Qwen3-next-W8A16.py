from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier

# =========================
# Load ENV Variables
# =========================
from pathlib import Path
import os
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
# Model (Qwen3-Next-80B-A3B-Instruct)
# =========================
MODEL_ID = require_env("SRC_DIR")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# =========================
# Calibration data (WikiText)
# =========================
NUM_CALIBRATION_SAMPLES = 64
MAX_SEQUENCE_LENGTH = 1024

DATASET_ID = "wikitext"
DATASET_NAME = "wikitext-2-raw-v1"
DATASET_SPLIT = "validation"

ds = load_dataset(DATASET_ID, DATASET_NAME, split=DATASET_SPLIT)
ds = ds.filter(lambda ex: ex.get("text", "").strip() != "")

n = min(NUM_CALIBRATION_SAMPLES, len(ds))
ds = ds.shuffle(seed=42).select(range(n))

# Render to chat-style text (batch)
def preprocess(batch):
    rendered = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": t}],
            tokenize=False,
        )
        for t in batch["text"]
    ]
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
# AWQ recipe with config_groups
#  - Weight-only INT4 (W4A16 **symmetric**)
#  - group_size: 128
#  - IMPORTANT: do NOT ignore FFN gate/up/down proj (quantize them)
#  - Keep MoE router-related linears and output head unquantized
# =========================
router_keywords = (
    "router",
    "expert_choice",
    "dispatch",
    "scores",
    "route",
    "topk",
    "switch",
)

def _should_ignore_module(module_name: str) -> bool:
    name = module_name.lower()
    # Keep FFN projections quantized
    if any(x in name for x in ("gate_proj", "up_proj", "down_proj")):
        return False
    # Ignore routing-related linears
    if any(k in name for k in router_keywords):
        return True
    # Do not quantize final output head
    if name.endswith("lm_head") or name == "lm_head":
        return True
    return False

# Build ignore list dynamically based on module names in the loaded model
moe_ignores = []
for mod_name, mod in model.named_modules():
    if isinstance(mod, nn.Linear) and _should_ignore_module(mod_name):
        moe_ignores.append(mod_name)
moe_ignores = sorted(set(moe_ignores + ["lm_head"]))

recipe = [
    AWQModifier(
        targets=["Linear"],        # quantize all Linear layers uniformly
        ignore=moe_ignores,
        config_groups={
            "group_0": {
                # Uniformly quantize Linear layers; ignore list handles routers/lm_head
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 8,
                    "type": "int",
                    "symmetric": True,   # W4A16 (symmetric)
                    "strategy": "group",
                    "group_size": 128,    # robust to non power-of-two channel dims
                    "dynamic": False,
                },
            },
        },
        # Optional mappings can be added, but not required
    ),
]

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
    calibrate_moe_context=True,
#    output_dir=SAVE_DIR,
)

# (Optional redundant save)
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

print("Saved to:", SAVE_DIR)
