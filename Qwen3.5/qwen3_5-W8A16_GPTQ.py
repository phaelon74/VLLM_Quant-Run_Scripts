"""
Qwen3.5 W8A16 GPTQ (non-AWQ) quantization.

Use this if AWQ produces garbage output. GPTQ uses a different algorithm
(no scale smoothing) and may work better for Qwen3.5's hybrid architecture.
"""
import argparse
import json
import yaml

import torch.nn as nn
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
)

# Transformers v5 compatibility
import transformers.modeling_utils as _tmu
if not hasattr(_tmu, "TORCH_INIT_FUNCTIONS"):
    _tmu.TORCH_INIT_FUNCTIONS = {
        "uniform_": nn.init.uniform_, "normal_": nn.init.normal_,
        "trunc_normal_": nn.init.trunc_normal_, "constant_": nn.init.constant_,
        "xavier_uniform_": nn.init.xavier_uniform_, "xavier_normal_": nn.init.xavier_normal_,
        "kaiming_uniform_": nn.init.kaiming_uniform_, "kaiming_normal_": nn.init.kaiming_normal_,
    }

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

# =========================
# Parse Command-Line Arguments
# =========================
parser = argparse.ArgumentParser(
    description="Run W8A16 GPTQ quantization on Qwen3.5 (no AWQ). Use if AWQ produces garbage."
)
parser.add_argument("model_path", type=str, help="Path to the source model directory.")
parser.add_argument("output_path", type=str, help="Path to save quantized model.")
parser.add_argument("recipe_yaml", type=str, help="Path to dataset recipe YAML.")
parser.add_argument("group_size", type=int, help="Group size (e.g., 32, 64, 128).")

args = parser.parse_args()

# =========================
# Load Recipe
# =========================
with open(args.recipe_yaml, 'r') as f:
    recipe_config = yaml.safe_load(f)
calibration_config = recipe_config.get('calibration_set', {})
MAX_SEQUENCE_LENGTH = calibration_config['max_seq_length']
SHUFFLE = calibration_config.get('shuffle', True)
SEED = calibration_config.get('seed', 42)
datasets_config = calibration_config.get('datasets', [])

# =========================
# Model
# =========================
config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
model_type = getattr(config, "model_type", "")
if model_type in ("qwen3_5", "qwen3_5_moe"):
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, dtype="auto", trust_remote_code=True
    )
else:
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, dtype="auto", trust_remote_code=True
        )
    except ValueError:
        model = AutoModel.from_pretrained(
            args.model_path, dtype="auto", trust_remote_code=True
        )
tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

# =========================
# Ignore list — same as NVFP4A16 reference
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
    if any(k in name_lower for k in vision_keywords):
        ignore_list.append(mod_name)
    elif "mtp" in name_lower or "lm_head" in name_lower:
        ignore_list.append(mod_name)
    elif "linear_attn" in name_lower:
        ignore_list.append(mod_name)
    elif mod.out_features % args.group_size != 0:
        ignore_list.append(mod_name)
ignore_list = sorted(set(ignore_list))
print(f"Ignored {len(ignore_list)} linear modules")

# =========================
# Dataset
# =========================
def format_example(example, columns, tokenizer):
    if 'messages' in example and example['messages']:
        try:
            return {'text': tokenizer.apply_chat_template(
                example['messages'], tokenize=False
            )}
        except Exception:
            pass
    texts = [str(example.get(c, '')) for c in columns if c in example and example.get(c)]
    return {'text': ' '.join(texts) if texts else ''}

all_ds = []
for ds_cfg in datasets_config:
    ds = load_dataset(ds_cfg['dataset'], split=ds_cfg.get('split', 'train'))
    n = min(ds_cfg.get('num_samples', 128), len(ds))
    ds = ds.shuffle(seed=SEED).select(range(n))
    cols = ds_cfg.get('columns', ds.column_names[:1])
    ds = ds.map(lambda x: format_example(x, cols, tokenizer), remove_columns=ds.column_names)
    ds = ds.filter(lambda x: len(x.get('text', '')) > 0)
    all_ds.append(ds)
ds = concatenate_datasets(all_ds)
if SHUFFLE:
    ds = ds.shuffle(seed=SEED)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )
ds = ds.map(tokenize, batched=True, remove_columns=ds.column_names)
NUM_SAMPLES = len(ds)
print(f"Calibration samples: {NUM_SAMPLES}")

# =========================
# GPTQ recipe (no AWQ)
# =========================
from compressed_tensors.quantization import QuantizationScheme, QuantizationArgs

quant_scheme = QuantizationScheme(
    targets=["Linear"],
    weights=QuantizationArgs(
        num_bits=8,
        type="int",
        symmetric=True,
        strategy="group",
        group_size=args.group_size,
    ),
    input_activations=None,
    output_activations=None,
)

recipe = [
    GPTQModifier(
        ignore=ignore_list,
        config_groups={"group_0": quant_scheme},
        dampening_frac=0.1,
        actorder="static",
        block_size=128,
    ),
]

# =========================
# Run
# =========================
print("\n=== Running GPTQ quantization ===")
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_SAMPLES,
    tokenizer=tokenizer,
)

model.save_pretrained(args.output_path, save_compressed=True)
tokenizer.save_pretrained(args.output_path)
print("\n=== Complete ===")
print("Saved to:", args.output_path)
