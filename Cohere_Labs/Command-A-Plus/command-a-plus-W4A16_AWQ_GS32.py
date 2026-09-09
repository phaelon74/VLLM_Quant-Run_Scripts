"""
Command A+ (CohereLabs/command-a-plus-05-2026-bf16) W4A16 AWQ — group size 32

  - AWQModifier with symmetric int4 group_size=32 (Marlin-friendly W4A16)
  - Calibration from Recipes/Datasets/General_reasoning.yaml (or any compatible recipe)
  - AutoModelForImageTextToText + AutoProcessor for vLLM Cohere2Vision paths
  - MoE: calibrate_moe_context + cohere2_moe_calibration (all 128 experts per layer)
  - Memory: model on CPU (device_map=None), pipeline=sequential loads one decoder
    layer at a time to GPU for AWQ (required at ~218B total params)

Example:
  python command-a-plus-W4A16_AWQ_GS32.py \\
    /path/to/command-a-plus-05-2026-bf16 \\
    /path/to/out-w4a16-gs32 \\
    ../Recipes/Datasets/General_reasoning.yaml

Requires:
  pip install -U transformers llmcompressor compressed-tensors accelerate datasets pyyaml
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from compressed_tensors.offload import dispatch_model
from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
)

import cohere2_moe_calibration  # noqa: F401 — registers CalibrationCohere2MoeSparseMoeBlock

# Transformers v5 compatibility
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

from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier, AWQMapping

_DEFAULT_RECIPE = (
    Path(__file__).resolve().parent.parent
    / "Recipes"
    / "Datasets"
    / "General_reasoning.yaml"
)

# MoE AWQ: input_layernorm feeds attention + all expert/shared MLP projections
COHERE2_MOE_AWQ_MAPPINGS = [
    AWQMapping(
        "re:.*input_layernorm$",
        [
            "re:.*self_attn.q_proj$",
            "re:.*self_attn.k_proj$",
            "re:.*self_attn.v_proj$",
            "re:.*mlp.experts.*.gate_proj$",
            "re:.*mlp.experts.*.up_proj$",
            "re:.*mlp.shared_experts.gate_proj$",
            "re:.*mlp.shared_experts.up_proj$",
        ],
    ),
    AWQMapping(
        "re:.*self_attn.v_proj$",
        ["re:.*self_attn.o_proj$"],
    ),
    AWQMapping(
        "re:.*up_proj$",
        ["re:.*down_proj$"],
    ),
]

# =========================
# CLI
# =========================
parser = argparse.ArgumentParser(
    description=(
        "Command A+ W4A16 AWQ (group_size=32 default). "
        "Sequential layer-by-layer calibration on CPU-loaded model."
    )
)
parser.add_argument("model_path", type=str, help="Path to the source BF16 model directory.")
parser.add_argument("output_path", type=str, help="Path to save the quantized model.")
parser.add_argument(
    "recipe_yaml",
    type=str,
    nargs="?",
    default=str(_DEFAULT_RECIPE),
    help=f"Calibration recipe YAML (default: {_DEFAULT_RECIPE}).",
)
parser.add_argument(
    "--group-size",
    type=int,
    default=32,
    choices=(32, 64, 128),
    help="AWQ weight group size (default: 32).",
)
parser.add_argument(
    "--max-seq-length",
    type=int,
    default=None,
    help="Override calibration_set.max_seq_length from the recipe.",
)
parser.add_argument(
    "--use-loss-mask",
    action="store_true",
    default=False,
    help="AWQ loss on assistant tokens only (loss_mask in tokenized dataset).",
)
parser.add_argument(
    "--skip-sample-gen",
    action="store_true",
    help="Skip post-quantization smoke generation.",
)
args = parser.parse_args()

MODEL_ID = args.model_path
use_loss_mask = args.use_loss_mask
group_size = args.group_size

# =========================
# Recipe YAML
# =========================
with open(args.recipe_yaml, "r", encoding="utf-8") as f:
    recipe_file = yaml.safe_load(f)

calibration_config = recipe_file.get("calibration_set", {})
MAX_SEQUENCE_LENGTH = calibration_config["max_seq_length"]
if args.max_seq_length is not None:
    MAX_SEQUENCE_LENGTH = args.max_seq_length
SHUFFLE = calibration_config.get("shuffle", True)
SEED = calibration_config.get("seed", 42)
datasets_config = calibration_config.get("datasets", [])

print(f"Loaded recipe: {args.recipe_yaml}")
print(f"  max_seq_length: {MAX_SEQUENCE_LENGTH}")
print(f"  shuffle: {SHUFFLE}  seed: {SEED}")
print(f"  group_size: {group_size}  use_loss_mask: {use_loss_mask}")
print(f"  dataset entries: {len(datasets_config)}")

# =========================
# Model (CPU + sequential onloading)
# =========================
print("\n" + "=" * 70)
print("Loading Command A+ to CPU (device_map=None)")
print("Sequential pipeline will process Cohere2MoeDecoderLayer one at a time on GPU")
print("=" * 70 + "\n")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(
            f"  GPU {i}: {props.name} — {props.total_memory / 1e9:.1f} GB total"
        )

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map=None,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if not getattr(tokenizer, "chat_template", None):
    _pt = getattr(processor, "tokenizer", None)
    if _pt is not None and getattr(_pt, "chat_template", None):
        tokenizer.chat_template = _pt.chat_template

print(f"Loaded: {MODEL_ID}")
print(f"Model class: {type(model).__name__}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
print(f"Residency: CPU (device_map=None)")

SRC_GEN_CONFIG_PATH = os.path.join(MODEL_ID, "generation_config.json")
try:
    source_generation_config = GenerationConfig.from_pretrained(
        MODEL_ID, trust_remote_code=True
    )
    print(f"Captured GenerationConfig eos_token_id={source_generation_config.eos_token_id}")
except Exception as e:
    source_generation_config = None
    print(f"WARNING: could not load GenerationConfig: {e}")

# =========================
# Dataset formatters (recipe schema)
# =========================
def format_sharegpt(example, columns, tokenizer):
    formatted_messages = []
    if len(columns) >= 2 and "system" in columns[0].lower():
        system_prompt = example.get(columns[0], "")
        if system_prompt:
            formatted_messages.append({"role": "system", "content": str(system_prompt)})
        conv_column = columns[1]
    else:
        conv_column = columns[0]

    messages = example.get(conv_column, [])
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except json.JSONDecodeError:
            formatted_messages.append({"role": "user", "content": messages})
            if formatted_messages:
                text = tokenizer.apply_chat_template(
                    formatted_messages, tokenize=False
                )
                return {"text": text, "messages": json.dumps(formatted_messages)}
            return {"text": "", "messages": ""}

    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", msg.get("from", "user"))
                content = msg.get("content", msg.get("value", ""))
                if role in ("human", "user"):
                    role = "user"
                elif role in ("gpt", "assistant", "bot"):
                    role = "assistant"
                elif role == "system":
                    role = "system"
                if content:
                    formatted_messages.append({"role": role, "content": str(content)})
            elif isinstance(msg, str):
                idx = len([m for m in formatted_messages if m["role"] != "system"])
                role = "user" if idx % 2 == 0 else "assistant"
                formatted_messages.append({"role": role, "content": str(msg)})

    if not formatted_messages:
        return {"text": "", "messages": ""}
    try:
        text = tokenizer.apply_chat_template(formatted_messages, tokenize=False)
        return {"text": text, "messages": json.dumps(formatted_messages)}
    except Exception:
        return {"text": "", "messages": ""}


def format_prompt_answer(example, columns, tokenizer):
    prompt_col = columns[0]
    answer_col = columns[1] if len(columns) > 1 else columns[0]
    messages = [
        {"role": "user", "content": str(example.get(prompt_col, ""))},
        {"role": "assistant", "content": str(example.get(answer_col, ""))},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": text, "messages": json.dumps(messages)}


def format_chat_completion(example, columns, tokenizer):
    for col in columns:
        if col not in example:
            continue
        data = example[col]
        if isinstance(data, list) and data:
            if isinstance(data[0], dict):
                text = tokenizer.apply_chat_template(data, tokenize=False)
                return {"text": text, "messages": json.dumps(data)}
            if isinstance(data[0], str):
                messages = [
                    {"role": "user" if i % 2 == 0 else "assistant", "content": str(item)}
                    for i, item in enumerate(data)
                ]
                text = tokenizer.apply_chat_template(messages, tokenize=False)
                return {"text": text, "messages": json.dumps(messages)}
        elif isinstance(data, str):
            return {"text": data, "messages": ""}
    text = " ".join(str(example.get(col, "")) for col in columns)
    return {"text": text, "messages": ""}


def format_raw_text(example, columns, tokenizer):
    del tokenizer
    texts = [str(example[col]) for col in columns if col in example and example[col]]
    return {"text": " ".join(texts), "messages": ""}


FORMATTERS = {
    "sharegpt": format_sharegpt,
    "prompt_answer": format_prompt_answer,
    "chat_completion": format_chat_completion,
    "raw_text": format_raw_text,
}


def generate_assistant_mask(messages, tokenizer, max_seq_length):
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
            return list(mask)[:max_seq_length]
    except (TypeError, ValueError, KeyError):
        pass

    try:
        if not messages or messages[-1].get("role", "user") != "assistant":
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


def _align_loss_mask_to_seq(mask, seq_len, pad=0):
    m = list(mask)[:seq_len]
    if len(m) < seq_len:
        m.extend([pad] * (seq_len - len(m)))
    return m


# =========================
# Load calibration datasets
# =========================
print("\n=== Loading datasets from recipe ===")
all_datasets = []
total_samples = 0

for ds_config in datasets_config:
    dataset_name = ds_config["dataset"]
    split = ds_config.get("split", "train")
    subset = ds_config.get("subset")
    columns = ds_config.get("columns", [])
    formatter_name = ds_config.get("formatter", "raw_text")
    num_samples = ds_config.get("num_samples", 10)
    streaming = ds_config.get("streaming", False)

    print(
        f"  {dataset_name} split={split} n={num_samples} "
        f"formatter={formatter_name} streaming={streaming}"
    )

    try:
        if streaming:
            if subset:
                ds = load_dataset(dataset_name, subset, split=split, streaming=True)
            else:
                ds = load_dataset(dataset_name, split=split, streaming=True)
            ds = Dataset.from_list(list(ds.take(num_samples)))
        else:
            if subset:
                ds = load_dataset(dataset_name, subset, split=split)
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
        ds = ds.filter(lambda x: len(x.get("text", "")) > 0)
        all_datasets.append(ds)
        total_samples += len(ds)
        print(f"    -> {len(ds)} samples")
    except Exception as e:
        print(f"    -> WARNING: failed: {e}")

if not all_datasets:
    raise ValueError("No datasets loaded from recipe.")

ds = concatenate_datasets(all_datasets)
if SHUFFLE:
    ds = ds.shuffle(seed=SEED)
print(f"\n=== Total calibration samples: {len(ds)} ===")

# =========================
# Tokenize
# =========================
print("\n=== Tokenizing ===")


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
            seq_len = len(result["input_ids"][i])
            if not messages_json:
                loss_masks.append([1] * seq_len)
            else:
                messages = json.loads(messages_json)
                mask = generate_assistant_mask(messages, tokenizer, MAX_SEQUENCE_LENGTH)
                loss_masks.append(_align_loss_mask_to_seq(mask, seq_len, pad=0))
        result["loss_mask"] = loss_masks
    return result


ds = ds.map(
    tokenize_with_mask,
    batched=True,
    remove_columns=ds.column_names,
    num_proc=1 if use_loss_mask else 4,
)

NUM_CALIBRATION_SAMPLES = len(ds)
print(
    f"Tokenized {NUM_CALIBRATION_SAMPLES} samples "
    f"(max_seq_length={MAX_SEQUENCE_LENGTH})"
)

# =========================
# AWQ recipe — W4A16 symmetric, group_size from CLI
# =========================
weight_args = QuantizationArgs(
    num_bits=4,
    type="int",
    symmetric=True,
    strategy="group",
    group_size=group_size,
)

quant_scheme = QuantizationScheme(
    targets=["Linear"],
    weights=weight_args,
    input_activations=None,
    output_activations=None,
)

recipe = [
    AWQModifier(
        ignore=[
            "lm_head",
            "re:.*vision_tower.*",
            "re:.*multi_modal_projector.*",
            "re:.*mlp\\.gate$",
        ],
        mappings=COHERE2_MOE_AWQ_MAPPINGS,
        config_groups={"group_0": quant_scheme},
    ),
]

# =========================
# One-shot AWQ (sequential = one decoder layer at a time on GPU)
# =========================
print(
    f"\n=== Running W4A16 AWQ (group_size={group_size}, "
    f"samples={NUM_CALIBRATION_SAMPLES}, pipeline=sequential) ==="
)
print("  sequential_targets: Cohere2MoeDecoderLayer")
print("  calibrate_moe_context: True (all routed experts see calibration data)")
print("  moe_calibrate_all_experts: True")

oneshot_kwargs = dict(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    tokenizer=tokenizer,
    pipeline="sequential",
    sequential_targets=["Cohere2MoeDecoderLayer"],
    calibrate_moe_context=True,
    moe_calibrate_all_experts=True,
    use_loss_mask=use_loss_mask,
)

oneshot(**oneshot_kwargs)

# =========================
# Optional smoke generation
# =========================
if not args.skip_sample_gen:
    print("\n\n========== SAMPLE GENERATION ==============")
    dispatch_model(model)
    messages = [{"role": "user", "content": "Hello my name is"}]
    if getattr(tokenizer, "chat_template", None):
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt_text = "Hello my name is"
    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    output = model.generate(**inputs, max_new_tokens=64)
    print(tokenizer.decode(output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True))
    print("==========================================\n\n")

# =========================
# Save
# =========================
SAVE_DIR = args.output_path
if source_generation_config is not None:
    model.generation_config = source_generation_config

model.save_pretrained(SAVE_DIR, save_compressed=True, max_shard_size="5GB")
processor.save_pretrained(SAVE_DIR)

if os.path.isfile(SRC_GEN_CONFIG_PATH):
    shutil.copy2(SRC_GEN_CONFIG_PATH, os.path.join(SAVE_DIR, "generation_config.json"))

print("\n=== Complete ===")
print("Saved to:", SAVE_DIR)
