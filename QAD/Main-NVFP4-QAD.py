"""
NVFP4 QAD (Quantization-Aware Distillation) Script

Performs NVFP4 quantization with QAD using NVIDIA Model-Optimizer's QADTrainer.
Loads datasets from recipe YAML files and supports multi-GPU via accelerate/FSDP2.

QAD is distillation (teacher BF16 -> student NVFP4), not full training.
For best quality (near-BF16 accuracy), use QAD rather than PTQ.

Usage:
    accelerate launch --config_file Main/accelerate_config/fsdp2.yaml Main/Main-NVFP4-QAD.py \\
        --model_path /path/to/model \\
        --output_path /path/to/output \\
        --recipe_yaml Recipes/Datasets/StoryWriting_Default.yaml

Data volume note: QAD paper recommends ~0.5B-2.5B tokens for 24B models.
StoryWriting_Default yields ~512 samples. At 4096 seq len that is ~2M tokens.
For best results on large models, consider a larger recipe or repeating the dataset.
"""

import gc
import json
import os
import types
from dataclasses import dataclass, field
from functools import partial
from warnings import warn

import torch
import yaml
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    default_data_collator,
)
from transformers.trainer_utils import get_last_checkpoint

# Model-Optimizer imports
try:
    import modelopt.torch.opt as mto
    import modelopt.torch.quantization as mtq
    from modelopt.torch.distill.plugins.huggingface import LMLogitsLoss
    from modelopt.torch.quantization.plugins.transformers_trainer import QADTrainer
    from modelopt.torch.utils import print_rank_0
    mto.enable_huggingface_checkpointing()
except ImportError as e:
    raise ImportError(
        "nvidia-modelopt is required. Install with: pip install nvidia-modelopt>=0.35.0"
    ) from e

IGNORE_INDEX = -100


# =============================================================================
# Argument Dataclasses
# =============================================================================


@dataclass
class ModelArguments:
    model_path: str = field(
        metadata={"help": "Source model (HuggingFace ID or local path)."}
    )
    output_path: str = field(
        metadata={"help": "Destination directory for quantized model."}
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Allow custom model code."},
    )
    fsdp_transformer_layer_cls_to_wrap: str | None = field(
        default=None,
        metadata={"help": "Override decoder layer class for FSDP (auto-detect if omitted)."},
    )


@dataclass
class DataArguments:
    recipe_yaml: str = field(
        metadata={"help": "Path to dataset recipe YAML (e.g. Recipes/Datasets/StoryWriting_Default.yaml)."}
    )
    train_size: int = field(
        default=0,
        metadata={"help": "Max train samples (0 = use all)."},
    )
    eval_size: int = field(
        default=200,
        metadata={"help": "Eval samples for calibration + validation."},
    )


@dataclass
class QuantizationArguments:
    quant_cfg: str = field(
        default="NVFP4_DEFAULT_CFG",
        metadata={"help": "Quantization config name (resolved via getattr(mtq, name))."},
    )
    calib_size: int = field(
        default=512,
        metadata={"help": "Calibration samples for quantization scales."},
    )
    compress: bool = field(
        default=False,
        metadata={"help": "Whether to compress model weights after quantization."},
    )


@dataclass
class QADTrainingArguments:
    num_epochs: int = field(default=3, metadata={"help": "Training epochs (QAD paper default: 3)."})
    lr: float = field(default=1e-6, metadata={"help": "Learning rate (QAD paper: 1e-6 for SFT)."})
    per_device_train_batch_size: int = field(default=4, metadata={"help": "Batch size."})
    gradient_accumulation_steps: int = field(default=1, metadata={"help": "Gradient accumulation."})


# =============================================================================
# Dataset Formatters (from llama-NVFP4.py)
# =============================================================================


def format_sharegpt(example, columns, tokenizer):
    """Format ShareGPT-style conversations."""
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
        except Exception:
            formatted_messages.append({"role": "user", "content": messages})
            if formatted_messages:
                text = tokenizer.apply_chat_template(formatted_messages, tokenize=False)
                return {"text": text}
            return {"text": ""}
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", msg.get("from", "user"))
                content = msg.get("content", msg.get("value", ""))
                if role in ["human", "user"]:
                    role = "user"
                elif role in ["gpt", "assistant", "bot"]:
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
        return {"text": ""}
    try:
        text = tokenizer.apply_chat_template(formatted_messages, tokenize=False)
        return {"text": text}
    except Exception:
        return {"text": ""}


def format_prompt_answer(example, columns, tokenizer):
    """Format prompt/answer pairs."""
    prompt_col = columns[0]
    answer_col = columns[1] if len(columns) > 1 else columns[0]
    prompt = example.get(prompt_col, "")
    answer = example.get(answer_col, "")
    messages = [
        {"role": "user", "content": str(prompt)},
        {"role": "assistant", "content": str(answer)},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": text}


def format_chat_completion(example, columns, tokenizer):
    """Format chat completion style data."""
    for col in columns:
        if col in example:
            data = example[col]
            if isinstance(data, list) and len(data) > 0:
                if isinstance(data[0], dict):
                    text = tokenizer.apply_chat_template(data, tokenize=False)
                    return {"text": text}
                elif isinstance(data[0], str):
                    messages = []
                    for i, item in enumerate(data):
                        role = "user" if i % 2 == 0 else "assistant"
                        messages.append({"role": role, "content": str(item)})
                    text = tokenizer.apply_chat_template(messages, tokenize=False)
                    return {"text": text}
            elif isinstance(data, str):
                return {"text": str(data)}
    text = " ".join(str(example.get(col, "")) for col in columns)
    return {"text": text}


def format_raw_text(example, columns, tokenizer):
    """Format raw text data."""
    texts = [str(example[col]) for col in columns if col in example and example[col]]
    return {"text": " ".join(texts)}


FORMATTERS = {
    "sharegpt": format_sharegpt,
    "prompt_answer": format_prompt_answer,
    "chat_completion": format_chat_completion,
    "raw_text": format_raw_text,
}


# =============================================================================
# Recipe Data Module (fixed-length padded sequences for QAD)
# =============================================================================


def _tokenize_and_pad(example, tokenizer, max_seq_length, pad_token_id):
    """Tokenize and pad to fixed length. labels = input_ids for non-pad, -100 for pad."""
    text = example.get("text", "")
    enc = tokenizer(
        text,
        max_length=max_seq_length,
        truncation=True,
        padding="max_length",
        add_special_tokens=False,
        return_tensors=None,
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    labels = [tid if attn == 1 else IGNORE_INDEX for tid, attn in zip(input_ids, attention_mask)]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def make_recipe_data_module(
    calibration_set: dict,
    tokenizer,
    max_seq_length: int,
    train_size: int = 0,
    eval_size: int = 200,
):
    """
    Load datasets from a parsed calibration_set dict and produce train/eval
    datasets with input_ids, attention_mask, labels (fixed-length, left-padded).
    """
    datasets_config = calibration_set.get("datasets", [])
    if not datasets_config:
        raise ValueError("calibration_set must have 'datasets'.")
    shuffle = calibration_set.get("shuffle", True)
    seed = calibration_set.get("seed", 42)

    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id

    all_datasets = []
    for ds_config in datasets_config:
        dataset_name = ds_config["dataset"]
        split = ds_config.get("split", "train")
        columns = ds_config.get("columns", [])
        formatter_name = ds_config.get("formatter", "raw_text")
        num_samples = ds_config.get("num_samples", 10)
        streaming = ds_config.get("streaming", False)
        try:
            if streaming:
                ds = load_dataset(dataset_name, split=split, streaming=True)
                ds = ds.take(num_samples)
                ds = Dataset.from_list(list(ds))
            else:
                ds = load_dataset(dataset_name, split=split)
                n = min(num_samples, len(ds))
                ds = ds.shuffle(seed=seed).select(range(n))
            formatter_fn = FORMATTERS.get(formatter_name, format_raw_text)
            ds = ds.map(
                lambda x: formatter_fn(x, columns, tokenizer),
                remove_columns=ds.column_names,
                num_proc=1,
            )
            ds = ds.filter(lambda x: len(x.get("text", "")) > 0)
            all_datasets.append(ds)
        except Exception as e:
            print_rank_0(f"WARNING: Failed to load {dataset_name}: {e}")
            continue

    if not all_datasets:
        raise ValueError("No datasets were successfully loaded from the recipe.")

    ds = concatenate_datasets(all_datasets)
    if shuffle:
        ds = ds.shuffle(seed=seed)

    def tokenize_fn(ex):
        return _tokenize_and_pad(ex, tokenizer, max_seq_length, pad_token_id)

    ds = ds.map(tokenize_fn, remove_columns=ds.column_names, num_proc=1)

    total = len(ds)
    eval_size = min(eval_size, total - 1) if total > 1 else 0
    train_avail = total - eval_size
    train_size = train_size if train_size > 0 else train_avail
    train_size = min(train_size, train_avail)

    if eval_size > 0:
        ds = ds.train_test_split(test_size=eval_size, shuffle=True, seed=seed)
        train_ds = ds["train"].select(range(min(train_size, len(ds["train"]))))
        eval_ds = ds["test"]
    else:
        train_ds = ds.select(range(min(train_size, len(ds))))
        eval_ds = ds.select(range(min(32, len(ds))))

    return {
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "data_collator": default_data_collator,
    }


# =============================================================================
# Decoder Layer Detection & Memory Leak Fix
# =============================================================================


def detect_sequential_targets(model):
    """Auto-detect decoder layer type for FSDP transformer_layer_cls_to_wrap."""
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
    for name, module in model.named_modules():
        t = type(module).__name__
        if t.endswith("DecoderLayer"):
            return [t]
    return None


def monkey_patch_training_step_to_fix_memory_leak(trainer):
    """Workaround for QAT memory leak (from Model-Optimizer utils.py)."""

    def new_func(original_f_name, trainer, *args, **kwargs):
        gc.collect()
        return getattr(trainer, original_f_name)(*args, **kwargs)

    for f_name in ["training_step", "prediction_step", "_load_best_model"]:
        setattr(trainer, "_original_" + f_name, getattr(trainer, f_name))
        setattr(
            trainer,
            f_name,
            types.MethodType(partial(new_func, "_original_" + f_name), trainer),
        )


# =============================================================================
# Main
# =============================================================================


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, QuantizationArguments, QADTrainingArguments)
    )
    model_args, data_args, quant_args, qad_args = parser.parse_args_into_dataclasses()

    if not os.path.isfile(data_args.recipe_yaml):
        raise FileNotFoundError(f"Recipe YAML not found: {data_args.recipe_yaml}")

    with open(data_args.recipe_yaml, "r") as f:
        recipe_config = yaml.safe_load(f)
    calib = recipe_config.get("calibration_set", {})
    if not calib:
        raise ValueError(f"Recipe {data_args.recipe_yaml} must have 'calibration_set' section.")
    max_seq_length = calib.get("max_seq_length", 4096)

    print_rank_0(f"Model: {model_args.model_path}")
    print_rank_0(f"Output: {model_args.output_path}")
    print_rank_0(f"Recipe: {data_args.recipe_yaml}")
    print_rank_0(f"max_seq_length: {max_seq_length}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_path,
        trust_remote_code=model_args.trust_remote_code,
        model_max_length=max_seq_length,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=model_args.trust_remote_code,
    )
    model.config.use_cache = False

    decoder_cls = model_args.fsdp_transformer_layer_cls_to_wrap
    if decoder_cls is None:
        detected = detect_sequential_targets(model)
        decoder_cls = detected[0] if detected else None
    if decoder_cls:
        print_rank_0(f"FSDP transformer_layer_cls_to_wrap: {decoder_cls}")
    else:
        print_rank_0("WARNING: Could not detect decoder layer class for FSDP wrapping.")

    teacher_model = AutoModelForCausalLM.from_pretrained(
        model_args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=model_args.trust_remote_code,
    )
    distill_config = {
        "teacher_model": teacher_model,
        "criterion": LMLogitsLoss(),
    }

    data_module = make_recipe_data_module(
        calib,
        tokenizer,
        max_seq_length,
        train_size=data_args.train_size,
        eval_size=data_args.eval_size,
    )

    eval_dataset_size = len(data_module["eval_dataset"])
    if quant_args.calib_size > eval_dataset_size:
        warn(
            f"calib_size={quant_args.calib_size} > eval_dataset_size={eval_dataset_size}. "
            f"Setting calib_size to {eval_dataset_size}."
        )
        quant_args.calib_size = eval_dataset_size

    quant_cfg_resolved = getattr(mtq, quant_args.quant_cfg, None)
    if quant_cfg_resolved is None:
        raise ValueError(f"Unknown quant_cfg: {quant_args.quant_cfg}")
    quant_args.quant_cfg = quant_cfg_resolved

    fsdp_config = {"fsdp_cpu_ram_efficient_loading": False}
    if decoder_cls:
        fsdp_config["fsdp_transformer_layer_cls_to_wrap"] = decoder_cls

    training_args = TrainingArguments(
        output_dir=model_args.output_path,
        num_train_epochs=qad_args.num_epochs,
        learning_rate=qad_args.lr,
        per_device_train_batch_size=qad_args.per_device_train_batch_size,
        gradient_accumulation_steps=qad_args.gradient_accumulation_steps,
        bf16=True,
        dataloader_drop_last=True,
        do_train=True,
        do_eval=True,
        save_strategy="epoch",
        logging_steps=10,
        fsdp_config=fsdp_config if torch.cuda.device_count() > 1 else None,
    )

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        print_rank_0(f"Last checkpoint: {last_checkpoint}")

    if training_args.gradient_checkpointing and training_args.gradient_checkpointing_kwargs is None:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    trainer = QADTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        quant_args=quant_args,
        distill_config=distill_config,
        **data_module,
    )
    monkey_patch_training_step_to_fix_memory_leak(trainer)

    print_rank_0("Starting QAD training...")
    trainer.train(resume_from_checkpoint=last_checkpoint)
    print_rank_0("Training completed.")

    print_rank_0("Saving model...")
    trainer.save_model(model_args.output_path)
    tokenizer.save_pretrained(model_args.output_path)

    try:
        from modelopt.torch.export import export_hf_checkpoint
        print_rank_0("Exporting for vLLM...")
        with torch.inference_mode():
            export_hf_checkpoint(trainer.model, export_dir=model_args.output_path)
        print_rank_0(f"Model saved and exported to {model_args.output_path}")
    except Exception as e:
        print_rank_0(f"Export step skipped (model already saved): {e}")

    print_rank_0("Done.")


if __name__ == "__main__":
    main()
