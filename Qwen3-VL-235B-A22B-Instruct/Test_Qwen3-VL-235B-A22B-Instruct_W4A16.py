import base64
from io import BytesIO
import os

import torch
from datasets import load_dataset
from transformers import AutoProcessor
from dotenv import load_dotenv

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.utils import dispatch_for_generation

# Load environment variables for local model source/destination.
script_dir = os.path.dirname(os.path.abspath(__file__))
env_candidates = [
    os.path.join(script_dir, ".env"),
    os.path.join(script_dir, "Example .env"),
]
for env_path in env_candidates:
    if os.path.exists(env_path):
        load_dotenv(dotenv_path=env_path, override=True)
        break

SRC_DIR = os.getenv("SRC_DIR")
DST_DIR = os.getenv("DST_DIR")
DEVICE_MAP = os.getenv("DEVICE_MAP")  # e.g., "auto"
ATTN_IMPLEMENTATION = os.getenv("ATTN_IMPLEMENTATION")  # e.g., "flash_attention_2"
PIPELINE = os.getenv("PIPELINE")  # e.g., "independent" or "sequential"
QUANT_AWARE_CALIBRATION = os.getenv("QUANT_AWARE_CALIBRATION")  # e.g., "0"/"false" to disable

# Load model (prefer local path if provided). Default to Qwen3-VL Instruct.
source_model_id_or_path = SRC_DIR if SRC_DIR else "Qwen/Qwen3-VL-235B-A22B-Instruct"

QwenModelForConditionalGeneration = None
try:
    from transformers import Qwen3VLMoeForConditionalGeneration as _Qwen3
    QwenModelForConditionalGeneration = _Qwen3
except Exception:
    try:
        # Some transformer versions expose a generic vision2seq auto model
        from transformers import AutoModelForVision2Seq as _AutoVision2Seq
        QwenModelForConditionalGeneration = _AutoVision2Seq
    except Exception:
        from transformers import AutoModelForCausalLM as _AutoCausal
        QwenModelForConditionalGeneration = _AutoCausal

model_load_kwargs = {
    "dtype": "auto",
    "trust_remote_code": True,
}
if DEVICE_MAP and len(DEVICE_MAP.strip()) > 0:
    model_load_kwargs["device_map"] = DEVICE_MAP
if ATTN_IMPLEMENTATION and len(ATTN_IMPLEMENTATION.strip()) > 0:
    model_load_kwargs["attn_implementation"] = ATTN_IMPLEMENTATION

model = QwenModelForConditionalGeneration.from_pretrained(
    source_model_id_or_path,
    **model_load_kwargs,
)
processor = AutoProcessor.from_pretrained(source_model_id_or_path, trust_remote_code=True)

# Oneshot arguments
DATASET_ID = "lmms-lab/flickr30k"
DATASET_SPLIT = "test[:512]"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
ds = ds.shuffle(seed=42)


# Apply chat template and produce model-ready tensors (Qwen3-VL style).
def preprocess_and_tokenize(example):
    buffered = BytesIO()
    example["image"].save(buffered, format="PNG")
    encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    base64_qwen = f"data:image/png;base64,{encoded_image}"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": base64_qwen},
                {"type": "text", "text": "What does the image show?"},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    return inputs


ds = ds.map(preprocess_and_tokenize, remove_columns=ds.column_names)


# Define a oneshot data collator for multimodal inputs.
def data_collator(batch):
    assert len(batch) == 1
    sample = batch[0]
    collated = {}
    for key, value in sample.items():
        if torch.is_tensor(value):
            collated[key] = value
        else:
            collated[key] = torch.tensor(value)
    return collated


# Determine sequential target layer names dynamically for Qwen3.
def _detect_sequential_targets_for_qwen3(loaded_model):
    candidate_names = set()
    for module in loaded_model.modules():
        cls_name = module.__class__.__name__
        if cls_name in {
            "Qwen3VLMoeDecoderLayer",
            "Qwen3VLDecoderLayer",
            "QwenMoeDecoderLayer",
        }:
            candidate_names.add(cls_name)
    return sorted(candidate_names)


# Recipe
recipe = [
    GPTQModifier(
        targets="Linear",
        scheme="W4A16",
        ignore=[
            "lm_head",
            "re:visual.*",
            "re:model.visual.*",
            "re:.*router.*",
            "re:.*gate.*",
        ],
    ),
]

# Perform oneshot
maybe_seq_targets = _detect_sequential_targets_for_qwen3(model)
oneshot_kwargs = dict(
    model=model,
    tokenizer=source_model_id_or_path,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    trust_remote_code_model=True,
    data_collator=data_collator,
)
if PIPELINE and len(PIPELINE.strip()) > 0:
    oneshot_kwargs["pipeline"] = PIPELINE

# Only provide sequential_targets when not forcing independent pipeline
if maybe_seq_targets and (not PIPELINE or PIPELINE.lower() != "independent"):
    oneshot_kwargs["sequential_targets"] = maybe_seq_targets

if QUANT_AWARE_CALIBRATION is not None:
    qac_str = QUANT_AWARE_CALIBRATION.strip().lower()
    qac = qac_str not in {"0", "false", "no", "off"}
    oneshot_kwargs["quantization_aware_calibration"] = qac

oneshot(**oneshot_kwargs)

# Save to disk compressed.
suffix = "-W4A16-G128"
if DST_DIR and len(DST_DIR.strip()) > 0:
    SAVE_DIR = DST_DIR
else:
    base_name = os.path.basename(source_model_id_or_path.rstrip("/"))
    SAVE_DIR = base_name + suffix

os.makedirs(SAVE_DIR, exist_ok=True)
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)

