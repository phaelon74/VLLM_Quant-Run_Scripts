"""
Muse Glimmer 30B W8A16 PTQ (Post-Training Quantization)

  - QuantizationModifier with preset scheme (no AWQ, no GPTQ)
  - MuseGlimmerForConditionalGeneration to preserve VLM weight paths for vLLM
  - No calibration dataset needed
  - W8A16 preset: INT8 per-channel symmetric weights, FP16/BF16 activations
  - Vision / perception path left in BF16; DFlash drafter artifacts ignored

  Muse Glimmer-30B is a dense multimodal (vision-language) agentic model with:
    - 52-layer dense transformer decoder (~29.6B total incl. vision)
    - Perception encoder (ViT-G/14, ~1.8B, 50 layers) as vision_tower
    - vision_adapter + vision_projection bridging vision -> language hidden size
    - Hybrid sliding-window / full attention (3:1), 131K context
    - Image + video inputs (video as frames); text output

  Requires (nightly recommended):
    pip install -U transformers llmcompressor compressed-tensors accelerate

  Usage:
    python Cohere_Labs/Muse_Glimmer/Muse_Glimmer-W8A16_PTQ.py <model_path> <output_path>

  vLLM (after quant):
    vllm serve <output_path> -tp <N>
"""
import argparse
import os
import shutil

import torch.nn as nn
from compressed_tensors.offload import dispatch_model
from transformers import AutoProcessor, GenerationConfig

# Prefer the native VLM class (correct language_model / vision_* weight paths).
# Fall back to Auto loaders on older nightlies that only expose Auto mappings.
try:
    from transformers import MuseGlimmerForConditionalGeneration as _ModelCls
except ImportError:
    try:
        from transformers import AutoModelForMultimodalLM as _ModelCls
    except ImportError:
        from transformers import AutoModelForImageTextToText as _ModelCls

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

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# Sidecar artifacts that vLLM / Transformers need for tokenizer + vision preprocess.
# Weights / config.json are produced by save_pretrained(save_compressed=True).
SIDECAR_FILES = (
    "processor_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "chat_template.jinja",
    "chat_template.json",
    "generation_config.json",
)

# Skip DFlash / drafter companion weights if present next to the BF16 checkpoint.
DRAFTER_NAME_MARKERS = ("draft", "drafter", "dflash")


def _is_drafter_name(name: str) -> bool:
    lower = name.lower()
    return any(marker in lower for marker in DRAFTER_NAME_MARKERS)


def copy_sidecar_artifacts(source_dir: str, save_dir: str) -> None:
    """
    Ensure processor / tokenizer / chat-template sidecars land in the quant dir.

    processor.save_pretrained usually writes these, but transformers + llmcompressor
    nightlies can omit jinja templates or coerce generation_config. Byte-copy from
    the source checkpoint so vLLM has a complete multimodal package.
    """
    os.makedirs(save_dir, exist_ok=True)
    copied = []

    for filename in SIDECAR_FILES:
        if _is_drafter_name(filename):
            continue
        src = os.path.join(source_dir, filename)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(save_dir, filename))
            copied.append(filename)

    # Catch any extra *.jinja chat / tool templates not in the fixed list.
    try:
        for filename in os.listdir(source_dir):
            if not filename.endswith(".jinja"):
                continue
            if _is_drafter_name(filename) or filename in copied:
                continue
            src = os.path.join(source_dir, filename)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(save_dir, filename))
                copied.append(filename)
    except OSError as e:
        print(f"WARNING: could not scan source dir for .jinja files: {e}")

    if copied:
        print(f"Copied sidecar artifacts -> {save_dir}: {', '.join(copied)}")
    else:
        print(
            f"WARNING: no sidecar artifacts found under {source_dir}; "
            "verify processor/tokenizer files manually."
        )


# =========================
# Parse Command-Line Arguments
# =========================
parser = argparse.ArgumentParser(
    description=(
        "Run W8A16 PTQ on Muse Glimmer 30B (meta-models/Muse-Glimmer-30B). "
        "No calibration data needed. Vision / perception path preserved."
    )
)
parser.add_argument("model_path", type=str, help="Path to the source model directory.")
parser.add_argument("output_path", type=str, help="Path to save quantized model.")
parser.add_argument(
    "--skip-sample-gen",
    action="store_true",
    help="Skip the post-quantization text generation smoke test.",
)

args = parser.parse_args()

# =========================
# Model
# =========================
MODEL_ID = args.model_path

# MuseGlimmerForConditionalGeneration keeps:
#   model.language_model.layers.*
#   model.vision_tower.*
#   model.vision_adapter.* / model.vision_projection.*
# which vLLM expects. AutoModelForCausalLM would drop the vision path.
model = _ModelCls.from_pretrained(MODEL_ID, dtype="auto", trust_remote_code=True)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
print(f"Loaded model: {MODEL_ID}")
print(f"Model class: {type(model).__name__}")

# ----------------------------------------------------------------------
# Snapshot the source GenerationConfig BEFORE quantization / save.
#
# Muse ships list-valued eos_token_id (e.g. [200001, 200008]). transformers v5
# + llmcompressor oneshot / save_pretrained can coerce that list to a scalar,
# which breaks multi-stop decoding in vLLM. Capture, restore, and byte-copy.
# ----------------------------------------------------------------------
SRC_GEN_CONFIG_PATH = os.path.join(MODEL_ID, "generation_config.json")
try:
    source_generation_config = GenerationConfig.from_pretrained(
        MODEL_ID, trust_remote_code=True
    )
    print(
        f"Captured source GenerationConfig: "
        f"eos_token_id={source_generation_config.eos_token_id}, "
        f"bos_token_id={source_generation_config.bos_token_id}, "
        f"pad_token_id={source_generation_config.pad_token_id}"
    )
except Exception as e:
    source_generation_config = None
    print(f"WARNING: could not load source GenerationConfig from {MODEL_ID}: {e}")

# =========================
# Quantization recipe
# =========================
# W8A16 preset: INT8 per-channel symmetric weights, activations untouched.
#
# Ignore list (dense VLM; aligned with Gemma4 / LLaVA / Command-A+ patterns):
#   - lm_head: output projection; quantizing hurts quality
#   - vision_tower: Perception Encoder (~1.8B ViT-G/14) — keep BF16
#   - vision_adapter / vision_projection: Muse's multimodal projector stack
#   - multi_modal_projector / mm_projector: defensive aliases (Kimi parent / renames)
#   - draft / dflash: speculative drafter companion (not part of this PTQ)
# Dense model: no MoE router / expert ignores needed.
recipe = QuantizationModifier(
    targets="Linear",
    scheme="W8A16",
    ignore=[
        "lm_head",
        "re:.*vision_tower.*",
        "re:.*vision_adapter.*",
        "re:.*vision_projection.*",
        "re:.*multi_modal_projector.*",
        "re:.*mm_projector.*",
        "re:.*draft.*",
        "re:.*dflash.*",
    ],
)

# =========================
# Apply quantization (no calibration data needed for W8A16 PTQ)
# =========================
print("\n=== Running W8A16 PTQ ===")
oneshot(model=model, recipe=recipe)

# =========================
# Quick sanity generation (text-only)
# =========================
if not args.skip_sample_gen:
    print("\n\n========== SAMPLE GENERATION ==============")
    dispatch_model(model)

    SAMPLE_PROMPT = "Hello my name is"
    messages = [{"role": "user", "content": SAMPLE_PROMPT}]

    _tok = getattr(processor, "tokenizer", processor)
    has_chat_tmpl = getattr(_tok, "chat_template", None) is not None or getattr(
        processor, "chat_template", None
    ) is not None

    if has_chat_tmpl:
        prompt_text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt_text = SAMPLE_PROMPT

    inputs = processor(text=[prompt_text], return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items() if hasattr(v, "to")}
    input_len = inputs["input_ids"].shape[-1]

    output = model.generate(**inputs, max_new_tokens=100)
    print(processor.decode(output[0][input_len:], skip_special_tokens=True))
    print("==========================================\n\n")

# =========================
# Save compressed model
# =========================
SAVE_DIR = args.output_path

if source_generation_config is not None:
    model.generation_config = source_generation_config
    print(
        f"Restored model.generation_config.eos_token_id = "
        f"{model.generation_config.eos_token_id} prior to save."
    )

model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)

# Belt-and-suspenders: restore generation_config + copy processor/tokenizer sidecars.
copy_sidecar_artifacts(MODEL_ID, SAVE_DIR)

print("\n=== Complete ===")
print("Saved to:", SAVE_DIR)
print("\nLoad in vLLM (example):")
print(f"  vllm serve {SAVE_DIR} -tp <num_gpus>")
