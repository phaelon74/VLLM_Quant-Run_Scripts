import os

import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoModelForCausalLM

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.utils import dispatch_for_generation

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

MODEL_ID = "mistralai/Devstral-2-123B-Instruct-2512"
SAVE_DIR = "Devstral-2-123B-Instruct-2512-W4A16-awq"

CALIB_DATASET = "wikitext"          # you can swap this for a code-heavy set
CALIB_SPLIT = "wikitext-2-raw-v1"
CALIB_SIZE = 128                    # bump this up if you have time/VRAM

def get_tokenizer(model_id: str):
    # Devstral uses mistral-common tokenizer via tekken.json
    tekken_path = hf_hub_download(model_id, "tekken.json")
    tokenizer = MistralTokenizer.from_file(tekken_path)
    return tokenizer

def get_calib_dataset(tokenizer, calib_size: int):
    ds = load_dataset(CALIB_DATASET, CALIB_SPLIT, split="train")
    # simple text field name; adjust if you change the dataset
    texts = [row["text"] for row in ds.select(range(calib_size))]
    # convert to token ids for llmcompressor
    def gen():
        for t in texts:
            ids = tokenizer.encode(t, add_bos=True, add_eos=True)
            yield {"input_ids": ids}
    return gen

def main():
    torch.set_grad_enabled(False)

    print("Loading config…")
    cfg = AutoConfig.from_pretrained(MODEL_ID)
    print("Config model_type:", cfg.model_type, "architectures:", cfg.architectures)

    print("Loading model…")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",          # let accelerate/transformers shard across GPUs
    )

    tokenizer = get_tokenizer(MODEL_ID)

    # Simple W4A16 AWQ recipe – you can refine targets later
    recipe = AWQModifier(
        scheme="W4A16",
        targets="Linear",           # quantize all Linear modules
        ignore=["lm_head"],         # keep final head in full precision
    )

    calib_gen = get_calib_dataset(tokenizer, CALIB_SIZE)

    print("Running AWQ W4A16 oneshot compression…")
    compressed_model = oneshot(
        model=model,
        recipe=recipe,
        dataset=calib_gen,
        max_steps=CALIB_SIZE,
        output_dir=SAVE_DIR,
    )

    # optional: quick generation smoke test
    dispatch_for_generation(compressed_model)
    prompt = "Write a small Python function that reverses a string."
    input_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    input_ids = torch.tensor([input_ids], device=next(compressed_model.parameters()).device)

    with torch.no_grad():
        out = compressed_model.generate(input_ids=input_ids, max_new_tokens=64)

    text = tokenizer.decode(out[0].tolist())
    print("Sample output:\n", text)

if __name__ == "__main__":
    main()
