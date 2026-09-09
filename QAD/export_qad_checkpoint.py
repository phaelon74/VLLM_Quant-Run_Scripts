"""
Export a QAD training checkpoint to vLLM-ready format.

After QAD training, trainer.save_model() saves an uncompressed checkpoint
(~300GB for 8B) with modelopt quantization state. This script loads that
checkpoint and exports it to the compressed format that vLLM can serve.

Usage:
    python Main/export_qad_checkpoint.py \
        /path/to/NVFP4-QAD \
        /path/to/NVFP4-QAD-exported
"""

import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import modelopt.torch.opt as mto
    from modelopt.torch.export import export_hf_checkpoint
    mto.enable_huggingface_checkpointing()
except ImportError:
    print("ERROR: nvidia-modelopt is required. Install with: pip install nvidia-modelopt>=0.35.0")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Export QAD checkpoint to vLLM-ready format.")
    parser.add_argument("source", type=str, help="Path to QAD training checkpoint (trainer.save_model output).")
    parser.add_argument("destination", type=str, help="Output path for exported vLLM-ready model.")
    args = parser.parse_args()

    print(f"Source:      {args.source}")
    print(f"Destination: {args.destination}")

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.source,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.source)

    print("Exporting for vLLM...")
    with torch.inference_mode():
        export_hf_checkpoint(model, export_dir=args.destination)

    tokenizer.save_pretrained(args.destination)

    print(f"Exported to {args.destination}")
    print("You can now serve with: vllm serve {args.destination} --quantization modelopt")


if __name__ == "__main__":
    main()
