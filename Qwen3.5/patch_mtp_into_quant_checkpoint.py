"""
Copy mtp.* weights from the base Qwen3.6 checkpoint into a quantized output folder.

Transformers does not load top-level mtp.* tensors into AutoModelForImageTextToText,
so save_pretrained() omits them even when the quant recipe ignores re:.*mtp.*.

Usage:
  python patch_mtp_into_quant_checkpoint.py \\
      /path/to/Qwen/Qwen3.6-27B \\
      /path/to/Qwen3.6-27B-INT8
"""
import argparse
import glob
import json
import os
import sys

from safetensors import safe_open
from safetensors.torch import save_file


def _is_mtp_key(key: str) -> bool:
    return key.startswith("mtp.")


def _collect_mtp_tensors(source_dir: str) -> dict:
    """Load all mtp.* tensors from a HF model dir (sharded or single file)."""
    index_path = os.path.join(source_dir, "model.safetensors.index.json")
    mtp_tensors = {}

    if os.path.isfile(index_path):
        with open(index_path, encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]
        shard_to_keys = {}
        for key, shard in weight_map.items():
            if _is_mtp_key(key):
                shard_to_keys.setdefault(shard, []).append(key)
        if not shard_to_keys:
            return mtp_tensors
        for shard, keys in sorted(shard_to_keys.items()):
            shard_path = os.path.join(source_dir, shard)
            print(f"  Reading {len(keys)} MTP keys from {shard}")
            with safe_open(shard_path, framework="pt") as f:
                for key in keys:
                    mtp_tensors[key] = f.get_tensor(key).clone()
        return mtp_tensors

    safetensor_files = sorted(glob.glob(os.path.join(source_dir, "*.safetensors")))
    if not safetensor_files:
        raise FileNotFoundError(f"No *.safetensors found under {source_dir}")

    for fpath in safetensor_files:
        with safe_open(fpath, framework="pt") as f:
            for key in f.keys():
                if _is_mtp_key(key):
                    mtp_tensors[key] = f.get_tensor(key).clone()
    return mtp_tensors


def _merge_mtp_into_save_dir(save_dir: str, mtp_tensors: dict) -> None:
    """Merge mtp.* tensors into every safetensors file in save_dir."""
    if not mtp_tensors:
        raise ValueError("No mtp.* tensors found in source checkpoint.")

    safetensor_files = sorted(glob.glob(os.path.join(save_dir, "*.safetensors")))
    if not safetensor_files:
        raise FileNotFoundError(f"No *.safetensors found under {save_dir}")

    for fpath in safetensor_files:
        print(f"\nMerging into {os.path.basename(fpath)} ...")
        with safe_open(fpath, framework="pt") as f:
            metadata = f.metadata() or {}
            tensors = {key: f.get_tensor(key).clone() for key in f.keys()}

        overlap = sorted(set(tensors) & set(mtp_tensors))
        if overlap:
            print(f"  Replacing {len(overlap)} existing MTP keys")
        tensors.update(mtp_tensors)

        tmp_path = fpath + ".tmp"
        save_file(tensors, tmp_path, metadata=metadata)
        os.replace(tmp_path, fpath)
        print(f"  Wrote {len(tensors)} tensors total")

    with safe_open(safetensor_files[0], framework="pt") as f:
        mtp_keys = sorted(k for k in f.keys() if _is_mtp_key(k))
        print(f"\nVerification: {len(mtp_keys)} MTP keys in output")
        for key in mtp_keys:
            t = f.get_tensor(key)
            print(f"  {key}: {t.dtype} {tuple(t.shape)}")
        if "mtp.fc.weight" not in mtp_keys:
            print("WARNING: mtp.fc.weight still missing after patch", file=sys.stderr)
            sys.exit(1)
        fc_dtype = str(f.get_tensor("mtp.fc.weight").dtype).replace("torch.", "")
        if fc_dtype not in ("bfloat16", "float16"):
            print(
                f"WARNING: mtp.fc.weight dtype is {fc_dtype}, expected bfloat16/float16",
                file=sys.stderr,
            )


def main():
    parser = argparse.ArgumentParser(
        description="Copy mtp.* BF16 weights from base Qwen3.6 into a quantized checkpoint."
    )
    parser.add_argument(
        "source_dir",
        help="Base model directory (e.g. Qwen/Qwen3.6-27B snapshot).",
    )
    parser.add_argument(
        "save_dir",
        help="Quantized output directory containing model.safetensors.",
    )
    args = parser.parse_args()

    source_dir = os.path.abspath(args.source_dir)
    save_dir = os.path.abspath(args.save_dir)

    print(f"Source: {source_dir}")
    print(f"Output: {save_dir}")
    print("\nCollecting mtp.* from source ...")
    mtp_tensors = _collect_mtp_tensors(source_dir)
    print(f"Found {len(mtp_tensors)} MTP tensors")
    if not mtp_tensors:
        print("ERROR: source has no mtp.* keys", file=sys.stderr)
        sys.exit(1)

    _merge_mtp_into_save_dir(save_dir, mtp_tensors)
    print("\nDone. Re-upload to Hugging Face if needed, then restart vLLM with MTP enabled.")


if __name__ == "__main__":
    main()
