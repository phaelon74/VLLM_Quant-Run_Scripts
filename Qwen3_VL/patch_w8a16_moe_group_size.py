#!/usr/bin/env python3
"""
Patch compressed-tensors W8A16 MoE checkpoints for vLLM Marlin MoE.

Channel-wise W8A16 PTQ often writes:
  "strategy": "channel", "group_size": null

vLLM Marlin MoE then crashes:
  TypeError: '>' not supported between instances of 'NoneType' and 'int'
  in marlin_moe_padded_intermediate()

Marlin's channel-wise convention is group_size=-1. This rewrites null/None
group_size fields to -1 in config.json (and quantization_config.json if present).

Example:
  python patch_w8a16_moe_group_size.py \\
    /media/fmodels/TheHouseOfTheDude/Qwen3-VL-30B-A3B-Instruct-Heretic_INT8-PTQ
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Any


def _patch_obj(obj: Any) -> int:
    """Return number of group_size null -> -1 replacements."""
    changed = 0
    if isinstance(obj, dict):
        if "group_size" in obj and obj["group_size"] is None:
            obj["group_size"] = -1
            changed += 1
        for v in obj.values():
            changed += _patch_obj(v)
    elif isinstance(obj, list):
        for item in obj:
            changed += _patch_obj(item)
    return changed


def patch_file(path: str) -> int:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    n = _patch_obj(data)
    if n == 0:
        return 0
    bak = path + ".bak"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return n


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Set null W8A16 group_size to -1 for vLLM Marlin MoE."
    )
    parser.add_argument("model_dir", type=str, help="Quantized model directory.")
    args = parser.parse_args()

    model_dir = args.model_dir
    if not os.path.isdir(model_dir):
        raise SystemExit(f"Not a directory: {model_dir}")

    candidates = [
        os.path.join(model_dir, "config.json"),
        os.path.join(model_dir, "quantization_config.json"),
    ]
    total = 0
    for path in candidates:
        if not os.path.isfile(path):
            continue
        n = patch_file(path)
        print(f"{path}: patched {n} group_size field(s)")
        total += n

    if total == 0:
        print(
            "No null group_size fields found. If load still fails, inspect "
            "config.json quantization_config manually."
        )
    else:
        print(f"Done. Total replacements: {total}. Re-run the vLLM launch script.")


if __name__ == "__main__":
    main()
