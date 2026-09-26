"""
Qwen3.8-27B INT8 W8A8 PTQ via Intel AutoRound

Target: Ampere (SM80/86 - RTX 3060, CMP 170HX). W8A8 keeps the GEMMs in INT8
tensor-core space instead of dequantizing INT8 weights into BF16 math the way
W8A16/Marlin does. Expect faster prefill; decode is unchanged or slightly worse.

Quantizer: AutoRound (sign-gradient rounding + clipping search). No GPTQ.

Calibration comes from the YAML recipes (General_reasoning.yaml,
calibrate_software_engineer.yaml): the script reads the recipe, pulls each
dataset, applies the named formatter, renders through the model's chat
template, and hands AutoRound a tokenized DataLoader.

Preserved at BF16 (never quantized):
  - mtp.*          speculative-decoding head, also re-copied post-save because
                   Transformers never loads or writes it
  - visual/merger  vision tower, so the checkpoint stays multimodal
  - lm_head
  - Gated DeltaNet recurrent pieces (conv1d, A_log, dt_bias, gates, norms)

Weight strategy (--weight-strategy):
  channel  per-output-channel scales -> compressed-tensors int-quantized ->
           vLLM CUTLASS INT8 W8A8 kernels. The known-fast path on Ampere.
  group    group_size scales (default 32) -> pack-quantized -> vLLM's wNa8
           path. Better weight fidelity, different kernel. UNVERIFIED for speed
           on SM86: benchmark before adopting.

Variants (--variant):
  v1  Gated DeltaNet projections stay BF16 (conservative, default)
  v2  Gated DeltaNet in/out projections quantized, gates/norms/conv stay BF16

Usage:
  # 0. classify every Linear first; nothing is quantized
  python Qwen3.8-27B_int8-w8a8.py SRC OUT --dump-modules

  # 1. per-channel build with the software-engineering recipe
  python Qwen3.8-27B_int8-w8a8.py SRC OUT-v1-chan \
      --recipe calibrate_software_engineer.yaml --variant v1 --weight-strategy channel

  # 2. group-32 build, same data, for the A/B
  python Qwen3.8-27B_int8-w8a8.py SRC OUT-v1-g32 \
      --recipe calibrate_software_engineer.yaml --variant v1 \
      --weight-strategy group --group-size 32
"""
import argparse
import glob
import json
import os
import random
import re
import sys
import zlib

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

# Transformers v5 compatibility shim (carried over from the W8A16 script)
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


# =====================================================================
# Module targeting
# =====================================================================
BASE_IGNORE = [
    "lm_head",
    "re:.*visual.*",
    "re:.*merger.*",
    "re:.*mtp.*",
    "re:.*conv1d.*",
    "re:.*(A_log|dt_bias).*",
]
V1_IGNORE = ["re:.*linear_attn.*"]
V2_IGNORE = ["re:.*linear_attn.*(gate|norm|beta|dt).*"]


def build_ignore(variant):
    return BASE_IGNORE + (V1_IGNORE if variant == "v1" else V2_IGNORE)


def matches_any(name, patterns):
    for pat in patterns:
        if pat.startswith("re:"):
            if re.search(pat[3:], name):
                return True
        elif pat == name or name.endswith("." + pat):
            return True
    return False


def linear_module_names(model):
    return [
        name for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear) or mod.__class__.__name__.endswith("Linear")
    ]


def classify(name):
    if "visual" in name or "merger" in name:
        return "vision (BF16)"
    if "mtp" in name:
        return "mtp (BF16)"
    if "linear_attn" in name:
        return "gated-deltanet (v1 BF16 / v2 quantized)"
    if name.endswith("lm_head"):
        return "lm_head (BF16)"
    if re.search(r"\.(q|k|v|o)_proj$", name):
        return "full-attention (quantized)"
    if re.search(r"\.(gate|up|down)_proj$", name):
        return "mlp (quantized)"
    return "OTHER - decide explicitly"


def dump_linear_modules(model):
    counts = {}
    for name in linear_module_names(model):
        group = classify(name)
        counts[group] = counts.get(group, 0) + 1
        print(f"{group:<42} {name}")
    print("\n=== counts ===")
    for group, count in sorted(counts.items()):
        print(f"  {count:>5}  {group}")
    if any(g.startswith("OTHER") for g in counts):
        print("\nWARNING: unclassified Linear modules above. Decide where each "
              "belongs before quantizing.")


# Modules whose input comes straight out of an RMSNorm carry huge per-channel
# outliers (probe: max/median 150-226). Per-token INT8 activations lose ~11% of
# the signal there versus ~2% for FP8, so they keep INT8 weights but 16-bit
# activations. vLLM fuses q/k/v and the linear_attn in_proj_* into single
# layers, so every member of a fused group must share one scheme.
A16_PRESETS = {
    "none": [],
    "norm-fed": [r"re:.*self_attn\.(q|k|v)_proj$", r"re:.*linear_attn\.in_proj_.*"],
    "norm-fed+mlp-in": [r"re:.*self_attn\.(q|k|v)_proj$", r"re:.*linear_attn\.in_proj_.*",
                        r"re:.*mlp\.(gate|up)_proj$"],
}


def build_layer_config(model, ignore, args):
    """AutoRound per-layer overrides.

    ignored      -> BF16 weights and activations
    a16 patterns -> INT8 per-channel weights, 16-bit activations (W8A16)
    everything   -> the global INT8 W8A8 scheme
    """
    a16 = A16_PRESETS[args.a16]
    layer_config = {}
    kept, quantized, w8a16 = [], [], []
    for name in linear_module_names(model):
        if matches_any(name, ignore):
            layer_config[name] = {"bits": 16, "act_bits": 16, "data_type": "float"}
            kept.append(name)
        elif a16 and matches_any(name, a16):
            layer_config[name] = {"bits": 8, "group_size": -1, "sym": True,
                                  "data_type": "int", "act_bits": 16}
            w8a16.append(name)
            quantized.append(name)
        else:
            quantized.append(name)
    if a16:
        print(f"\nW8A16 overrides ({args.a16}): {len(w8a16)} modules keep 16-bit activations")
        for kind in sorted({classify(n) for n in w8a16}):
            print(f"  a16   {sum(classify(n) == kind for n in w8a16):>5}  {kind}")
    print(f"\nlayer_config: {len(quantized)} modules quantized, {len(kept)} kept at BF16")
    by_group = {}
    for name in quantized:
        by_group[classify(name)] = by_group.get(classify(name), 0) + 1
    for group, count in sorted(by_group.items()):
        print(f"  quantized  {count:>5}  {group}")
    if not quantized:
        sys.exit("ERROR: ignore list matched everything; nothing would be quantized.")
    for probe in ("mtp", "visual", "merger"):
        leaked = [n for n in quantized if probe in n]
        if leaked:
            sys.exit(f"ERROR: {len(leaked)} '{probe}' modules would be quantized, "
                     f"e.g. {leaked[0]}. Fix the ignore patterns.")
    return layer_config


# =====================================================================
# Calibration recipe (YAML)
# =====================================================================
def _stable_hash(text):
    return zlib.crc32(str(text).encode("utf-8"))


def render_prefix(prefix, row):
    """Render a formatter_params prefix.

    Supports the recipe's template form:
      "... {{ ['Zephyr', 'Prolog', ...][hash(row|string) % 60] }} ..."
    Jinja2 is used when available; otherwise the list is parsed and indexed
    with the same stable hash so builds stay reproducible.
    """
    if not prefix or "{{" not in prefix:
        return prefix or ""
    try:
        from jinja2 import Environment

        env = Environment()
        env.filters["string"] = lambda v: str(v)
        env.globals["hash"] = _stable_hash
        return env.from_string(prefix).render(row=row)
    except Exception:  # noqa: BLE001 - jinja2 missing or template unsupported
        match = re.search(r"\[([^\]]*)\]\s*\[\s*hash\([^)]*\)\s*%\s*(\d+)\s*\]", prefix)
        if not match:
            return re.sub(r"\{\{.*?\}\}", "", prefix)
        options = re.findall(r"'([^']*)'|\"([^\"]*)\"", match.group(1))
        options = [a or b for a, b in options]
        modulus = int(match.group(2))
        choice = options[_stable_hash(row) % modulus] if options else ""
        return re.sub(r"\{\{.*?\}\}", choice, prefix)


def _messages_to_text(messages, tokenizer):
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, enable_thinking=True
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False)
    except Exception:  # noqa: BLE001 - template rejects the role sequence
        return "\n\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages
        )


SHAREGPT_ROLES = {
    "human": "user", "user": "user", "prompter": "user",
    "gpt": "assistant", "assistant": "assistant", "chatgpt": "assistant",
    "system": "system", "tool": "tool", "function": "tool", "observation": "tool",
}


def format_row(row, spec, tokenizer):
    """Turn one dataset row into calibration text, per the recipe's formatter."""
    formatter = spec.get("formatter", "raw_text")
    columns = spec.get("columns") or []
    prefix = render_prefix((spec.get("formatter_params") or {}).get("prefix"), row)

    def col(i):
        if i < len(columns) and columns[i] in row:
            return row[columns[i]]
        return None

    if formatter == "prompt_answer":
        prompt, answer = col(0), col(1)
        if prompt is None:
            return None
        messages = [{"role": "user", "content": prefix + str(prompt)}]
        if answer:
            messages.append({"role": "assistant", "content": str(answer)})
        return _messages_to_text(messages, tokenizer)

    if formatter in ("chat_completion", "chat_completion_with_sysprompt"):
        messages = None
        system = None
        for name in columns:
            value = row.get(name)
            if isinstance(value, list) and value and isinstance(value[0], dict):
                messages = value
            elif name in ("system", "system_prompt") and value:
                system = str(value)
        if messages is None:
            return None
        if isinstance(messages[0], dict) and "from" in messages[0]:
            return format_row({"__c": messages}, {"formatter": "sharegpt", "columns": ["__c"]}, tokenizer)
        messages = [
            {"role": m.get("role", "user"), "content": str(m.get("content", ""))}
            for m in messages if isinstance(m, dict)
        ]
        if system:
            messages = [{"role": "system", "content": system}] + messages
        return _messages_to_text(messages, tokenizer) if messages else None

    if formatter == "sharegpt":
        turns = col(0)
        if not isinstance(turns, list):
            return None
        messages = []
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            role = SHAREGPT_ROLES.get(str(turn.get("from", "user")).lower(), "user")
            messages.append({"role": role, "content": str(turn.get("value", ""))})
        return _messages_to_text(messages, tokenizer) if messages else None

    if formatter == "deepmind_code_contests":
        name = col(0)
        if not name:
            return None
        messages = [{"role": "user", "content":
                     f"{prefix}Solve this competitive programming problem: {name}"}]
        return _messages_to_text(messages, tokenizer)

    # raw_text: concatenate the listed columns, prefix first
    parts = [str(row[c]) for c in columns if row.get(c)]
    if not parts:
        return None
    text = prefix + "\n\n".join(parts)
    if prefix:  # a prefix turns raw text into an instruction
        return _messages_to_text([{"role": "user", "content": text}], tokenizer)
    return text


def load_recipe_texts(recipe_path, tokenizer, args):
    import yaml
    from datasets import load_dataset

    with open(recipe_path, encoding="utf-8") as fh:
        recipe = yaml.safe_load(fh)["calibration_set"]

    seed = recipe.get("seed", args.seed)
    max_len = args.seqlen or recipe.get("max_seq_length", 2048)
    specs = recipe["datasets"]
    print(f"\n=== calibration recipe: {os.path.basename(recipe_path)} ===")
    print(f"{len(specs)} sources, max_seq_length={max_len}, seed={seed}")

    texts, failures = [], []
    for spec in specs:
        name = spec["dataset"]
        want = int(spec.get("num_samples", 8) * args.sample_scale)
        if want <= 0:
            continue
        try:
            kwargs = {"split": spec.get("split", "train")}
            if spec.get("subset"):
                kwargs["name"] = spec["subset"]
            if spec.get("data_files"):
                kwargs["data_files"] = spec["data_files"]
            if spec.get("streaming"):
                kwargs["streaming"] = True
            ds = load_dataset(name, **kwargs)
            rows = []
            if spec.get("streaming"):
                for row in ds:
                    rows.append(row)
                    if len(rows) >= want * 4:
                        break
            else:
                pool = min(len(ds), want * 4)
                rows = list(ds.shuffle(seed=seed).select(range(pool)))
            got = 0
            for row in rows:
                text = format_row(row, spec, tokenizer)
                if text and len(text.strip()) > 32:
                    texts.append(text)
                    got += 1
                if got >= want:
                    break
            print(f"  ok   {got:>4}/{want:<4} {name}")
            if got < want:
                failures.append(f"{name}: only {got}/{want}")
        except Exception as exc:  # noqa: BLE001 - one bad source must not kill the run
            print(f"  FAIL   0/{want:<4} {name}: {type(exc).__name__}: {str(exc)[:120]}")
            failures.append(f"{name}: {type(exc).__name__}")

    if recipe.get("shuffle", True):
        random.Random(seed).shuffle(texts)

    print(f"\ncollected {len(texts)} calibration samples "
          f"({len(failures)} source(s) short or failed)")
    if failures and args.strict_recipe:
        sys.exit("ERROR: --strict-recipe set and some sources failed:\n  "
                 + "\n  ".join(failures))
    if len(texts) < 64:
        sys.exit(f"ERROR: only {len(texts)} samples collected; too few to calibrate.")
    return texts, max_len


def load_cached_texts(path, args):
    """Read the JSONL cache written by check_calibration_recipe.py."""
    texts = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                texts.append(json.loads(line)["text"])
    max_len = args.seqlen or 8192
    print(f"\n=== calibration cache: {os.path.basename(path)} ===")
    print(f"{len(texts)} samples, max_seq_length={max_len}")
    if len(texts) < 64:
        sys.exit(f"ERROR: only {len(texts)} samples in {path}; too few to calibrate.")
    return texts, max_len


def build_dataloader(texts, tokenizer, max_len, batch_size):
    """AutoRound accepts a DataLoader yielding dicts of input_ids."""
    from torch.utils.data import DataLoader

    encoded = []
    for text in texts:
        ids = tokenizer(
            text, truncation=True, max_length=max_len, add_special_tokens=True,
            return_tensors="pt",
        ).input_ids[0]
        if ids.numel() >= 16:
            encoded.append(ids)
    lengths = [int(x.numel()) for x in encoded]
    print(f"tokenized {len(encoded)} samples: min/mean/max = "
          f"{min(lengths)}/{sum(lengths)//len(lengths)}/{max(lengths)} tokens")

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    def collate(batch):
        width = max(int(x.numel()) for x in batch)
        out = torch.full((len(batch), width), pad_id, dtype=torch.long)
        for i, ids in enumerate(batch):
            out[i, : ids.numel()] = ids
        return {"input_ids": out}

    return DataLoader(encoded, batch_size=batch_size, shuffle=False, collate_fn=collate)


# =====================================================================
# Post-save fixes (carried over from the W8A16 script)
# =====================================================================
KEY_REMAPS = [
    ("model.language_model.language_model.", "model.language_model."),
    ("model.language_model.visual.", "model.visual."),
]


def fix_saved_weight_keys(save_dir):
    files = sorted(glob.glob(os.path.join(save_dir, "*.safetensors")))
    total_remapped = 0
    for fpath in files:
        with safe_open(fpath, framework="pt") as fh:
            metadata = fh.metadata() or {}
            tensors, remapped = {}, 0
            for key in fh.keys():
                new_key, changed = key, True
                while changed:
                    changed = False
                    for old, new in KEY_REMAPS:
                        if new_key.startswith(old):
                            new_key = new + new_key[len(old):]
                            changed = True
                            break
                if new_key != key:
                    remapped += 1
                tensors[new_key] = fh.get_tensor(key).clone()
        if remapped:
            tmp = fpath + ".tmp"
            save_file(tensors, tmp, metadata=metadata)
            os.replace(tmp, fpath)
        print(f"  {os.path.basename(fpath)}: remapped {remapped}/{len(tensors)}")
        total_remapped += remapped
        del tensors
    if total_remapped == 0:
        print("  (no double-nested keys; nothing to fix)")


def _is_mtp_key(key):
    return key.startswith("mtp.") or ".mtp." in key


def copy_mtp_weights_from_source(source_dir, save_dir):
    existing = {k for k in output_keys(save_dir) if _is_mtp_key(k)}
    if existing:
        print(f"  {len(existing)} MTP tensors already present "
              f"(AutoRound copies them into model_extra_tensors.safetensors); "
              f"nothing to do")
        return

    index_path = os.path.join(source_dir, "model.safetensors.index.json")
    mtp = {}
    if os.path.isfile(index_path):
        with open(index_path, encoding="utf-8") as fh:
            weight_map = json.load(fh)["weight_map"]
        shards = {}
        for key, shard in weight_map.items():
            if _is_mtp_key(key):
                shards.setdefault(shard, []).append(key)
        for shard, keys in sorted(shards.items()):
            print(f"  reading {len(keys)} MTP keys from {shard}")
            with safe_open(os.path.join(source_dir, shard), framework="pt") as fh:
                for key in keys:
                    mtp[key] = fh.get_tensor(key).clone()
    else:
        for fpath in sorted(glob.glob(os.path.join(source_dir, "*.safetensors"))):
            with safe_open(fpath, framework="pt") as fh:
                for key in fh.keys():
                    if _is_mtp_key(key):
                        mtp[key] = fh.get_tensor(key).clone()

    if not mtp:
        print("WARNING: no mtp.* tensors in source - MTP will not work in vLLM.")
        return

    target = sorted(glob.glob(os.path.join(save_dir, "*.safetensors")))[0]
    with safe_open(target, framework="pt") as fh:
        metadata = fh.metadata() or {}
        tensors = {key: fh.get_tensor(key).clone() for key in fh.keys()}
    already = [k for k in tensors if _is_mtp_key(k)]
    tensors.update(mtp)
    tmp = target + ".tmp"
    save_file(tensors, tmp, metadata=metadata)
    os.replace(tmp, target)
    print(f"  copied {len(mtp)} MTP tensors into {os.path.basename(target)} "
          f"({len(already)} were already present)")

    index_out = os.path.join(save_dir, "model.safetensors.index.json")
    if os.path.isfile(index_out):
        with open(index_out, encoding="utf-8") as fh:
            idx = json.load(fh)
        for key in mtp:
            idx["weight_map"][key] = os.path.basename(target)
        with open(index_out, "w", encoding="utf-8") as fh:
            json.dump(idx, fh, indent=2)
        print(f"  index updated with {len(mtp)} MTP entries")


def ensure_mtp_in_ignore(save_dir):
    """Add mtp.* modules to the config's ignore list.

    Transformers never loads the MTP head, so it is missing from
    model.named_modules() when we build layer_config, and the exported config
    therefore omits it. vLLM would then build the MTP head as INT8 W8A8, find
    BF16 weights with no scales, and draft garbage (acceptance 0.00%).
    """
    cfg_path = os.path.join(save_dir, "config.json")
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    qcfg = cfg.get("quantization_config")
    if not qcfg:
        print("  no quantization_config; nothing to do")
        return
    ignore = list(qcfg.get("ignore", []))

    wanted = set()
    for fpath in sorted(glob.glob(os.path.join(save_dir, "*.safetensors"))):
        with safe_open(fpath, framework="pt") as fh:
            for key in fh.keys():
                if _is_mtp_key(key) and key.endswith(".weight"):
                    if len(fh.get_slice(key).get_shape()) == 2:   # Linear only
                        wanted.add(key[: -len(".weight")])
    missing = sorted(m for m in wanted if m not in ignore)
    if not missing:
        print(f"  MTP already covered ({len(wanted)} modules)")
        return
    qcfg["ignore"] = ignore + missing
    with open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)
    print(f"  added {len(missing)} MTP modules to the ignore list "
          f"(was {len(ignore)}, now {len(qcfg['ignore'])})")


def fix_mixed_precision_config(save_dir, a16_preset):
    """Make a W8A8 + W8A16 export loadable by vLLM.

    AutoRound stores the two groups differently (W8A8 as raw INT8 `weight`,
    W8A16 as INT32 `weight_packed` + `weight_shape`) and writes the right
    per-group formats, but it leaves the top-level format as int-quantized and
    targets both groups at the class name "Linear". vLLM keys schemes by target
    string, so the groups collide. Verified with resolve_ct_schemes.py: after
    this fix every fused layer (qkv_proj, in_proj_qkvz, in_proj_ba) resolves to
    CompressedTensorsWNA16 and the rest to CompressedTensorsW8A8Int8.
    """
    if a16_preset == "none":
        return
    cfg_path = os.path.join(save_dir, "config.json")
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    q = cfg["quantization_config"]
    groups = q.get("config_groups", {})
    a8 = [k for k, g in groups.items() if (g.get("input_activations") or {}).get("num_bits") == 8]
    a16 = [k for k, g in groups.items() if not (g.get("input_activations") or {}).get("num_bits")]
    if len(a8) != 1 or len(a16) != 1:
        print(f"  WARNING: expected one W8A8 and one W8A16 group, found {a8} / {a16}; not changed")
        return

    # vLLM layer names are fused: q/k/v -> qkv_proj, in_proj_qkv/z -> in_proj_qkvz,
    # in_proj_b/a -> in_proj_ba. Name/regex matches win over the "Linear" class match.
    a16_targets = [r"re:.*self_attn\.(qkv|q|k|v)_proj$", r"re:.*linear_attn\.in_proj_.*"]
    if a16_preset == "norm-fed+mlp-in":
        a16_targets.append(r"re:.*mlp\.(gate_up|gate|up)_proj$")

    q["format"] = "mixed-precision"
    groups[a8[0]]["format"] = "int-quantized"
    groups[a8[0]]["targets"] = ["Linear"]
    groups[a16[0]]["format"] = "pack-quantized"
    groups[a16[0]]["targets"] = a16_targets
    with open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)
    print(f"  format=mixed-precision; {a8[0]} int-quantized on Linear; "
          f"{a16[0]} pack-quantized on {a16_targets}")


def resolve_output_dir(requested):
    """AutoRound may save into a <requested>/<model-name>-w8a8/ subdirectory."""
    if os.path.isfile(os.path.join(requested, "config.json")):
        return requested
    candidates = [
        os.path.join(requested, entry) for entry in sorted(os.listdir(requested))
        if os.path.isfile(os.path.join(requested, entry, "config.json"))
    ] if os.path.isdir(requested) else []
    if len(candidates) == 1:
        print(f"  note: AutoRound saved to {candidates[0]}")
        return candidates[0]
    if len(candidates) > 1:
        sys.exit(f"ERROR: several model directories under {requested}: {candidates}")
    sys.exit(f"ERROR: no config.json under {requested}; the save step did not complete.")


def output_keys(save_dir):
    """Every tensor name in the output, from the index and from the shards."""
    keys = set()
    index_path = os.path.join(save_dir, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path, encoding="utf-8") as fh:
            keys.update(json.load(fh)["weight_map"])
    for fpath in sorted(glob.glob(os.path.join(save_dir, "*.safetensors"))):
        with safe_open(fpath, framework="pt") as fh:
            keys.update(fh.keys())
    return keys


def copy_aux_files(source_dir, save_dir):
    for name in ("generation_config.json", "preprocessor_config.json",
                 "video_preprocessor_config.json", "processor_config.json",
                 "chat_template.json", "chat_template.jinja"):
        src, dst = os.path.join(source_dir, name), os.path.join(save_dir, name)
        if os.path.isfile(src) and not os.path.isfile(dst):
            with open(src, "rb") as a, open(dst, "wb") as b:
                b.write(a.read())
            print(f"  copied {name}")


# =====================================================================
# Verification
# =====================================================================
def verify_output(save_dir):
    ok = True

    def check(label, condition, detail=""):
        nonlocal ok
        print(f"{'OK  ' if condition else 'FAIL'}  {label} {detail}")
        ok = ok and condition

    with open(os.path.join(save_dir, "config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    qcfg = cfg.get("quantization_config", {})
    fmt = qcfg.get("format") or qcfg.get("quant_method")
    groups = qcfg.get("config_groups", {})
    print(f"\nformat: {fmt}")

    w_ok = a_ok = False
    for name, group in groups.items():
        w = group.get("weights") or {}
        a = group.get("input_activations") or {}
        print(f"  {name}: weights {w.get('num_bits')}bit {w.get('type')} "
              f"strategy={w.get('strategy')} group_size={w.get('group_size')} "
              f"sym={w.get('symmetric')} | activations {a.get('num_bits')}bit "
              f"{a.get('type')} strategy={a.get('strategy')} dynamic={a.get('dynamic')}")
        if w.get("num_bits") == 8 and str(w.get("type")).endswith("int"):
            w_ok = True
        if a.get("num_bits") == 8 and a.get("dynamic"):
            a_ok = True
            strategy = (w.get("strategy") or "").lower()
            if strategy == "channel":
                print("     -> vLLM path: CUTLASS INT8 W8A8 (the fast Ampere path)")
            elif strategy in ("group", "tensor_group"):
                print("     -> vLLM path: wNa8 pack-quantized + INT8 activations. "
                      "Benchmark this against the channel build before adopting.")
            else:
                print(f"     -> unexpected weight strategy '{strategy}'")

    check("INT8 weights", w_ok)
    check("INT8 dynamic activations (this is what makes it W8A8, not W8A16)", a_ok)
    a16_groups = [g for g in groups.values()
                  if (g.get("weights") or {}).get("num_bits") == 8
                  and not (g.get("input_activations") or {}).get("num_bits")]
    if len(groups) > 1:
        targets = [tuple(g.get("targets") or []) for g in groups.values()]
        check("config groups have distinct targets (else vLLM collapses them into one)",
              len(set(targets)) == len(targets))
        check("mixed build declares format=mixed-precision", qcfg.get("format") == "mixed-precision")
    print(f"..    W8A16 (16-bit activation) groups: {len(a16_groups)}"
          f"{'  targets: ' + str(a16_groups[0].get('targets'))[:120] if a16_groups else ''}")

    keys = sorted(output_keys(save_dir))

    scaled = {k.rsplit(".", 1)[0] for k in keys
              if k.endswith(("weight_scale", "weight_scale_inv", "scales"))}
    mtp_keys = [k for k in keys if _is_mtp_key(k)]
    vis_keys = [k for k in keys if "visual" in k or "merger" in k]

    ignored = qcfg.get("ignore", [])
    check("MTP head in the quantization ignore list",
          any("mtp" in p.lower() for p in ignored),
          f"({sum('mtp' in p.lower() for p in ignored)} of {len(ignored)} entries) "
          f"-- without this vLLM drafts with misread weights and acceptance is 0%")
    check("MTP tensors present", len(mtp_keys) > 0, f"({len(mtp_keys)} keys)")
    check("MTP not quantized", not any(_is_mtp_key(k) for k in scaled))
    check("vision tower present", len(vis_keys) > 0, f"({len(vis_keys)} keys)")
    check("vision tower not quantized",
          not any("visual" in k or "merger" in k for k in scaled))
    check("attention/MLP quantized",
          any(re.search(r"\.(q|k|v|o|gate|up|down)_proj$", k) for k in scaled),
          f"({len(scaled)} quantized modules)")
    check("lm_head not quantized", not any(k.startswith("lm_head") for k in scaled))
    print(f"..    linear_attn quantized modules: "
          f"{len([k for k in scaled if 'linear_attn' in k])} (0 for v1, >0 for v2)")
    for name in ("tokenizer_config.json", "preprocessor_config.json"):
        check(f"{name} present", os.path.isfile(os.path.join(save_dir, name)))

    print("\n=== VERIFY: " + ("PASS" if ok else "FAIL") + " ===")
    return ok


# =====================================================================
# Main
# =====================================================================
def main():
    ap = argparse.ArgumentParser(
        description="INT8 W8A8 PTQ for Qwen3.8-27B using Intel AutoRound.")
    ap.add_argument("model_path")
    ap.add_argument("output_path")
    ap.add_argument("--recipe", help="Calibration YAML (General_reasoning.yaml, "
                                     "calibrate_software_engineer.yaml).")
    ap.add_argument("--calib-jsonl", help="Pre-built calibration cache from "
                                          "check_calibration_recipe.py (skips the recipe).")
    ap.add_argument("--variant", choices=["v1", "v2"], default="v1")
    ap.add_argument("--a16", choices=sorted(A16_PRESETS), default="none",
                    help="modules that keep 16-bit activations (INT8 weights). "
                         "'norm-fed' covers q/k/v and linear_attn in_proj_*, the "
                         "outlier-heavy inputs found by activation_outlier_probe.py")
    ap.add_argument("--weight-strategy", choices=["channel", "group"], default="channel")
    ap.add_argument("--group-size", type=int, default=32)
    ap.add_argument("--dump-modules", action="store_true")
    ap.add_argument("--iters", type=int, default=400,
                    help="AutoRound tuning steps per block (200 fast, 1000 best).")
    ap.add_argument("--nsamples", type=int, default=512)
    ap.add_argument("--seqlen", type=int, default=0,
                    help="Override the recipe's max_seq_length.")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--gradient-accumulate-steps", type=int, default=8)
    ap.add_argument("--sample-scale", type=float, default=1.0,
                    help="Multiply every recipe num_samples (2.0 doubles the set).")
    ap.add_argument("--strict-recipe", action="store_true",
                    help="Abort if any recipe source fails to load.")
    ap.add_argument("--low-gpu-mem-usage", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--format", default="auto_round:llm_compressor",
                    help="Export format; llm_compressor keeps it compressed-tensors.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-key-remap", action="store_true")
    ap.add_argument("--mllm-mode", choices=["text", "processor"], default="text",
                    help="text: force AutoRound's text calibration path (default). "
                         "processor: use the multimodal path with the model processor.")
    ap.add_argument("--template", default=None,
                    help="AutoRound template name, only for --mllm-mode processor.")
    ap.add_argument("--extra-ar-kwargs", default=None,
                    help='JSON of extra AutoRound kwargs, e.g. \'{"nblocks":1}\'.')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print(f"Loading {args.model_path}")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, dtype="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    try:
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    except Exception as exc:  # noqa: BLE001
        processor = None
        print(f"WARNING: no processor ({exc}); vision configs will be copied instead.")

    if args.dump_modules:
        dump_linear_modules(model)
        return

    if not args.recipe and not args.calib_jsonl:
        sys.exit("ERROR: pass --recipe or --calib-jsonl (AutoRound calibrates on real data).")

    ignore = build_ignore(args.variant)
    print(f"\n=== variant {args.variant} / weights {args.weight_strategy} / a16 {args.a16} ===")
    for pat in ignore:
        print(f"  ignore {pat}")
    layer_config = build_layer_config(model, ignore, args)

    if args.calib_jsonl:
        texts, max_len = load_cached_texts(args.calib_jsonl, args)
    else:
        texts, max_len = load_recipe_texts(args.recipe, tokenizer, args)
    dataloader = build_dataloader(texts, tokenizer, max_len, args.batch_size)

    group_size = -1 if args.weight_strategy == "channel" else args.group_size
    print(f"\n=== AutoRound: INT8_W8A8, group_size={group_size}, iters={args.iters}, "
          f"nsamples={min(args.nsamples, len(texts))}, seqlen={max_len} ===")

    import inspect

    from auto_round import AutoRound

    ar_kwargs = dict(
        tokenizer=tokenizer,
        scheme="INT8_W8A8",          # int8 weights + int8 dynamic activations
        group_size=group_size,       # -1 = per output channel, 32 = grouped
        sym=True,
        layer_config=layer_config,   # BF16 islands: mtp, vision, lm_head, GDN state
        dataset=dataloader,
        nsamples=min(args.nsamples, len(texts)),
        seqlen=max_len,
        iters=args.iters,
        batch_size=args.batch_size,
        gradient_accumulate_steps=args.gradient_accumulate_steps,
        low_gpu_mem_usage=args.low_gpu_mem_usage,
        device=args.device,
        seed=args.seed,
    )

    # This is a vision-language model, so AutoRound routes calibration through
    # its multimodal path. That path needs BOTH the tokenizer and the processor.
    # We calibrate the language layers on text; the vision tower stays BF16.
    params = inspect.signature(AutoRound.__init__).parameters
    has_var_kw = any(v.kind is inspect.Parameter.VAR_KEYWORD for v in params.values())
    accepted = set(params)
    print(f"AutoRound signature: {'**kwargs (pass-through)' if has_var_kw else sorted(accepted - {'self', 'model'})}")

    if args.mllm_mode == "text" and "mllm" in accepted:
        ar_kwargs["mllm"] = False
        print("AutoRound: forcing the text calibration path (mllm=False)")
    else:
        if processor is None:
            sys.exit("ERROR: this AutoRound needs a processor for vision-language "
                     "models, but AutoProcessor failed to load. Fix the processor "
                     "load (check preprocessor_config.json in the source model).")
        ar_kwargs["processor"] = processor
        if args.template:
            ar_kwargs["template"] = args.template
        print("AutoRound: multimodal calibration path, passing tokenizer + processor"
              + (f", template={args.template}" if args.template else ""))

    if args.extra_ar_kwargs:
        ar_kwargs.update(json.loads(args.extra_ar_kwargs))
        print(f"AutoRound: extra kwargs {json.loads(args.extra_ar_kwargs)}")

    # Only prune when the signature is explicit. With **kwargs we cannot tell
    # what is supported, and pruning would strip required arguments.
    if not has_var_kw:
        dropped = [k for k in ar_kwargs if k not in accepted]
        if dropped:
            print(f"AutoRound: dropping kwargs this version does not accept: {dropped}")
            for key in dropped:
                ar_kwargs.pop(key)

    for required in ("tokenizer", "dataset", "layer_config"):
        if required not in ar_kwargs:
            sys.exit(f"ERROR: '{required}' was dropped from the AutoRound call. "
                     f"Refusing to run: without it the build would be wrong.")

    try:
        autoround = AutoRound(model, **ar_kwargs)
    except TypeError as exc:
        sys.exit(f"ERROR: AutoRound rejected an argument ({exc}).\n"
                 f"Inspect the signature and re-run with --extra-ar-kwargs, e.g.\n"
                 f"  python -c \"import auto_round, inspect; "
                 f"print(inspect.signature(auto_round.AutoRound.__init__))\"")

    print("\n=== Quantizing (this is the long part) ===")
    try:
        autoround.quantize_and_save(output_dir=args.output_path, format=args.format)
    except AssertionError as exc:
        sys.exit(f"ERROR: AutoRound calibration failed: {exc}\n"
                 f"If this mentions processor/tokenizer/template, the multimodal "
                 f"path is missing something: try --template default, or upgrade "
                 f"auto-round to a version with an 'mllm' switch.")

    save_dir = resolve_output_dir(args.output_path)

    if not args.skip_key_remap:
        print("\n=== Fixing transformers v5 weight keys ===")
        fix_saved_weight_keys(save_dir)

    print("\n=== Checking MTP weights ===")
    copy_mtp_weights_from_source(args.model_path, save_dir)

    print("\n=== Ensuring the MTP head is excluded from quantization ===")
    ensure_mtp_in_ignore(save_dir)

    if args.a16 != "none":
        print("\n=== Writing mixed-precision config (W8A8 + W8A16 groups) ===")
        fix_mixed_precision_config(save_dir, args.a16)

    if processor is not None:
        try:
            processor.save_pretrained(save_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: processor.save_pretrained failed ({exc})")
    print("\n=== Copying auxiliary configs ===")
    copy_aux_files(args.model_path, save_dir)

    print("\n=== Verifying output ===")
    ok = verify_output(save_dir)

    print("\n=== Complete ===")
    print("Saved to:", save_dir)
    print("Next: serve on the pinned vLLM build, confirm the startup log selects an "
          "INT8 activation scheme (not weight-only Marlin), then run the context "
          "sweep, prefix-cache test and long-context recall checks.")
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
