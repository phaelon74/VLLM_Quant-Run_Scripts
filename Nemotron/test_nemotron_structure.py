# Save as: Nemotron/test_nemotron_structure.py
# Run: python Nemotron/test_nemotron_structure.py
from transformers import AutoConfig, AutoModelForCausalLM

MODEL_PATH = "/media/fmodels/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"
config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)

# 1. Find first Mamba layer index and layer structure
layers_block_type = getattr(config, "layers_block_type", [])
first_mamba_idx = next((i for i, t in enumerate(layers_block_type) if t == "mamba"), None)
print(f"First Mamba layer index: {first_mamba_idx}")
print(f"layers_block_type (first 10): {layers_block_type[:10]}")

# 2. Try to load model; fall back to config-only if mamba_ssm is missing
model = None
try:
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype="auto", trust_remote_code=True)
except ImportError as e:
    if "mamba" in str(e).lower():
        print("\n[INFO] mamba_ssm not installed - using config-based structure inference.")
        print("       Install with: pip install mamba-ssm causal-conv1d")
        print("       For full verification, install and re-run.")
    else:
        raise

if model is not None:
    names = [n for n, _ in model.named_modules()]
    norm_names = [n for n in names if "norm" in n.lower()]
    in_proj_names = [n for n in names if "in_proj" in n]
    out_proj_names = [n for n in names if "out_proj" in n]
    mamba_mixer_names = [n for n in names if "mixer" in n and any(x in n for x in ["in_proj", "out_proj", "norm"])]

    print("\n=== Norm layers (sample) ===")
    for n in norm_names[:15]:
        print(f"  {n}")

    print("\n=== in_proj layers (sample) ===")
    for n in in_proj_names[:10]:
        print(f"  {n}")

    print("\n=== out_proj layers (sample) ===")
    for n in out_proj_names[:10]:
        print(f"  {n}")

    print("\n=== Mamba mixer submodules (first 2 Mamba layers) ===")
    next_mamba_idx = next((i for i in range(first_mamba_idx + 1, len(layers_block_type)) if layers_block_type[i] == "mamba"), None) if first_mamba_idx is not None else None
    for n in mamba_mixer_names:
        if first_mamba_idx is not None and f"layers.{first_mamba_idx}." in n:
            print(f"  {n}")
        if next_mamba_idx is not None and f"layers.{next_mamba_idx}." in n:
            print(f"  {n}")
else:
    # Config-only: infer expected structure from transformers NemotronH architecture
    base = getattr(config, "base_model_prefix", "model") or "model"
    print(f"\n=== Expected structure (from config, base_prefix={base}) ===")
    if first_mamba_idx is not None:
        print(f"  Block norm:   {base}.layers.{first_mamba_idx}.norm")
        print(f"  Mamba in_proj: {base}.layers.{first_mamba_idx}.mixer.in_proj")
        print(f"  Mamba norm:   {base}.layers.{first_mamba_idx}.mixer.norm")
        print(f"  Mamba out_proj: {base}.layers.{first_mamba_idx}.mixer.out_proj")
    print("\n  Regex patterns should match these. Verify after installing mamba_ssm if needed.")
