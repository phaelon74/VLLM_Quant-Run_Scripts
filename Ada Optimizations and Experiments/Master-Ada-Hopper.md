# Master-Ada-Hopper.md — FP8 W8A8 on Ada (and what changes on Hopper)

Reference model: **Qwen3.8-27B**. Hardware: **1× RTX 4090 48GB (modded), SM 8.9**.
Serving: vLLM 0.29.1rc1.dev178+g6906c9f24, torch 2.13 + CUDA 13.0, driver 595.

Final build: **FP8 W8A8, per-channel weights + per-token dynamic activations**,
30 GB, built with llm-compressor `FP8_DYNAMIC`, no calibration data.

---

## 1. Why the "normal" FP8 checkpoint is slow on Ada

Ada (SM 8.9) has native FP8 tensor cores. Whether you reach them depends
entirely on **how the weight scales are laid out**:

| Layout | Kernel on SM 8.9 | Compute |
|---|---|---|
| **Block-scaled** (e.g. 128×128, what most vendor FP8 checkpoints ship) | falls back to Marlin, weight-only | **BF16** |
| **Per-channel** or per-tensor | `CutlassFP8ScaledMMLinearKernel` | **native FP8** |

vLLM gates this explicitly, and you can query it:

```bash
python -c "from vllm import _custom_ops as ops; \
  print([(c, ops.cutlass_scaled_mm_supports_block_fp8(c)) for c in (89, 90)])"
# [(89, False), (90, True)]
```

Block-scaled FP8 GEMMs exist only from Hopper (SM 9.0). On Ada, a block-scaled
checkpoint loads fine, serves fine, and **silently runs weight-only** — i.e. it
behaves like W8A16: 8-bit weights, BF16 math, no prefill gain. There is a vLLM
issue open describing exactly this for `Qwen/Qwen3.8-27B-FP8` on Ada hardware.

So on Ada, "FP8" is not one thing. `format: float-quantized` with
`strategy: channel` is a different product from a block-scaled checkpoint, and
only the first one uses the silicon you bought.

**On Hopper (H100/H200, SM 9.0)** the situation reverses: block-scaled FP8 is
natively supported, so the vendor checkpoint is fine and generally preferred
(finer scales, better fidelity). The build procedure in this document is for
Ada; on Hopper, try the official FP8 first and verify the kernel the same way.

### Why FP8 is the right choice on Ada, where INT8 was a fight

Ampere has INT8 tensor cores but no FP8, so W8A8 there means INT8 activations —
a uniform grid that collapses under per-channel outliers. Our probe
(`activation_outlier_probe.py`, same model, BF16 activations):

| Input to | INT8 err | FP8 err | INT8/FP8 |
|---|---|---|---|
| `q/k/v_proj` | 0.1095 | **0.0170** | 8.1× |
| `linear_attn.in_proj_*` | 0.1087 | **0.0182** | 6.7× |
| `mlp.down_proj` | 0.0799 | **0.0250** | 3.3× |
| `mlp.gate/up_proj` | 0.0620 | **0.0240** | 2.7× |

FP8's exponent represents small channels next to 200× outliers. Consequence:
on Ampere we needed a mixed-precision build (`--a16 norm-fed`) to keep quality;
on Ada, **one uniform FP8 scheme covers every module**, including the recurrent
`linear_attn` projections. Simpler build, fewer config traps.

---

## 2. The progression

One build. This is the honest summary: FP8 W8A8 on Ada is a **data-free PTQ**,
about 15 minutes of work, and it landed above the ship gate on the first try.

| Build | Method | Calibration | Main-eval conf. agree | Outcome |
|---|---|---|---|---|
| W8A16 (`lued` INT8) | baseline | — | 99.888% | fidelity leader, BF16-speed prefill |
| **FP8 W8A8** | llm-compressor `FP8_DYNAMIC` | **none** | **99.599%** | ships: +47–80% prefill |
| Official Qwen FP8 (block) | vendor | — | not collected | pointless on Ada: weight-only fallback |

Compare with the Ampere INT8 path, which took **five builds** (v1 → v2 →
v2-qwentools → v2-a16, plus an INT4 anchor), an activation probe, a
mixed-precision config, and three config-format fixes to reach 99.52%.

Why FP8 is easier:

- **No calibration data.** Weight scales come from the weights; activation
  scales are computed per token at runtime. No recipe, no held-out set, no
  contamination risk, no 6-hour tuning run.
- **No outlier workaround.** No probe-driven module exclusions.
- **No mixed formats.** One config group, one kernel, no `mixed-precision`
  header, no target-collision trap.

Final layout:

```
FP8 W8A8  (per-channel weights, per-token dynamic activations, float-quantized)
  496 modules: 64 attention + 240 gated-deltanet + 192 MLP
BF16 (untouched)
  lm_head, vision tower + merger (333 tensors), MTP head (15 tensors),
  GDN conv1d / A_log / dt_bias
```

One nice detail from the build log: llm-compressor reported
`Found 161 offset-norm modules to convert`, i.e. it recognised the zero-centred
RMSNorm (`x * (1 + w)`) this architecture uses. That is the same convention
that makes hand-rolled SmoothQuant dangerous on Ampere.

---

## 3. Performance evaluation

Same harness and rules as Ampere: `bench-ctx-sweep.sh`, single stream,
prefill = `input_len / TTFT`, decode = `1000 / TPOT`, MTP on, prefix caching on,
16,384 batched tokens, server restarted before the sweep.

**RTX 4090 48GB, FP8 W8A8 vs INT8 W8A16**

| Context | Prefill W8A16 | Prefill FP8 | Δ | Decode W8A16 | Decode FP8 | Δ |
|---|---|---|---|---|---|---|
| 128 | 982 | 1,057 | +8% | 77.0 | 69.6 | −10% |
| 1K | 2,344 | **3,675** | +57% | 82.4 | 75.1 | −9% |
| 4K | 2,445 | **4,150** | +70% | 73.1 | 64.7 | −11% |
| 8K | 2,425 | **4,276** | +76% | 67.9 | 66.1 | −3% |
| 16K | 2,368 | **4,262** | +80% | 59.5 | 57.7 | −3% |
| 32K | 2,241 | **3,884** | +73% | 50.3 | 47.8 | −5% |
| 64K | 2,025 | **3,290** | +62% | 35.3 | 34.4 | −3% |
| 131K | 1,691 | **2,488** | +47% | 22.5 | 22.9 | +2% |

The decode cost is per-token activation quantization work at batch 1, and it
**converges to parity by 131K** as attention comes to dominate weight reads.

**Power limiting is cheap here.** At ~320 W instead of 450 W:

| Context | Prefill 450 W | Prefill ~320 W | Δ | Decode Δ |
|---|---|---|---|---|
| 8K | 4,276 | 4,007 | −6.3% | +1.1% |
| 32K | 3,884 | 3,597 | −7.4% | +0.2% |
| 131K | 2,488 | 2,305 | −7.4% | −0.9% |

~30% less power for ~7% less prefill, decode untouched — prefill is
compute-bound and pays; decode is memory-bound and does not care.

**KV cache**: 152,307 tokens at 131K context with BF16 KV on 48 GB
(`gpu-memory-utilization 0.92`), i.e. 1.16× headroom. Ada supports FP8 KV
natively if you want the full 262K.

---

## 4. Accuracy evaluation

Same harness (`kld_eval.py`), same two evals, same BF16 reference collected on
the 8×3060 rig (BF16 is 54 GB and does not fit on 48 GB).

**Cross-rig control.** Because the reference came from another machine at a
different TP size, we collected the *same* W8A16 checkpoint on both rigs:

| Candidate | Rig / TP | conf. agree | flips/10k | KLD mean |
|---|---|---|---|---|
| W8A16 | 8×3060, TP8 | 99.888% | 5.38 | 0.014378 |
| W8A16 | 4090, TP1 | 99.888% | 5.38 | 0.014644 |

Identical to three decimals. **Cross-rig noise is effectively zero**, so FP8's
gap is real and not an artifact. Always run this control when the reference and
the candidate live on different machines.

**General eval (261,888 positions, BF16 PPL 8.5765)**

| Candidate | conf. agree | flips/10k | all-pos | Δ PPL | KLD mean | entropy |
|---|---|---|---|---|---|---|
| W8A16 | 99.888% | 5.38 | 98.32% | +0.43% | 0.014378 | −0.02% |
| **FP8 W8A8 (Ada)** | **99.599%** | 19.21 | 95.60% | −4.13% | 0.050629 | **+0.32%** |
| INT8 W8A8 v2-a16 (Ampere) | 99.521% | 22.95 | 94.93% | +3.64% | 0.065910 | −1.77% |
| INT4 W4A16 (anchor) | 98.608% | 66.67 | 91.43% | −4.71% | 0.138157 | +1.24% |

**Tool-calling eval (68,541 positions)**

| Candidate | conf. agree | flips/10k | KLD mean | entropy |
|---|---|---|---|---|
| W8A16 (4090) | 99.674% | 17.51 | 0.042285 | +0.54% |
| **FP8 W8A8** | **98.408%** | 85.35 | 0.209199 | −0.04% |
| INT8 W8A8 v2-a16 | 97.927% | 111.17 | 0.274336 | −3.32% |
| INT4 W4A16 | 93.927% | 325.64 | 0.632020 | +3.83% |

Notes:

- FP8 clears the 99.5% ship gate on general text and is the **best W8A8 build on
  either architecture** for tool calling.
- It is the **only** W8A8 variant with positive entropy (+0.32%), meaning it did
  not narrow the distribution — the flattening/repetition risk that applies to
  every INT8 build does not apply here.
- W8A16 still leads on fidelity. FP8 is ~3.5× its mean KLD. If maximum
  faithfulness matters more than prefill, W8A16 remains the better pick.

---

## 5. How to replicate on a new model (Ada)

### Phase 0 — Confirm the hardware path

```bash
python -c "from vllm import _custom_ops as ops; print(ops.cutlass_scaled_mm_supports_block_fp8(89))"
# False -> per-channel FP8 required on this card
```

If you are on Hopper, this prints True for 90; try the vendor's block FP8 first
and skip to Phase 3.

### Phase 1 — Check for a usable vendor checkpoint

Inspect `config.json` → `quantization_config`:

- `strategy: channel` or `tensor` → usable on Ada, go straight to validation.
- `block` / `block_structure` / a 2-D `weight_block_size` → **build your own**.

### Phase 2 — Build (data-free, ~15 min)

```bash
python <model>_fp8-w8a8.py /path/BF16 ./OUT-FP8-W8A8
```

The script (`Qwen3.8-27B_fp8-w8a8.py` here) does:

1. `AutoModelForImageTextToText` load — preserves VLM weight paths vLLM expects.
2. `QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=[...])`,
   `oneshot()` with no dataset.
3. Ignore list: `lm_head`, vision tower + merger, MTP, and (belt and braces)
   GDN state parameters. **Quantize the recurrent projections** — FP8 handles them.
4. Post-save: transformers-v5 key remap, copy `mtp.*` from source + update the
   shard index, add MTP names **and** `re:^mtp.*` / `re:.*mtp.*` to `ignore`,
   copy processor/template configs.
5. Verify: weights 8-bit float **per-channel** (fails the build if block-scaled),
   activations 8-bit float dynamic per-token, MTP/vision preserved and unquantized.

`--exclude-gdn` exists if you ever want the Ampere-style A/B, but on Ada it
should not be needed. Run the activation probe first only if the model is
unfamiliar; FP8 error was 0.017–0.026 everywhere on this one.

### Phase 3 — Serve and confirm you got what you paid for

```bash
grep -oE "Selected [A-Za-z0-9]+ for [A-Za-z0-9]+" server.log
# want: Selected CutlassFP8ScaledMMLinearKernel for CompressedTensorsW8A8Fp8
```

`MarlinLinearKernel for CompressedTensorsWNA16` means weight-only: either you
are serving the wrong directory or the checkpoint is block-scaled.

Also check `'model_tag'` in the log. Serving the wrong model is the single most
common self-inflicted error; it once produced a full "FP8" benchmark table that
exactly reproduced the W8A16 baseline.

### Phase 4 — Validate

1. **Context sweep** vs the W8A16/BF16 baseline. Expect large prefill gains and
   a small decode cost that shrinks with context.
2. **First-row `AccLen` ≈ 3.5** if the model has an MTP head.
3. **KLD, both evals**, with the **cross-rig W8A16 control** if the BF16
   reference was collected elsewhere. Ship gate: ≥99.5% confident agreement.
4. **Prefix-cache test** — the kernel path changed, so prior passes do not carry.
5. **Needle recall** at long contexts as a regression guard.

### Phase 5 — Lock it

Archive the build log, verify output, benchmark TSVs, `.npz` files, and the
vLLM commit + wheel sha256 next to the checkpoint. There is no calibration data
to archive, which is one of FP8's quiet advantages.

---

## 6. Ada vs Ampere, at a glance

| | Ampere (SM 8.0/8.6) | Ada (SM 8.9) | Hopper (SM 9.0) |
|---|---|---|---|
| 8-bit tensor cores | INT8 only | INT8 + FP8 | INT8 + FP8 |
| Best W8A8 format | INT8 | **FP8 per-channel** | FP8 (block OK) |
| Vendor block-FP8 usable | no | **no** (weight-only fallback) | yes |
| Activation outliers | need mixed precision or SmoothQuant | handled by FP8 exponent | handled |
| Calibration needed | yes (AutoRound, 512 samples, hours) | **no** (data-free PTQ) | no |
| Builds to get it right | 5 | **1** | 1 (expected) |
