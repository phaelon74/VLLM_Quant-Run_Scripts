# Master-Ampere.md — INT8 W8A8 on Ampere, end to end

Reference model: **Qwen3.8-27B** (dense 27B, hybrid 3× Gated DeltaNet + 1× gated
attention per block, 64 layers, hidden 5120, 24 Q / 4 KV heads, MTP head, vision tower).
Hardware: **8× RTX 3060 12GB (TP8, PCIe 4.0 x4)** and **1× CMP 170HX 64GB HBM2e**.
Serving: vLLM 0.29.1rc1.dev177+gbfd713bf8, CUDA 13.0, driver 595.

Final build: `TheHouseOfTheDude/Qwen3.8-27B_INT8-W8A8-v2-a16`

---

## 1. Why plain W8A16 leaves Ampere performance on the table

**W8A16** stores INT8 weights but computes in BF16: the Marlin kernel dequantizes
each weight tile back to 16-bit and runs a 16-bit GEMM. Activations are never
quantized. So you get the memory saving of 8-bit weights and *none* of the
compute saving. Decode doesn't care (it is memory-bandwidth bound, and the bytes
read are the same), but **prefill is compute-bound, and it runs at BF16 speed**.

**W8A8** quantizes activations too, so the GEMM itself runs on Ampere's INT8
tensor cores (`CutlassInt8ScaledMMLinearKernel`). That is the entire point: the
matrix math moves from BF16 into native INT8.

Measured, same model, same rigs:

| Rig | Prefill W8A16 | Prefill W8A8 (final) | Gain |
|---|---|---|---|
| 8× RTX 3060 @8K | 899 tok/s | 1,036 tok/s | +15% |
| CMP 170HX @8K | 1,755 tok/s | 3,644 tok/s | **+108%** |

The 3060 rig gains far less because eight cards on PCIe 4.0 x4 are
*interconnect*-bound during prefill, not compute-bound: ~128 all-reduces per
chunk (2 per layer × 64 layers). Speeding up the math cannot fix a bus. The
single 170HX has no such tax and shows what the kernel change is really worth.

**Decode is unchanged by design.** Both formats read the same 8-bit weights per
step. Any decode difference comes from file size and per-token quantization
overhead, not from the arithmetic.

### The trap: INT8 activations and outliers

Ampere has INT8 tensor cores but **no FP8**. INT8 is a uniform grid: per-token
scaling sets the step from that token's largest channel. If one channel is 200×
the median, every smaller channel rounds toward zero.

`activation_outlier_probe.py` measures this directly — it hooks every Linear
input on the BF16 model and simulates both formats on the same activations:

| Input to | INT8 rel. err | FP8 rel. err | INT8/FP8 | max/median channel |
|---|---|---|---|---|
| `q/k/v_proj` (after `input_layernorm`) | 0.1095 | 0.0170 | **8.1×** | 226 |
| `linear_attn.in_proj_*` (after `input_layernorm`) | 0.1087 | 0.0182 | **6.7×** | 152 |
| `mlp.down_proj` | 0.0799 | 0.0250 | 3.3× | 140 |
| `mlp.gate/up_proj` | 0.0620 | 0.0240 | 2.7× | 51 |
| `linear_attn.out_proj` | 0.0535 | 0.0244 | 2.2× | 208 |
| `self_attn.o_proj` | 0.0378 | 0.0256 | 1.5× | 84 |

**Norm-fed projections are the problem.** Error does *not* grow with depth
(layers 0–7: 0.091; layers 56–63: 0.061), so "quantize less at the end" is the
wrong instinct — it is about which input, not which layer.

Two ways to fix it: SmoothQuant (migrate the per-channel scale into the weights)
or mixed precision (leave those modules' activations at 16-bit). **We chose
mixed precision**, because Qwen hybrids use zero-centred RMSNorm (`x * (1 + w)`)
and SmoothQuant folding assumes `x * w`; getting that wrong corrupts every layer
silently.

---

## 2. The progression (what we built, in order)

| Build | What changed | Main-eval conf. agree | Tool-eval conf. agree | Verdict |
|---|---|---|---|---|
| W8A16 (baseline, `lued`) | INT8 weights, BF16 activations | 99.888% | 99.576% | fidelity leader, slow prefill |
| **v1** | W8A8; `linear_attn` left BF16 | 99.102% | 95.889% | +15%/+104% prefill, −12% decode |
| **v2** | v1 + `linear_attn` projections quantized | 98.827% | 94.542% | worse quality, smaller file |
| **v2-qwentools** | v2 + calibration in native Qwen tool format | 98.793% | 95.081% | tools +0.5pt, rest flat |
| **v2-a16** ✅ | v2 + 16-bit activations on norm-fed modules | **99.521%** | **97.927%** | ships |
| (reference) W4A16 INT4 | third-party AutoRound INT4 | 98.608% | 93.927% | scale anchor |

Lessons, in order of how much they mattered:

1. **Mixed precision on norm-fed inputs was the single biggest quality win**
   (+0.69 pt main, +2.85 pt tools over v2-qwentools) at no measurable prefill cost.
2. **Quantizing the recurrent (`linear_attn`) projections costs quality** —
   v2 is worse than v1 — but with `--a16 norm-fed` their *input* projections run
   W8A16 and only `out_proj` stays W8A8, which recovers it and shrinks the file
   from 35 GB to 30 GB (helping decode).
3. **Calibration format matters, but less than architecture.** Replacing
   Hermes-format tool data with self-generated native-Qwen tool traces moved the
   tool eval by only +0.5 pt. v1 with *old* calibration still beat
   v2-qwentools with *new* calibration.
4. **Perplexity is useless here.** v2 scored *better* than BF16 (−0.07%) while
   having the worst agreement of the INT8 family.

### Final layout of v2-a16

```
W8A16 (INT8 weights, BF16 activations, pack-quantized)
  self_attn.q/k/v_proj          64 modules
  linear_attn.in_proj_*        192 modules
W8A8 (INT8 weights + INT8 dynamic per-token activations, int-quantized)
  self_attn.o_proj, mlp.gate/up/down_proj, linear_attn.out_proj
BF16 (untouched)
  lm_head, vision tower + merger, MTP head, GDN conv1d / A_log / dt_bias
```

---

## 3. Performance evaluation

Tool: `bench-ctx-sweep.sh` (single stream, 8/4/2 prompts by length, warmup
discarded, `--ignore-eos` so every request emits exactly 256 tokens).

Two metrics, and the distinction matters:

- **Prefill tok/s = input_len / mean TTFT.** The steady-state "input token
  throughput" vLLM prints averages over the whole window including decode, and
  understates prefill badly.
- **Decode tok/s = 1000 / mean TPOT.** Under MTP, `ITL` is per *streamed chunk*
  (~3.5 tokens), so ITL looks 3.5× slower than reality. TPOT is per token.

### Results, MTP on, prefix caching on, 16,384 batched tokens

**8× RTX 3060 (TP8, 100 W per card)**

| Context | Prefill W8A16 | v1 | **v2-a16** | Decode W8A16 | v1 | **v2-a16** |
|---|---|---|---|---|---|---|
| 128 | 690 | 654 | 569 | 99.6 | 99.9 | 104.1 |
| 1K | 874 | 985 | 954 | 109.9 | 103.7 | 115.9 |
| 4K | 898 | 1,033 | 1,030 | 94.1 | 84.4 | 91.2 |
| 8K | 899 | 1,039 | 1,036 | 83.3 | 73.0 | 81.6 |
| 16K | 893 | 1,030 | 1,029 | 62.0 | 56.5 | 61.4 |
| 32K | 875 | 1,007 | 1,007 | 46.1 | 42.9 | 45.4 |
| 64K | 842 | 964 | 963 | 28.1 | 26.8 | 28.1 |
| 131K | 784 | 888 | 888 | 16.8 | 16.4 | 16.9 |

v2-a16 = v1's prefill **and** W8A16's decode.

**CMP 170HX (single card)**

| Context | Prefill W8A16 | v1 | **v2-a16 (shrouded)** | Decode W8A16 | v1 | **v2-a16** |
|---|---|---|---|---|---|---|
| 1K | 1,642 | 2,840 | 3,352 | 119.3 | 99.5 | 102.1 |
| 4K | 1,810 | 3,556 | 3,693 | 103.5 | 92.9 | 107.3 |
| 8K | 1,755 | 3,574 | 3,644 | 97.3 | 86.2 | 91.7 |
| 16K | 1,461 | 3,211 | 3,509 | 76.6 | 67.1 | 74.8 |
| 32K | 1,366 | 2,745 | 3,089 | 63.3 | 59.4 | 63.9 |
| 64K | 1,172 | 2,044 | 2,443 | 43.3 | 40.9 | 45.9 |
| 131K | 971 | 1,437 | 1,727 | 27.4 | 27.2 | 27.9 |

**Thermals are a first-class variable.** Before a shroud was fitted, the same
card and build produced 2,597 / 1,924 / 1,381 / 948 at 16K–131K — a 35–82%
loss, growing with context length because long prefills mean sustained load.
Short contexts looked fine. Always sanity-check long-context prefill against
`nvidia-smi --query-gpu=temperature.gpu,clocks.sm,clocks_event_reasons.active`.

### Benchmarking rules learned the hard way

- **Restart the server before each sweep.** A warm prefix cache turns a sweep
  into a cache-hit test; a friend's 4×3090 run reported 14,556 tok/s "prefill"
  at 16K and a *lower* TTFT at 16K than 4K — the giveaway.
- **Check `AccLen` on the first row.** 1.00 means MTP is not running and decode
  is ~half; every number below it is meaningless for comparison.
- **Verify the served path and the selected kernel every time.** Two copies of a
  launch script in different directories cost us a full benchmark run against
  the wrong model.
- **Watch for stale environment variables.** `MAX_MODEL_LEN=8192` left over from
  a KLD session silently capped a production sweep and produced a table of zeros
  from 8K up.

---

## 4. Accuracy evaluation

Perplexity and mean KLD are **early-warning systems, not decision metrics**.
The decision metric is **confident top-1 agreement**: at positions where BF16 was
>90% certain, does the quant pick the same token? Flips there are capability
loss; flips at 51/49 positions are interchangeable word choice.

### Harness

`kld_eval.py collect` → teacher-forced scoring against a served model, saving
top-k logprobs to `.npz`; `report` ranks candidates against a BF16 reference.

Setup that made the numbers trustworthy:

- **Reference = the exact BF16 checkpoint that was quantized**, never an upstream base.
- **Held-out corpus.** `make_kld_eval_set.py` subtracts the calibration cache by
  text hash. This is not optional: streaming HF datasets ignore the shuffle seed,
  so a differently-seeded pool still overlapped **470 of 974 rows**.
- **Long enough documents.** `kld_eval.py` only takes documents ≥ `--seq-len`;
  our first pool had a median of 519 tokens and would have discarded most of it.
- **Identical serving flags for every collection**, including TP size: 8192
  context, `--max-logprobs 64`, `--max-num-seqs 1`, `--max-num-batched-tokens 1024`,
  `--gpu-memory-utilization 0.80`, no MTP, no prefix caching.
- **Two evals**: a general one (261,888 positions) and a native-Qwen tool-calling
  one (68,541 positions) built from held-out self-generated traces, because the
  agent workload is tool calls and generic text does not measure them.

### Results (BF16 reference, PPL 8.5765)

**General eval**

| Candidate | conf. agree | flips/10k | all-pos | Δ PPL | KLD mean | entropy |
|---|---|---|---|---|---|---|
| W8A16 | 99.888% | 5.38 | 98.32% | +0.43% | 0.014378 | −0.02% |
| **W8A8 v2-a16** | **99.521%** | 22.95 | 94.93% | +3.64% | 0.065910 | −1.77% |
| W8A8 v1 | 99.102% | 43.03 | 93.35% | +4.12% | 0.124400 | −1.14% |
| W8A8 v2 | 98.827% | 56.21 | 92.07% | −0.07% | 0.155567 | −1.26% |
| W8A8 v2-qwentools | 98.793% | 57.81 | 92.05% | +1.83% | 0.157736 | −2.28% |
| W4A16 (INT4 ref) | 98.608% | 66.67 | 91.43% | −4.71% | 0.138157 | +1.24% |

**Tool-calling eval**

| Candidate | conf. agree | flips/10k | all-pos | KLD mean | entropy |
|---|---|---|---|---|---|
| W8A16 | 99.576% | 22.76 | 95.87% | 0.044100 | +0.50% |
| **W8A8 v2-a16** | **97.927%** | 111.17 | 87.23% | 0.274336 | −3.32% |
| W8A8 v1 | 95.889% | 220.45 | 83.39% | 0.478978 | −3.35% |
| W8A8 v2 | 94.542% | 292.67 | 80.65% | 0.633556 | −6.36% |
| W4A16 (INT4 ref) | 93.927% | 325.64 | 79.60% | 0.632020 | +3.83% |

Reading: **>99.5% confident agreement is indistinguishable in practice.**
v2-a16 clears it on general text and lands far above the INT4 anchor on both.
Tool calling is the hardest domain for any W8A8 build — structured output has
low-entropy positions where a flip breaks a parse — and it is where W8A16 keeps
a real lead.

### Supporting checks

- **Needle recall** (`needle_recall_test.py`): one code at 5 depths × 3 lengths
  (32K/64K/131K) with 4 decoy records. W8A16 and W8A8 v1 both scored 15/15. This
  is a regression guard, not a discriminator — retrieval is an easy task.
- **Prefix-cache correctness** (`prefix_cache_test.py`): record with caching off,
  then on, compare token-by-token. Classifies EXACT / BENIGN (near-tie float
  noise) / FAIL (NaN, degenerate repetition, lost needle, non-tie divergence).
  W8A16 passed 89 EXACT / 7 BENIGN / 0 FAIL. **Must be re-run whenever the
  kernel path changes.**

---

## 5. How to replicate on a new model (Ampere)

### Phase 0 — Environment, pinned

1. vLLM from a commit with a published precompiled wheel; archive the wheel
   locally (`install_vllm_4090.sh` pattern: download + sha256 + `UV_CONSTRAINT`).
2. CUDA toolkit matching torch's CUDA build (13.0 for torch 2.13), toolkit-only,
   `apt-mark hold`. FlashInfer reads `nvcc` before torch; a mismatch breaks JIT.
3. `python3.X-dev` installed — Triton compiles a C helper at import time.
4. Verify: `check_vllm_fixes.sh` style script confirming relevant upstream
   patches are present in the tree (e.g. hybrid GDN/MTP fixes).

### Phase 1 — Profile the model before quantizing

```bash
python activation_outlier_probe.py /path/BF16 --texts calib.jsonl \
    --samples 8 --seq-len 2048 --out probe.json
```

Read the "by module kind" table. Any module with **INT8/FP8 ratio > ~4× or
max/median > ~100** is a candidate for 16-bit activations. Do not assume it is
`down_proj`; on this architecture it was the norm-fed projections.

Also dump the module tree (`--dump-modules`) and classify every Linear. Anything
printed as `OTHER` needs an explicit decision.

### Phase 2 — Build the calibration set

1. Write a YAML recipe weighted like your *production traffic*, not like a
   generic benchmark. Check the model card for what the vendor optimized for.
2. **Tool-calling data must be in the model's own template**, rendered with
   `apply_chat_template(..., tools=[...])`. Best: self-generate from the BF16
   model (`make_qwen_tool_calib.py selfgen`), which captures its real thinking
   blocks and call syntax.
3. `check_calibration_recipe.py` — validates every source exists, has the
   expected columns, and yields the requested count; writes a JSONL cache and
   its sha256. Re-run until it prints PASS.
4. Calibrate at the length you serve, not 2048. Recurrent layers accumulate
   state; short-only calibration never sees long-context activation ranges.

### Phase 3 — Quantize

```bash
# conservative first
python <model>_int8-w8a8.py /path/BF16 ./OUT-v1 \
    --calib-jsonl calib.jsonl --seqlen 8192 --variant v1 \
    --weight-strategy channel --iters 400 --nsamples 512

# then the mixed build informed by the probe
python <model>_int8-w8a8.py /path/BF16 ./OUT-v2-a16 \
    --calib-jsonl calib.jsonl --seqlen 8192 --variant v2 --a16 norm-fed ...
```

Rules:

- **Per-channel weights, dynamic per-token activations.** Group-wise INT8 routes
  to a pack-quantized path, not the CUTLASS INT8 GEMM. If you A/B a group size,
  use 128, not 32 — at 8 bits the fidelity gain is negligible and the scale
  overhead is real.
- **AutoRound, not GPTQ.** Sign-gradient rounding with 400 iterations; block
  losses converged to 0.004–0.009 on this model.
- **Smoke test with `--iters 1 --nsamples 32` first** — 3 minutes, catches API
  mismatches before a 3–6 hour run.

Post-save steps that are mandatory for hybrid + MTP models:

| Step | Why |
|---|---|
| Copy `mtp.*` from source, update the shard index | Transformers never loads or writes the MTP head |
| Add MTP modules **and** `re:^mtp.*` / `re:.*mtp.*` to `ignore` | Otherwise vLLM builds the head as W8A8, reads BF16 weights as INT8, and **acceptance is 0.00%** while the server looks healthy |
| For mixed builds: `format: mixed-precision`, per-group formats, distinct targets | AutoRound targets both groups at the class name `Linear`; vLLM keys schemes by target string and collapses them. W8A16 modules are stored `weight_packed` (pack-quantized), W8A8 as raw INT8 (int-quantized) |
| Preserve vision tower / `lm_head` in BF16 | Free, and keeps multimodal capability |

Verify before serving — a static check needs no GPU:

```bash
python resolve_ct_schemes.py config.json   # every layer -> intended scheme,
                                           # using vLLM's FUSED names
                                           # (qkv_proj, in_proj_qkvz, in_proj_ba)
```

### Phase 4 — Validate

1. **Serve and check kernels.** Mixed build must show *both*
   `CutlassInt8ScaledMMLinearKernel for CompressedTensorsW8A8Int8` and
   `...for CompressedTensorsWNA16`. One kernel means the split failed.
2. **`AccLen` ≈ 3.5** on the first bench row.
3. **Context sweep** vs the W8A16 baseline. Targets: prefill up, decode within
   ~10%, no failures.
4. **KLD, both evals**, plus the INT4 anchor if you want the scale.
   Ship gate: ≥99.5% confident agreement on general text.
5. **Prefix-cache test and needle recall**, because the kernel path changed.

### Phase 5 — Lock it

Archive next to the checkpoint: calibration JSONL + sha256, the recipe YAML,
build log, verify output, benchmark TSVs, `.npz` files, and the exact vLLM
commit + wheel sha256. Everything above is reproducible only if these travel
together.
