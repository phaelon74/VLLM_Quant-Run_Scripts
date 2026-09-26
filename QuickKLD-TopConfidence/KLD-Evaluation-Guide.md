# KLD-Evaluation-Guide.md — measuring quantization damage properly

Companion to `Master-Ampere.md` §4 and `Master-Ada-Hopper.md` §4. This is the
full procedure for deciding whether a quantized checkpoint is good enough to
ship, using **confident top-1 agreement** as the decision metric and KLD as an
early-warning system.

Harness: `kld_eval.py` (`collect` / `compare` / `report`).
Helpers: `make_kld_eval_set.py`, `check_calibration_recipe.py`,
`make_qwen_tool_calib.py`.

---

## 1. What we measure, and why not the usual things

The question is narrow on purpose: **when the BF16 weights are certain about the
next token, does the quantized checkpoint make the same call?** Not "is this
model good" — that is what task benchmarks are for, and they cannot resolve
differences this small.

### Perplexity is blind to reshuffling

PPL only looks at the probability assigned to the one token that actually came
next. A quant can hold PPL flat (or improve it) while reordering everything
below the argmax, which is what shapes sampled generation.

Real example from this project: **W8A8 v2 scored PPL −0.07% vs BF16 — better
than the reference — while having the worst confident agreement of the INT8
family (98.827%).** FP8 on Ada scored −4.13% PPL and was the second-best build
overall. PPL's sign carries no information here.

What PPL *is* good for: catching a structurally broken export. A miswired export
in earlier work scored PPL 16,329 and 1.388% agreement. Both metrics scream.

### KLD is entropy-confounded and tail-dominated

Mean KLD measures how far the whole distribution moved, in nats. Its magnitude
scales with how wide the reference distribution was to begin with, so creative
writing looks "worse" than arithmetic for intrinsic reasons.

It is also dominated by rare catastrophic positions. W8A8 v2 had a **median KLD
of 0.0069 and a p99 of 4.4 nats**, with 2.55% of positions above 1 nat — most
tokens fine, a small set badly wrong. That shape made its mean KLD (0.156) rank
*below* an INT4 build (0.138) even though v2 beat INT4 on confident agreement.

Use KLD as a gate ("is this export sane?"), not as a ranking.

### Confident top-1 agreement

At each scored position, compare argmaxes. Then **stratify by the reference's own
confidence**, because a single agreement number conflates two different events:

- BF16 was split 51/49 between two synonyms and the quant picked the other one.
  Costs nothing.
- BF16 was 97% certain and the quant picked something else. That is capability
  loss — a decision the unquantized model considered settled.

Read the tiers bottom-up. From the FP8 Ada build:

| tier | agreement | positions |
|---|---|---|
| all positions | 95.60% | 261,888 |
| reference p > 0.5 | ~98% | — |
| reference p > 0.9 | **99.599%** | ~125,000 |

Ranking on the all-positions number would have made this look mediocre when
nearly all of its disagreement is the harmless kind.

**`flips / 10k`** expresses the same thing as a rate over *all* scored positions
(subset sizes differ between candidates, so percentages of different
denominators are not comparable). It also maps onto something physical:
confident decisions changed per ten thousand tokens written.

### Entropy: the one thing agreement cannot see

Agreement is one-directional. It catches the candidate failing to reproduce the
reference's confidence, but not the candidate becoming *more* confident where
the reference was uncertain. A quant that is sharper than BF16 can score high
agreement while producing flatter, more repetitive prose. Negative entropy delta
= narrowed. Watch it.

Observed: FP8 Ada **+0.32%** (did not narrow), INT8 v2-a16 **−1.77%**,
INT8 v2-qwentools **−2.28%** (worst).

### What this harness structurally cannot see

Every position is scored given the **reference's** prefix, so the candidate is
put back on the rails at every token and never allowed to drift. It cannot see
the failure where a slightly-wrong choice at token 500 leads somewhere worse by
token 5,000. That needs a sampled longform A/B, which is a different harness.

---

## 2. Decision thresholds

| Signal | Reading |
|---|---|
| **conf. agree** | The decision metric. **>99.5% is indistinguishable in practice.** Rank on this. |
| **flips/10k** | Same thing, physical. Single digits excellent; 20-25 fine; >60 is INT4 territory |
| **all-pos top-1** | Mostly interchangeable word choice. Do not rank on it |
| **entropy** | Negative = narrowed = flatter, more repetitive prose |
| **mean KLD** | <0.01 excellent, 0.01-0.05 good, 0.05-0.15 noticeable, >0.15 expect visible loss in long generations |
| **KLD p99** | Matters more than the mean for prose: the tail where the quant changed its mind |
| **ref mass in cand top-k** | Below 98% means `-k` was too small and the KLD tail is unreliable |

Anchor the scale with an **INT4 W4A16 build from the same base weights**. Absolute
KLD bands mean little on their own; "better or worse than the INT4 people run
productively" is a judgement readers and you can both act on. Ours scored
98.608% / 66.67 flips, and every 8-bit build beat it.

---

## 3. Building the evaluation set

The eval text decides what you are measuring. Three requirements.

### 3.1 Held out from calibration — verify, do not assume

Streaming HuggingFace datasets **ignore the shuffle seed** and hand back the same
first N rows every time. A differently-seeded pool still overlapped **470 of 974
rows** with our calibration cache. Subtract by exact text hash:

```bash
# over-collect a pool with a different seed
python check_calibration_recipe.py calibrate_<model>.yaml \
  --tokenizer /path/BF16 --seqlen 8192 --seed 1234 --sample-scale 2.0 \
  --out pool_seed1234.jsonl

# subtract calibration, filter by real token length, label buckets
python make_kld_eval_set.py pool_seed1234.jsonl \
  --exclude calib_general_8k.jsonl \
  --tokenizer /path/BF16 --seq-len 1024 --target-windows 256 --per-bucket 48 \
  --out <model>_eval_8192.jsonl

# prove it
python -c "import json,hashlib; d=lambda p:{hashlib.sha1(json.loads(l)['text'].strip().encode()).hexdigest() for l in open(p) if l.strip()}; print('overlap:', len(d('<model>_eval_8192.jsonl') & d('calib_general_8k.jsonl')))"
# overlap: 0
```

### 3.2 Documents must be at least `--seq-len` tokens

`kld_eval.py` slices non-overlapping windows and **discards any document shorter
than the window**. Our first pool had a median of 519 tokens and would have
thrown away most of itself. `make_kld_eval_set.py --tokenizer ... --seq-len 1024`
filters by real token count and projects the window yield before you serve
anything:

```
token filter (>= 1024): kept 144, dropped 269 (kld_eval.py would reject those outright)
projected windows at --seq-len 1024: 474 (484,902 scored positions)
collect with: --seq-len 1024 --max-seqs 256
```

### 3.3 A second eval for structured output

Generic text does not measure tool calling, and tool calling is where quants
hurt most. Build a domain-specific eval from **self-generated** traces so the
syntax matches what you serve:

```bash
# render prompts through the model's own template with tool schemas, then let
# BF16 continue them raw (/v1/completions, so <think> and <tool_call> survive)
python make_qwen_tool_calib.py selfgen --tokenizer /path/BF16 \
  --model qwen38-bf16 --base-url http://localhost:8080/v1 --api-key "$VLLM_API_KEY" \
  --num 120 --two-turn --out qwen_tool_selfgen.jsonl

# split: half to calibration, half held out as the eval
python - <<'EOF'
import json, random
rows=[json.loads(l) for l in open("qwen_tool_selfgen.jsonl")]
random.Random(7).shuffle(rows)
open("qwen_tool_selfgen_calib.jsonl","w").write("\n".join(json.dumps(r) for r in rows[:60]))
open("qwen_tool_eval.jsonl","w").write("\n".join(
    json.dumps({"text":r["text"],"bucket":"tool_use_qwen"}) for r in rows[60:]))
EOF
```

**Why this matters:** our first tool eval used Hermes-format text while the model
serves Qwen-format tool calls, so it was scoring a syntax the agent never emits.
Scores on the native-format eval are 1-2 points lower and far more informative.

### 3.4 Bucket labels

`make_kld_eval_set.py` attaches a `bucket` per row (code, tool_use, reasoning,
professional, long_doc, creative, chat, multilingual) so `compare` can break
every metric down by domain. Read the per-domain table with window counts in
mind — buckets are not equally sampled, since only long documents survive the
length filter.

---

## 4. Serving configuration (identical for every collection)

The reference and every candidate must be scored under **byte-identical
conditions**. Numerics change with batch composition and all-reduce ordering.

| Flag | Value | Why |
|---|---|---|
| `--max-logprobs` | **64** | vLLM's default of 20 is too coarse for a stable KLD tail; must be >= `-k` |
| `--max-model-len` | 8192 | Only the window size matters; keeps memory free |
| `--max-num-seqs` | **1** | Prompt logprobs at k=64 materialize a large logits tensor per chunk |
| `--max-num-batched-tokens` | **1024** | One window per forward pass; smallest logits spike |
| `--gpu-memory-utilization` | **0.80** | Leaves headroom for that spike. 0.92 OOMs |
| speculative decoding | **off** | Logprob semantics under MTP verification are not worth trusting |
| prefix caching | **off** | Determinism |
| tensor-parallel size | same for all | TP changes all-reduce ordering, therefore numerics |
| `--language-model-only` | on | Matches how you serve; skips the vision tower |
| `--kv-cache-dtype` | same for all | Do not mix BF16 and FP8 KV between collections |

Example (adjust path/name only):

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_NO_USAGE_STATS=1 \
vllm serve /path/to/BF16 \
  --served-model-name qwen38-bf16 \
  --tensor-parallel-size 8 --dtype bfloat16 --trust-remote-code --language-model-only \
  --max-model-len 8192 --max-num-seqs 1 --max-num-batched-tokens 1024 \
  --max-logprobs 64 --gpu-memory-utilization 0.80 \
  --no-enable-prefix-caching --mamba-cache-mode align \
  --host 0.0.0.0 --port 8080 2>&1 | tee server-bf16-kld.log
```

Verify before every collection — this is the step that saved six GPU-hours:

```bash
curl -s -H "Authorization: Bearer $VLLM_API_KEY" localhost:8080/v1/models \
  | python -c "import json,sys;d=json.load(sys.stdin)['data'][0];print(d['id'],'->',d.get('root'))"
grep -o "'max_logprobs': [0-9]*" server-*-kld.log | head -1        # 64
grep -o "'speculative_config'" server-*-kld.log | head -1           # nothing = MTP off
grep -oE "(Selected|Using) [A-Za-z0-9]+ (Kernel )?for [A-Za-z0-9]+" server-*-kld.log | sort | uniq -c
```

A served *name* is a label and can point anywhere after a re-serve. Check the
`root` path, not the name.

---

## 5. Collecting

Reference first, then every candidate, **changing only `--model` and `--out`**:

```bash
mkdir -p kld
export TOK=/path/to/BF16      # the BF16 source tokenizer, for EVERY collection

python kld_eval.py collect --model qwen38-bf16 \
  --base-url http://localhost:8080/v1 --api-key "$VLLM_API_KEY" --concurrency 1 \
  --tokenizer $TOK --texts <model>_eval_8192.jsonl --out kld/bf16.npz \
  --seq-len 1024 --max-seqs 256 -k 64
```

The corpus header must be identical across runs:

```
documents tokenized            94
doc tokens                     min 1029  median 1850  max 21344
windows                        256 x 1024 tokens
scored positions               261888
```

A different count means the tokenizer or the text file differs, and the
comparison is invalid. `compare` also refuses to score files with different
position counts.

Repeat for the tool eval (`--texts qwen_tool_eval.jsonl --out kld/tool_*.npz`),
and for each candidate.

**Cross-rig control.** If the reference and a candidate were collected on
different machines (e.g. BF16 needs 54 GB and will not fit on the candidate's
card), collect one shared checkpoint on *both* rigs. Ours:

| Candidate | Rig / TP | conf. agree | flips/10k | KLD |
|---|---|---|---|---|
| W8A16 | 8x3060, TP8 | 99.888% | 5.38 | 0.014378 |
| W8A16 | 4090, TP1 | 99.888% | 5.38 | 0.014644 |

Identical to three decimals, so cross-rig noise is effectively zero and the FP8
gap measured against the remote reference is real. **Never skip this control**
when rigs differ; without it you cannot separate quantization damage from
environment.

---

## 6. Reporting and reading

```bash
# rank every candidate against the reference
python kld_eval.py report kld/bf16.npz kld/w8a16.npz kld/w8a8_v1.npz \
  kld/w8a8_v2_a16.npz kld/fp8_4090.npz kld/w4a16.npz

# per-domain detail for one candidate
python kld_eval.py compare kld/bf16.npz kld/w8a8_v2_a16.npz
```

Globbing avoids the classic mistake of feeding a main-eval file into the tool
report (it gets skipped for scoring different positions):

```bash
python kld_eval.py report kld/bf16.npz $(ls kld/*.npz | grep -v '/tool_' | grep -v 'kld/bf16')
python kld_eval.py report kld/tool_bf16.npz $(ls kld/tool_*.npz | grep -v 'tool_bf16')
```

### Worked example — the general eval

| Candidate | conf. agree | flips/10k | all-pos | Δ PPL | KLD mean | entropy |
|---|---|---|---|---|---|---|
| W8A16 | 99.888% | 5.38 | 98.32% | +0.43% | 0.014378 | −0.02% |
| FP8 W8A8 (Ada) | 99.599% | 19.21 | 95.60% | −4.13% | 0.050629 | +0.32% |
| INT8 W8A8 v2-a16 | 99.521% | 22.95 | 94.93% | +3.64% | 0.065910 | −1.77% |
| INT8 W8A8 v1 | 99.102% | 43.03 | 93.35% | +4.12% | 0.124400 | −1.14% |
| INT8 W8A8 v2 | 98.827% | 56.21 | 92.07% | −0.07% | 0.155567 | −1.26% |
| INT4 W4A16 | 98.608% | 66.67 | 91.43% | −4.71% | 0.138157 | +1.24% |

How to read that:

1. **Rank on column 1.** Both shipping builds clear 99.5%; v1 and v2 do not.
2. **Sanity-check with KLD.** Nothing is above 0.16, so no export is broken.
3. **Ignore the PPL column's sign.** v2 "beat" BF16 and is the second-worst build.
4. **Check entropy before shipping a prose model.** v2-a16 at −1.77% narrowed;
   FP8 did not.
5. **Use the INT4 row as the floor.** Every 8-bit build sits above it.

### Worked example — the domain breakdown

From `compare` on v2:

| domain | KLD mean | KLD p99 | top-1 | top-1 where ref p>0.9 |
|---|---|---|---|---|
| reasoning | 0.026506 | 0.199289 | 95.87% | 99.89% |
| creative | 0.076180 | 0.847016 | 89.32% | 99.95% |
| long_doc | 0.078335 | 1.087636 | 92.40% | 99.27% |
| professional | 0.141491 | 4.221960 | 92.76% | 98.98% |
| code | 0.235207 | 7.116040 | 93.38% | 98.72% |
| **tool_use** | **0.574808** | **10.729678** | **83.56%** | **95.36%** |

This is what an aggregate hides. Reasoning was nearly untouched; tool calling
was 20× worse on mean KLD. At 95.36% near-certain agreement, roughly 1 in 22
confident decisions changes inside tool-call generation, where a single wrong
token breaks a parse. That single table is what sent us back to fix calibration
format and then activation precision.

The two top-1 columns are the fairer cross-domain read, because creative writing
is intrinsically higher-entropy and some of its KLD is not damage.

---

## 7. Pitfalls, each of which bit us

| Symptom | Cause | Fix |
|---|---|---|
| `Connection refused` | harness defaults to port 8000 | pass `--base-url http://localhost:8080/v1` |
| `401 Unauthorized` | harness defaults `--api-key EMPTY`; vLLM picks up `VLLM_API_KEY` from the environment automatically | pass `--api-key "$VLLM_API_KEY"` |
| `404 model does not exist` | `--model` must match `--served-model-name` | check `/v1/models` |
| `400 prompt logprobs of 64 > max allowed: 20` | `--max-logprobs` never reached the server | put it on the `vllm serve` line; check `'max_logprobs': 64` in the log |
| CUDA OOM during collection | k=64 prompt logprobs spike | `--gpu-memory-utilization 0.80`, `--max-num-seqs 1`, `--max-num-batched-tokens 1024`, `--concurrency 1` |
| `SKIPPED: scored different positions` | wrong npz in the report (main vs tool) | glob by prefix |
| `ref mass inside cand top-k` < 98% | `-k` too small for that candidate | re-collect at `-k 128` with `--max-logprobs 128` |
| Numbers look flattering | eval overlaps calibration | hash-subtract; streaming sources ignore seeds |
| Most of the corpus rejected | documents shorter than `--seq-len` | filter with `make_kld_eval_set.py --tokenizer` |
| Candidate scores oddly well/poorly | served path is not what you think | grep `'model_tag'` and `root` |

---

## 8. Full run sheet

```
[ ] 1. Build eval set(s), prove zero overlap with calibration, confirm projected windows >= 256
[ ] 2. Decide the serving config; write it down; use it for EVERY collection
[ ] 3. Serve BF16 reference -> collect main + tool -> kld/bf16.npz, kld/tool_bf16.npz
[ ] 4. Serve each candidate -> collect main + tool (identical flags)
[ ] 5. If rigs differ: collect one shared checkpoint on both -> cross-rig control
[ ] 6. Optional: an INT4 W4A16 build from the same base weights, as the scale anchor
[ ] 7. report on both evals; compare on the shipping candidate for per-domain detail
[ ] 8. Ship gate: >= 99.5% confident agreement on the general eval,
        no domain below ~95% near-certain agreement,
        entropy not meaningfully negative,
        mean KLD < 0.15
[ ] 9. Archive: eval JSONLs, all .npz, the report output, serving flags, vLLM commit
```

---

## 9. What to do when a build fails the gate

In order of expected value:

1. **Run the activation probe** (`activation_outlier_probe.py`) and look for
   modules where INT8/FP8 error ratio > 4× or max/median > 100. Move those to
   16-bit activations. This was worth +0.69 pt on the general eval and +2.85 pt
   on tools — more than everything else combined.
2. **Check the domain table.** If one domain is far worse, it is usually a
   calibration-format mismatch, not a fundamental limit.
3. **Fix calibration to match production formats** (chat template, tool syntax,
   thinking blocks). Worth about +0.5 pt on the affected domain here.
4. **Reduce what you quantize** (leave recurrent projections in BF16, as in v1).
5. **More calibration samples / iterations** last: it moved the least of anything
   we tried.
