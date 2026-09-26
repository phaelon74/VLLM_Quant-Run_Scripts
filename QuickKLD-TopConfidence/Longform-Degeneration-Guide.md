# Longform-Degeneration-Guide.md — the failure KLD cannot see

Companion to `KLD-Evaluation-Guide.md`. Harness: `longform_ab.py`
(`generate` / `score`).

---

## 1. Why this exists

`kld_eval.py` is teacher-forced: every position is scored **given the
reference's prefix**, so the candidate is put back on the rails at each token
and never allowed to drift. That makes it deterministic, high-n and the right
thing to rank on — and structurally blind to two failure modes:

**Error compounding.** A slightly-wrong choice at token 500 leading somewhere
worse by token 5,000. Teacher forcing erases it by construction.

**Spurious confidence.** Agreement is one-directional: it catches the candidate
failing to reproduce the reference's confident decisions, but not the candidate
becoming confident where the reference was uncertain. A quant sharper than BF16
can score excellent agreement while writing flatter, more repetitive prose.

The `entropy` column in the KLD report is the early warning. Ours:

| Build | entropy vs BF16 | conf. agree | reading |
|---|---|---|---|
| INT8 W8A16 | −0.02% | 99.888% | unchanged width |
| **FP8 W8A8 (Ada)** | **+0.32%** | 99.599% | did not narrow |
| INT8 W8A8 v2-a16 (Ampere) | **−1.77%** | 99.521% | narrowed |
| INT8 W8A8 v2-qwentools | −2.28% | 98.793% | narrowed most |
| INT4 W4A16 | +1.24% | 98.608% | widened (different failure) |

Two builds with nearly identical agreement (99.599% vs 99.521%) sit on opposite
sides of the entropy line. That is the gap this guide measures.

**A worked precedent from earlier NVFP4 work:** a quant scored its per-position
entropy **0.43% wider** than BF16 — teacher-forced, it looked marginally *more*
diverse. Left to generate 15,000-word stories on its own it produced narrower,
more repetitive prose on every measure: repeated 8-gram rate 1.909% → 2.185%,
MATTR-500 0.5428 → 0.5380, MTLD 101.5 → 97.0, invented person-names per 10k
words 28.8 → 55.0, and detectable repetition loops in 8 of 12 stories against
BF16's 5. A strict superset, with no premise where the quant held together and
BF16 did not.

So: entropy tells you *where to look*; it does not tell you what happens.

---

## 2. What this harness can and cannot conclude

| | Teacher-forced (kld_eval) | Longform A/B (this) |
|---|---|---|
| Sampling | none, deterministic | yes, high variance |
| n | 261,888 positions | 12–36 generations |
| Answers | "does it reproduce confident decisions?" | "does anything degenerate over thousands of tokens?" |
| Status | **primary; rank on it** | secondary; a veto, not a ranking |

Twelve sampled stories cannot overturn 261,888 deterministic comparisons. They
can reveal a failure those comparisons are blind to. In the NVFP4 case the
5-versus-8 loop difference had an exact two-sided McNemar p of 0.25 —
directionally clean, statistically underpowered. Treat a result here as a
**flag for investigation**, not a verdict, unless the effect is large and
consistent across premises.

---

## 3. Metrics

All computed on whitespace/alphabetic word tokens, not model tokens, so results
are comparable across tokenizers.

| Metric | Direction | What it catches |
|---|---|---|
| `rep8` | lower better | fraction of 8-grams that are repeats — the blunt repetition signal |
| `distinct3` | higher better | distinct 3-grams / total — local phrasing variety |
| `mattr500` | higher better | moving-average type-token ratio, window 500 — vocabulary richness, length-independent |
| `mtld` | higher better | measure of textual lexical diversity — how far the text goes before lexical variety collapses |
| `max_span` | lower better | longest verbatim repeated token span, capped at 64 (a reported 64 means "64 or more": a copied block) |
| `tail_loop` | False better | are the final 400 tokens dominated by a repeating cycle? The classic terminal loop |
| `len_words` | context | a model that stops at 2k tokens never gets the chance to degenerate; compare lengths before believing a "win" |

Validated against synthetic text:

| Text | rep8 | distinct3 | mattr500 | mtld | max_span | tail_loop |
|---|---|---|---|---|---|---|
| random/healthy | 0.0000 | 1.000 | 0.770 | 620 | 0 | False |
| phrase-repetitive | 0.4022 | 0.459 | 0.379 | 24.6 | 64 | True |
| healthy body + looping tail | 0.1463 | 0.853 | 0.699 | 43.9 | 64 | **True** |

Note the third row: body metrics stay healthy while `tail_loop` fires. That is
why the terminal-loop flag is separate from the aggregate rates.

Metrics deliberately **not** included: "invented person-names per 10k words"
(needs a name gazetteer and is model-family specific), self-BLEU across
generations (dominated by premise similarity), and any LLM-judge score (adds a
second model's biases to a measurement about distribution width).

---

## 4. Protocol

### 4.1 Hold everything constant except the build

- **Same premises, same seeds, same sampling parameters** for every build. The
  harness pairs generations by `(premise_index, rep)` so each comparison is
  within-premise.
- **Production sampling settings**, not greedy. Temperature 0 hides exactly the
  failure you are looking for: flattening shows up in how the tail of the
  distribution is sampled. Defaults here are Qwen's thinking-mode
  recommendation: temperature 1.0, top_p 0.95, top_k 20.
- **Same server flags** as your real deployment, including MTP. Speculative
  decoding is distribution-preserving, so it does not change what is sampled.
- **Long enough to fail.** 6,000–8,000 tokens minimum. Degeneration that appears
  at 15,000 words will not appear at 800.
- **Check `finish_reason`.** A build that stops early is not "less repetitive";
  it is shorter. Compare `len_words` first.

### 4.2 Sample size

- 12 premises × 1 repeat = **12 pairs**: the practical minimum, enough to see a
  gross effect (8 loops vs 5).
- 12 × 3 = **36 pairs**: enough for a McNemar test on loop counts to be worth
  computing. Roughly 10+ *discordant* pairs are needed before the p-value means
  much.
- Generation cost at 8k tokens and ~60 tok/s decode is about 2.2 minutes per
  story: 12 pairs ≈ 30 min per build, 36 pairs ≈ 90 min per build.

### 4.3 Which builds to compare

Always include **BF16 as the reference**. Then at minimum the shipping
candidate and the build with the opposite entropy sign — for us:

```
BF16  vs  FP8 W8A8 (+0.32%)  vs  INT8 W8A8 v2-a16 (−1.77%)
```

Optionally W8A16 (entropy −0.02%) as a "known-good quant" control: if it also
shows a degeneration delta, your harness or your premises are the cause, not
the quantization.

---

## 5. Running it

### Generate (one server per build, same flags as production)

```bash
python longform_ab.py generate --label bf16 --model qwen38-bf16 \
  --base-url http://localhost:8080/v1 --api-key "$VLLM_API_KEY" \
  --tokenizer /path/BF16 \
  --premises 12 --repeats 1 --max-tokens 8000 \
  --temperature 1.0 --top-p 0.95 --top-k 20 --seed 1000 \
  --out lf-bf16.json
```

Repeat with `--label fp8 --model qwen38-fp8 --out lf-fp8.json`, and so on.
**Keep `--seed`, `--premises`, `--repeats` and all sampling flags identical**;
only `--label`, `--model` and `--out` change.

The BF16 reference needs a rig that can host it (54 GB for a 27B), so this may
mean a different machine than the candidate. That is acceptable here in a way it
is not for KLD: sampled text is being compared on aggregate style statistics,
not on token-level numerics.

### Score

```bash
python longform_ab.py score lf-bf16.json lf-fp8.json lf-v2a16.json \
  --reference lf-bf16.json
```

Output: a per-build summary table, then a paired section per candidate showing
mean delta, how many pairs got worse, a rough t statistic, and the loop counts
with discordant-pair counts for McNemar.

---

## 6. Reading the result

**Look at, in order:**

1. **`len_words` parity.** If one build writes 30% less, stop and fix the
   prompt/`max_tokens` before comparing anything else.
2. **`tail_loop` counts and discordance.** The most interpretable signal:
   "loops in 8 of 12 vs 5 of 12". Note whether the discordant pairs are
   one-sided (a strict superset is far more convincing than a wash).
3. **`rep8` and `mtld`, paired.** "Worse on 11/12 pairs" is more informative
   than the mean, because premises differ wildly in intrinsic repetitiveness.
4. **`max_span` = 64.** Means a verbatim block of 64+ words was copied
   somewhere in the text. On a healthy generation this is 0–10.
5. **`mattr500`.** Slow, steady narrowing rather than outright looping. This is
   the metric that most closely tracks the KLD `entropy` column.

**Thresholds** (from the NVFP4 precedent, treat as rules of thumb):

| Signal | Benign | Investigate | Do not ship for prose |
|---|---|---|---|
| rep8 delta vs BF16 | < +0.2 pt | +0.2 to +0.5 pt | > +0.5 pt |
| MTLD delta | > −3% | −3% to −8% | > −8% |
| MATTR-500 delta | > −0.5% | −0.5% to −1.5% | > −1.5% |
| tail_loop | ≤ BF16 count | +1 to +2 | +3 or more, one-sided |

**What a clean result looks like:** deltas near zero with pairs split roughly
half/half, loop counts within one of BF16, and no `max_span` saturation. That
is the outcome that lets you ship a build whose KLD entropy was negative.

---

## 7. If a build fails here but passed KLD

The finding is real but narrow: the build is fine at reproducing individual
decisions and worse at sustaining a long generation. Options, in order:

1. **Ship it for non-prose work anyway.** Agentic tool calling, extraction,
   classification and RAG answers rarely run past 1,000 tokens, and the
   teacher-forced result governs there. Restrict the build to those endpoints.
2. **Reduce what you quantize.** The narrowing came from somewhere; the
   activation probe (`activation_outlier_probe.py`) tells you which modules are
   losing the most, and moving them to 16-bit activations is the same fix that
   raised agreement.
3. **Re-check sampling settings in production.** A build that narrows the
   distribution interacts badly with aggressive top-k. Loosening top_p/top_k for
   that build is a legitimate mitigation, though it changes behaviour.
4. **Prefer the other format.** In our case FP8 on Ada did not narrow at all
   (+0.32%), so on hardware where both are options, this metric is a tiebreak.

---

## 8. Run sheet

```
[ ] 1. Pick builds: BF16 reference + shipping candidate + opposite-entropy build
[ ] 2. Fix premises, seeds, sampling params, max_tokens; write them down
[ ] 3. Generate per build (12 pairs minimum, 36 if you want McNemar to speak)
[ ] 4. Check len_words parity and finish_reason before scoring anything
[ ] 5. Score with --reference; read tail_loop, then paired rep8/mtld/mattr
[ ] 6. Record the result next to the KLD report; they answer different questions
[ ] 7. State n and variance whenever you quote it
```
