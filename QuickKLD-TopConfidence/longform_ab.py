"""
Longform degeneration A/B.

Teacher-forced agreement (kld_eval.py) puts the candidate back on the
reference's rails at every token, so it cannot see error compounding: a
slightly-wrong choice at token 500 leading somewhere worse by token 5,000. It
also cannot see spurious confidence — a quant that is sharper than BF16 scores
high agreement while writing flatter prose. The entropy column hints at it;
this harness measures it.

Two subcommands:

  generate   sample N long continuations per build from a served model
  score      compute degeneration metrics, compare builds pairwise

  # one server per build, same sampling params, same premises, same seeds
  python longform_ab.py generate --label w8a16 --model qwen38-27b \
      --base-url http://localhost:8080/v1 --api-key "$VLLM_API_KEY" \
      --tokenizer /path/BF16 --max-tokens 8000 --out lf-w8a16.json

  python longform_ab.py score lf-bf16.json lf-w8a16.json lf-fp8.json \
      --reference lf-bf16.json

Metrics (all higher = more repetitive unless noted):
  rep8         fraction of 8-grams that are repeats within the text
  distinct3    distinct 3-grams / total 3-grams        (higher = better)
  mattr500     moving-average type-token ratio, window 500 (higher = better)
  mtld         measure of textual lexical diversity     (higher = better)
  max_span     longest verbatim repeated token span (capped at 64: a value of
               64 means "64 or more", i.e. a large copied block)
  tail_loop    True if the last 400 tokens are dominated by a repeating cycle
  len_tokens   generated length (early stops matter: a model that quits at
               2k tokens never gets the chance to degenerate)
"""
import argparse
import collections
import json
import math
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request

PREMISES = [
    "Write a long short story about a lighthouse keeper who begins receiving letters postmarked from the future.",
    "Write a long short story about two rival cartographers mapping the same uncharted valley in 1890.",
    "Write a long short story about a night-shift hospital cleaner who can hear the building's memories.",
    "Write a long short story about a generation ship whose crew has forgotten the ship is moving.",
    "Write a long short story about a small-town locksmith who is asked to open a door that should not exist.",
    "Write a long short story about a translator hired to interpret for a delegation that speaks in weather.",
    "Write a long technical narrative: an engineer debugging an intermittent failure in a rocket's telemetry stack over three weeks.",
    "Write a long short story about a beekeeper whose hives begin arranging themselves into a written language.",
    "Write a long short story about a woman who inherits a house where every room runs at a different speed of time.",
    "Write a long short story about an archivist cataloguing the belongings of people who were never born.",
    "Write a long short story about a deep-sea welder working on a pipeline that is repairing itself.",
    "Write a long short story about a village that hires a stranger to remember things on its behalf.",
]

INSTRUCTION = (" Write at least 6000 words. Do not summarize, do not stop early, "
               "and do not write an outline; write the prose itself.")


# ----------------------------------------------------------------- generate
class Client:
    def __init__(self, base, key, model, timeout):
        self.base, self.key, self.model, self.timeout = base.rstrip("/"), key, model, timeout

    def chat(self, prompt, max_tokens, temperature, top_p, top_k, seed):
        body = {"model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens, "temperature": temperature,
                "top_p": top_p, "seed": seed}
        if top_k is not None:
            body["top_k"] = top_k
        req = urllib.request.Request(self.base + "/chat/completions",
                                     data=json.dumps(body).encode(), method="POST")
        req.add_header("Authorization", f"Bearer {self.key}")
        req.add_header("Content-Type", "application/json")
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            return {"error": f"HTTP {exc.code}: {exc.read().decode(errors='replace')[:200]}"}
        except Exception as exc:  # noqa: BLE001
            return {"error": repr(exc)}
        msg = data["choices"][0]["message"]
        return {"text": msg.get("content") or "",
                "reasoning": msg.get("reasoning_content") or "",
                "finish_reason": data["choices"][0].get("finish_reason"),
                "latency_s": round(time.time() - t0, 1)}


def cmd_generate(args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    cli = Client(args.base_url, args.api_key, args.model, args.timeout)

    premises = PREMISES[: args.premises]
    runs = []
    for i, premise in enumerate(premises):
        for rep in range(args.repeats):
            seed = args.seed + i * 100 + rep
            res = cli.chat(premise + INSTRUCTION, args.max_tokens,
                           args.temperature, args.top_p, args.top_k, seed)
            if "error" in res:
                print(f"  premise {i} rep {rep}: {res['error']}")
                continue
            n = len(tok(res["text"], add_special_tokens=False)["input_ids"])
            runs.append({"premise_index": i, "rep": rep, "seed": seed,
                         "text": res["text"], "tokens": n,
                         "finish_reason": res["finish_reason"],
                         "latency_s": res["latency_s"]})
            print(f"  premise {i} rep {rep}: {n} tokens, "
                  f"finish={res['finish_reason']}, {res['latency_s']}s")

    if not runs:
        sys.exit("ERROR: nothing generated")
    payload = {"label": args.label, "model": args.model,
               "time": time.strftime("%Y-%m-%d %H:%M:%S"),
               "sampling": {"temperature": args.temperature, "top_p": args.top_p,
                            "top_k": args.top_k, "max_tokens": args.max_tokens,
                            "seed_base": args.seed},
               "runs": runs}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    lens = [r["tokens"] for r in runs]
    print(f"\nwrote {len(runs)} generations -> {args.out}")
    print(f"length tokens: min {min(lens)} median {int(statistics.median(lens))} max {max(lens)}")


# -------------------------------------------------------------------- score
WORD_RE = re.compile(r"[A-Za-z']+")


def words(text):
    return [w.lower() for w in WORD_RE.findall(text)]


def ngram_repeat_rate(toks, n=8):
    if len(toks) < n + 1:
        return 0.0
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    counts = collections.Counter(grams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / len(grams)


def distinct_n(toks, n=3):
    if len(toks) < n + 1:
        return 0.0
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return len(set(grams)) / len(grams)


def mattr(toks, window=500):
    if len(toks) <= window:
        return len(set(toks)) / max(len(toks), 1)
    ratios = []
    counts = collections.Counter(toks[:window])
    ratios.append(len(counts) / window)
    for i in range(window, len(toks)):
        counts[toks[i]] += 1
        counts[toks[i - window]] -= 1
        if counts[toks[i - window]] == 0:
            del counts[toks[i - window]]
        ratios.append(len(counts) / window)
    return statistics.mean(ratios)


def mtld(toks, threshold=0.72):
    """Measure of Textual Lexical Diversity, bidirectional mean."""
    def one_pass(seq):
        factors, types, count = 0.0, set(), 0
        for w in seq:
            types.add(w)
            count += 1
            if len(types) / count <= threshold:
                factors += 1
                types, count = set(), 0
        if count:
            ttr = len(types) / count
            factors += (1 - ttr) / (1 - threshold) if threshold < 1 else 0
        return len(seq) / factors if factors else float(len(seq))
    if len(toks) < 50:
        return 0.0
    return (one_pass(toks) + one_pass(list(reversed(toks)))) / 2


def longest_repeated_span(toks, cap=4000):
    """Longest token span that appears at least twice (greedy, bounded)."""
    toks = toks[:cap]
    best = 0
    for n in range(4, 65):        # 64 = cap; a reported 64 means "at least 64"
        grams = {}
        hit = False
        for i in range(len(toks) - n + 1):
            g = tuple(toks[i:i + n])
            if g in grams:
                hit = True
                break
            grams[g] = i
        if hit:
            best = n
        else:
            break
    return best


def tail_loop(toks, tail=400, n=6):
    """True if the final `tail` tokens are dominated by repeating n-grams."""
    seg = toks[-tail:]
    if len(seg) < n * 4:
        return False
    return ngram_repeat_rate(seg, n) > 0.25


def score_text(text):
    toks = words(text)
    return {"len_words": len(toks),
            "rep8": ngram_repeat_rate(toks, 8),
            "distinct3": distinct_n(toks, 3),
            "mattr500": mattr(toks, 500),
            "mtld": mtld(toks),
            "max_span": longest_repeated_span(toks),
            "tail_loop": tail_loop(toks)}


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def cmd_score(args):
    runs_by_label = {}
    for path in args.files:
        payload = load(path)
        label = payload["label"]
        scored = []
        for run in payload["runs"]:
            s = score_text(run["text"])
            s.update({"premise_index": run["premise_index"], "rep": run["rep"],
                      "tokens": run["tokens"], "finish_reason": run.get("finish_reason")})
            scored.append(s)
        runs_by_label[label] = scored
        print(f"{label}: {len(scored)} generations from {os.path.basename(path)}")

    print(f"\n{'build':<18} {'gens':>5} {'words':>7} {'rep8':>8} {'distinct3':>10} "
          f"{'mattr500':>9} {'mtld':>7} {'max_span':>9} {'loops':>6}")
    print("-" * 92)
    summary = {}
    for label, rows in runs_by_label.items():
        m = lambda k: statistics.mean([r[k] for r in rows])
        loops = sum(r["tail_loop"] for r in rows)
        summary[label] = {"rep8": m("rep8"), "distinct3": m("distinct3"),
                          "mattr500": m("mattr500"), "mtld": m("mtld"),
                          "max_span": m("max_span"), "loops": loops, "n": len(rows),
                          "words": m("len_words")}
        print(f"{label:<18} {len(rows):>5} {m('len_words'):>7.0f} {m('rep8'):>8.4f} "
              f"{m('distinct3'):>10.4f} {m('mattr500'):>9.4f} {m('mtld'):>7.1f} "
              f"{m('max_span'):>9.1f} {loops:>3}/{len(rows)}")

    ref = args.reference and load(args.reference)["label"]
    if not ref or ref not in runs_by_label:
        print("\n(no --reference given, or its label is absent; skipping paired comparison)")
        return

    print(f"\n=== paired against {ref} (same premise + seed) ===")
    ref_rows = {(r["premise_index"], r["rep"]): r for r in runs_by_label[ref]}
    for label, rows in runs_by_label.items():
        if label == ref:
            continue
        pairs = [(ref_rows[(r["premise_index"], r["rep"])], r)
                 for r in rows if (r["premise_index"], r["rep"]) in ref_rows]
        if not pairs:
            print(f"{label}: no matching premise/seed pairs")
            continue
        print(f"\n{label}  ({len(pairs)} pairs)")
        for key, better in (("rep8", "lower"), ("distinct3", "higher"),
                            ("mattr500", "higher"), ("mtld", "higher"),
                            ("max_span", "lower")):
            deltas = [b[key] - a[key] for a, b in pairs]
            mean_d = statistics.mean(deltas)
            worse = sum((d > 0) if better == "lower" else (d < 0) for d in deltas)
            sd = statistics.pstdev(deltas) or 1e-9
            t = mean_d / (sd / math.sqrt(len(deltas)))
            print(f"  {key:<10} mean Δ {mean_d:+.4f}  ({better} is better)  "
                  f"worse on {worse}/{len(deltas)} pairs  t≈{t:+.2f}")
        a_loop = sum(a["tail_loop"] for a, _ in pairs)
        b_loop = sum(b["tail_loop"] for _, b in pairs)
        disc_b = sum(1 for a, b in pairs if b["tail_loop"] and not a["tail_loop"])
        disc_a = sum(1 for a, b in pairs if a["tail_loop"] and not b["tail_loop"])
        print(f"  tail_loop  {ref}: {a_loop}/{len(pairs)}  {label}: {b_loop}/{len(pairs)}  "
              f"(discordant {disc_b} vs {disc_a}; McNemar needs ~10+ discordant pairs "
              f"to say anything)")

    print("\nReminder: n is small and variance is high. This harness answers one "
          "question teacher-forced scoring cannot -- does anything degenerate over "
          "thousands of tokens -- and it cannot overturn a KLD ranking.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate")
    g.add_argument("--label", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--model", required=True)
    g.add_argument("--tokenizer", required=True)
    g.add_argument("--base-url", default="http://localhost:8080/v1")
    g.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    g.add_argument("--premises", type=int, default=12)
    g.add_argument("--repeats", type=int, default=1)
    g.add_argument("--max-tokens", type=int, default=8000)
    g.add_argument("--temperature", type=float, default=1.0)
    g.add_argument("--top-p", type=float, default=0.95)
    g.add_argument("--top-k", type=int, default=20)
    g.add_argument("--seed", type=int, default=1000)
    g.add_argument("--timeout", type=int, default=3600)
    g.set_defaults(func=cmd_generate)

    s = sub.add_parser("score")
    s.add_argument("files", nargs="+")
    s.add_argument("--reference", help="JSON whose label is the baseline for pairing")
    s.set_defaults(func=cmd_score)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
