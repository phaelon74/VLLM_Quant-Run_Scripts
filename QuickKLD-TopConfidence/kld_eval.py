#!/usr/bin/env python3
"""Measure how far a quantized checkpoint's output distribution drifts from BF16.

KL divergence against the unquantized model is the metric that actually
separates two quant recipes. Perplexity alone hides damage: a quant can keep
PPL flat while reshuffling everything below the argmax, which is exactly what
degrades long-form generation. So this reports both, plus top-1 agreement.

Teacher-forced and deterministic. Prompts are sent as token IDs, not text, so
both runs score byte-identical positions and no tokenizer drift can creep in
between the BF16 reference and the candidate.

Two phases, because the BF16 model and the candidate rarely fit at once:

    collect   score a corpus against one served model, save logprobs to .npz
    compare   read two .npz files, report PPL, KLD, and agreement

The server must be started with --max-logprobs >= the -k used here; vLLM's
default of 20 is too coarse for a stable KLD tail.

Usage:
    # 1. serve BF16, collect the reference
    python tools/kld_eval.py collect \
        --base-url http://localhost:8000/v1 --model behemoth-bf16 \
        --tokenizer /media/fmodels/TheDrummer/Behemoth-R1-123B-v2 \
        --texts data/eval/behemoth_holdout.jsonl \
        --out /media/fmodels2/working_Model-Opt/kld/bf16.npz

    # 2. serve the NVFP4 export, collect the candidate (same --texts, same -k)
    python tools/kld_eval.py collect ... --out .../nvfp4_omlp.npz

    # 3. compare
    python tools/kld_eval.py compare .../bf16.npz .../nvfp4_omlp.npz
"""

import argparse
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

PAD_ID = -1


def line(label, value, note=""):
    print(f"  {label:<30} {value}{('   ' + note) if note else ''}")


# ---------------------------------------------------------------------------
# collect


def as_token_list(encoded):
    """Normalize whatever apply_chat_template returned into a flat list of ints.

    transformers 5 flipped apply_chat_template's return_dict default to True,
    so it hands back a BatchEncoding rather than a list. len() on that is the
    number of dict keys, which silently makes every document look 2 tokens long
    and yields zero windows with no error anywhere.
    """
    if hasattr(encoded, "keys") or isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    if encoded and isinstance(encoded[0], (list, tuple)):
        encoded = encoded[0]
    return list(encoded)


def load_windows(texts_path, tokenizer_dir, seq_len, max_seqs):
    """Tokenize the corpus into fixed-length windows of exactly seq_len tokens.

    Fixed length keeps every position equally weighted in the averages and
    makes the two runs trivially comparable. Short documents are dropped rather
    than padded; padding would score meaningless positions.

    Returns (windows, bucket label per window, rejected count, per-document
    token lengths). The lengths are only used to explain a zero-window result.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=False)

    windows, labels, rejected, doc_lens = [], [], [0], []
    with open(texts_path, encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            row = json.loads(raw)
            # "messages" rows are rendered through the chat template so the
            # scored tokens match what the model sees when actually served;
            # scoring raw text would measure a distribution nobody uses.
            if row.get("messages"):
                # Mistral v7 rejects non-alternating roles, which ShareGPT
                # sources do contain. Skip those rows instead of aborting a run
                # that is otherwise fine.
                try:
                    ids = as_token_list(
                        tok.apply_chat_template(row["messages"], tokenize=True))
                except Exception:
                    rejected[0] += 1
                    continue
            elif row.get("text"):
                ids = tok(row["text"], add_special_tokens=False)["input_ids"]
            else:
                continue

            doc_lens.append(len(ids))
            bucket = row.get("bucket") or "unlabeled"
            for start in range(0, len(ids) - seq_len + 1, seq_len):
                windows.append(ids[start:start + seq_len])
                labels.append(bucket)
                if len(windows) >= max_seqs:
                    return windows, labels, rejected[0], doc_lens
    return windows, labels, rejected[0], doc_lens


def score_window(session, args, ids):
    """Return (actual logprobs, top-k ids, top-k logprobs) for one window.

    Position 0 has no prediction, so every array is seq_len-1 long.
    """
    k = args.k
    resp = session.post(
        args.base_url.rstrip("/") + "/completions",
        # Token IDs, not text: the server must score exactly the positions we
        # tokenized, or the two runs are not comparable.
        json={"model": args.model, "prompt": ids, "max_tokens": 1,
              "temperature": 0.0, "echo": False, "prompt_logprobs": k},
        headers={"Authorization": f"Bearer {args.api_key}"},
        timeout=args.timeout,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"{resp.status_code} from server: {resp.text[:400]}")

    plp = resp.json()["choices"][0].get("prompt_logprobs")
    if plp is None:
        raise RuntimeError(
            "server returned no prompt_logprobs; it is likely too old or was "
            "started without --max-logprobs")

    n = len(ids) - 1
    actual = np.zeros(n, dtype=np.float32)
    top_ids = np.full((n, k), PAD_ID, dtype=np.int32)
    top_lp = np.full((n, k), -np.inf, dtype=np.float32)

    for pos in range(1, len(ids)):
        entry = plp[pos] or {}
        # Keys arrive as ints from the python client and strings over raw JSON.
        pairs = sorted(((int(t), v["logprob"] if isinstance(v, dict) else v.logprob)
                        for t, v in entry.items()),
                       key=lambda p: -p[1])
        i = pos - 1
        want = ids[pos]
        for tid, lp in pairs:
            if tid == want:
                actual[i] = lp
                break
        for slot, (tid, lp) in enumerate(pairs[:k]):
            top_ids[i, slot] = tid
            top_lp[i, slot] = lp

    return actual, top_ids, top_lp


def cmd_collect(args):
    import requests

    print("\n=== corpus ===")
    windows, labels, rejected, doc_lens = load_windows(
        args.texts, args.tokenizer, args.seq_len, args.max_seqs)
    line("texts", args.texts)
    line("documents tokenized", len(doc_lens))
    if doc_lens:
        line("doc tokens", f"min {min(doc_lens)}  median "
                           f"{int(np.median(doc_lens))}  max {max(doc_lens)}")
    if rejected:
        line("rejected by chat template", rejected, "non-alternating roles")

    if not windows:
        sys.exit(f"\n  no document reaches {args.seq_len} tokens, so no window "
                 f"could be cut.\n  Lower --seq-len, or check the numbers above: "
                 f"a median of 1-3 tokens means the\n  tokenizer returned a dict "
                 f"rather than a token list, not that the corpus is short.")

    line("windows", f"{len(windows)} x {args.seq_len} tokens")
    line("scored positions", len(windows) * (args.seq_len - 1))

    session = requests.Session()

    print("\n=== scoring ===")
    line("endpoint", args.base_url)
    line("model", args.model)
    line("top-k", args.k)

    results = [None] * len(windows)
    done = 0

    def run(i):
        return i, score_window(session, args, windows[i])

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for i, out in pool.map(run, range(len(windows))):
            results[i] = out
            done += 1
            if done % max(1, len(windows) // 20) == 0 or done == len(windows):
                print(f"  {done}/{len(windows)} windows", flush=True)

    actual = np.concatenate([r[0] for r in results])
    top_ids = np.concatenate([r[1] for r in results])
    top_lp = np.concatenate([r[2] for r in results])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        actual_lp=actual,
        top_ids=top_ids,
        top_lp=top_lp,
        prompt_ids=np.asarray(windows, dtype=np.int32),
        buckets=np.asarray(labels),
        meta=json.dumps({"model": args.model, "seq_len": args.seq_len,
                         "k": args.k, "texts": str(args.texts)}),
    )

    print("\n=== result ===")
    line("perplexity", f"{float(np.exp(-actual.mean())):.4f}")
    line("saved", str(out))


# ---------------------------------------------------------------------------
# compare


def logsumexp(a, axis):
    peak = np.max(a, axis=axis, keepdims=True)
    peak = np.where(np.isfinite(peak), peak, 0.0)
    return np.squeeze(peak, axis=axis) + np.log(
        np.exp(a - peak).sum(axis=axis))


def confident_flip_rate(kept_top1, ref_conf, floor_p=0.9):
    """Flips at near-certain positions, expressed per 10,000 generated tokens.

    Percentages of a subset are hard to compare across runs whose subset sizes
    differ. A rate over all scored positions is directly comparable and maps
    onto something concrete: how often a confident decision changes while
    writing.
    """
    sure = ref_conf > floor_p
    flips = int((~kept_top1[sure]).sum())
    return flips, flips / len(kept_top1) * 10000


def score_pair(ref, cand):
    """All per-position metrics for one candidate against the reference.

    Split out so `compare` and `report` cannot drift apart: a summary table that
    computes agreement even slightly differently from the detailed view is worse
    than no summary table.
    """
    ref_ids, cand_ids = ref["top_ids"], cand["top_ids"]

    # Mask padded slots so they contribute no probability mass.
    ref_lp = np.where(ref_ids == PAD_ID, -np.inf, ref["top_lp"])
    cand_lp = np.where(cand_ids == PAD_ID, -np.inf, cand["top_lp"])

    n = len(ref_lp)
    kld = np.empty(n, dtype=np.float64)
    covered = np.empty(n, dtype=np.float64)
    ref_ent = np.empty(n, dtype=np.float64)
    cand_ent = np.empty(n, dtype=np.float64)

    # Chunked: the id-matching broadcast is [chunk, k, k] and would otherwise
    # allocate tens of GB at corpus scale.
    step = 4096
    for s in range(0, n, step):
        e = min(s + step, n)
        r_ids, r_lp = ref_ids[s:e], ref_lp[s:e].astype(np.float64)
        c_ids, c_lp = cand_ids[s:e], cand_lp[s:e].astype(np.float64)

        match = r_ids[:, :, None] == c_ids[:, None, :]
        found = match.any(axis=2)
        slot = match.argmax(axis=2)
        q_lp = np.take_along_axis(c_lp, slot, axis=1)

        # A reference token outside the candidate's top-k gets the candidate's
        # smallest observed logprob. That is the most generous available bound,
        # so a reported KLD is a floor, never an exaggeration.
        floor = np.min(np.where(np.isfinite(c_lp), c_lp, np.inf), axis=1)
        q_lp = np.where(found, q_lp, floor[:, None])
        q_lp = np.where(np.isfinite(r_lp), q_lp, -np.inf)

        log_p = r_lp - logsumexp(r_lp, axis=1)[:, None]
        log_q = q_lp - logsumexp(q_lp, axis=1)[:, None]
        p = np.exp(log_p)

        terms = np.where(p > 0, p * (log_p - log_q), 0.0)
        kld[s:e] = terms.sum(axis=1)
        covered[s:e] = np.where(found, p, 0.0).sum(axis=1)

        # Entropy of each model's own top-k, renormalized. Agreement is
        # one-directional: it catches the candidate failing to reproduce the
        # reference's confidence, but not the candidate becoming confident where
        # the reference was not. That shows up here, as lower entropy.
        ref_ent[s:e] = -np.where(p > 0, p * log_p, 0.0).sum(axis=1)
        log_qn = c_lp - logsumexp(c_lp, axis=1)[:, None]
        qn = np.exp(log_qn)
        cand_ent[s:e] = -np.where(qn > 0, qn * log_qn, 0.0).sum(axis=1)

    idx = np.arange(n)
    kept_top1 = (ref_ids[idx, np.argmax(ref_lp, axis=1)]
                 == cand_ids[idx, np.argmax(cand_lp, axis=1)])

    return {
        "n": n,
        "kld": kld,
        "covered": covered,
        "kept_top1": kept_top1,
        # ref_lp are true logprobs, not renormalized, so this is the real
        # top-1 probability under the reference model.
        "ref_conf": np.exp(np.max(ref_lp, axis=1)),
        "ref_entropy": ref_ent,
        "cand_entropy": cand_ent,
        "ref_ppl": float(np.exp(-ref["actual_lp"].mean())),
        "cand_ppl": float(np.exp(-cand["actual_lp"].mean())),
    }


def position_buckets(ref, args):
    """Per-position bucket labels, or None if they cannot be established.

    Prefers labels saved at collect time. Falls back to re-deriving them from
    the eval JSONL, which makes .npz files collected before --keep-bucket
    existed usable without re-serving the reference model -- but only if the
    re-derived windows are token-identical to the saved ones. Mislabelled
    domains would be worse than no domain breakdown, so a mismatch reports and
    gives up rather than guessing.
    """
    n_seq, seq_len = ref["prompt_ids"].shape

    if "buckets" in ref.files:
        labels = [str(x) for x in ref["buckets"]]
    elif args.texts and args.tokenizer:
        windows, labels, _, _ = load_windows(args.texts, args.tokenizer,
                                             seq_len, n_seq)
        if not np.array_equal(np.asarray(windows, dtype=np.int32),
                              ref["prompt_ids"]):
            print("\n  note: --texts re-tokenized to different windows than the "
                  "reference was\n  scored on, so no domain breakdown. Rebuild "
                  "the eval JSONL with the same\n  seed, or re-collect with "
                  "--keep-bucket data.")
            return None
    else:
        return None

    if len(labels) != n_seq or set(labels) == {"unlabeled"}:
        return None
    return np.repeat(np.asarray(labels), seq_len - 1)


def cmd_compare(args):
    ref = np.load(args.reference, allow_pickle=False)
    cand = np.load(args.candidate, allow_pickle=False)
    ref_meta = json.loads(str(ref["meta"]))
    cand_meta = json.loads(str(cand["meta"]))

    print("\n=== inputs ===")
    line("reference", f"{ref_meta['model']}", str(args.reference))
    line("candidate", f"{cand_meta['model']}", str(args.candidate))

    if not np.array_equal(ref["prompt_ids"], cand["prompt_ids"]):
        sys.exit("prompt token IDs differ; the two runs did not score the same "
                 "positions and the comparison would be meaningless")
    line("positions", len(ref["actual_lp"]), "identical prompts")

    m = score_pair(ref, cand)
    kld, covered = m["kld"], m["covered"]
    kept_top1, ref_conf, n = m["kept_top1"], m["ref_conf"], m["n"]

    # This is the headline, printed first and on purpose. KL divergence measures
    # how far the distribution moved; agreement measures how often the decision
    # changed, and generation is a chain of decisions. Stratifying by the
    # reference's confidence separates the two failure modes that a single
    # agreement number conflates: a flip where BF16 was split 51/49 is style,
    # a flip where BF16 was at 90% is the model getting something wrong.
    print("\n=== token decision agreement  (capability proxy) ===")
    line("top-1 match, all positions", f"{kept_top1.mean() * 100:6.2f}%",
         f"{n} pos")
    for floor_p, label in ((0.5, "confident"), (0.9, "near-certain")):
        sel = ref_conf > floor_p
        if sel.sum():
            line(f"top-1 match where ref p>{floor_p}",
                 f"{kept_top1[sel].mean() * 100:6.2f}%",
                 f"{int(sel.sum())} pos, {label}")
    flips, per10k = confident_flip_rate(kept_top1, ref_conf)
    line("confident flips / 10k tokens", f"{per10k:6.2f}",
         f"{flips} total, one per ~{int(10000 / per10k) if per10k else 0} tokens")
    print("  Read the bottom rows first. Flips at near-certain positions are")
    print("  capability loss; flips at low-confidence positions are mostly")
    print("  interchangeable word choice and cost you little.")
    print("  The rate is per generated token, but scoring is teacher-forced, so")
    print("  it counts single decisions -- not the trajectory drift that follows")
    print("  one, which compounds over a long generation.")

    # One-directional agreement blind spot: the candidate becoming confident
    # where the reference was not.
    r_ent, c_ent = m["ref_entropy"].mean(), m["cand_entropy"].mean()
    print("\n=== output entropy  (flattening check) ===")
    line("reference mean entropy", f"{r_ent:.4f} nats")
    line("candidate mean entropy", f"{c_ent:.4f} nats",
         f"{(c_ent / r_ent - 1) * 100:+.2f}%")
    print("  Agreement cannot see spurious confidence. A candidate several")
    print("  percent below the reference has narrowed its distribution, which")
    print("  reads as flatter, more repetitive prose regardless of agreement.")

    ref_ppl, cand_ppl = m["ref_ppl"], m["cand_ppl"]

    print("\n=== perplexity ===")
    line("reference", f"{ref_ppl:.4f}")
    line("candidate", f"{cand_ppl:.4f}")
    line("ratio", f"{cand_ppl / ref_ppl:.4f}", f"{(cand_ppl / ref_ppl - 1) * 100:+.2f}%")

    print("\n=== KL divergence (nats) ===")
    line("mean", f"{kld.mean():.6f}")
    line("median", f"{np.median(kld):.6f}")
    line("p90", f"{np.percentile(kld, 90):.6f}")
    line("p99", f"{np.percentile(kld, 99):.6f}")
    line("max", f"{kld.max():.6f}")

    # A max over hundreds of thousands of positions is a single token and says
    # nothing about how often the model actually diverges. The thresholds do.
    print("\n=== divergence tail ===")
    for thresh in (0.05, 0.10, 0.25, 0.50, 1.00):
        share = float((kld > thresh).mean())
        line(f"positions > {thresh:.2f} nats",
             f"{share * 100:6.3f}%", f"{int(share * n):>8} of {n}")

    per_pos = position_buckets(ref, args)
    if per_pos is not None:
        print("\n=== by domain ===")
        print(f"  {'':<24} {'KLD mean':>9} {'KLD p99':>9} "
              f"{'top-1':>7} {'top-1 p>0.9':>12}")
        for name in sorted(set(per_pos.tolist())):
            sel = per_pos == name
            k_sel = kld[sel]
            sure = sel & (ref_conf > 0.9)
            sure_txt = (f"{kept_top1[sure].mean() * 100:11.2f}%"
                        if sure.sum() else f"{'n/a':>12}")
            print(f"  {name:<24} {k_sel.mean():9.6f} "
                  f"{np.percentile(k_sel, 99):9.6f} "
                  f"{kept_top1[sel].mean() * 100:6.2f}% {sure_txt}")
        print("  An aggregate cannot separate lost reasoning from lost prose")
        print("  style. This can. Creative writing is inherently higher-entropy,")
        print("  so some of its KLD is intrinsic rather than damage -- which is")
        print("  why the two top-1 columns are the fairer cross-domain read.")

    print("\n=== measurement quality ===")
    line("ref mass inside cand top-k", f"{covered.mean() * 100:.3f}%",
         "low values mean -k was too small")

    print("\n=== reading ===")
    print("  mean KLD  <0.01 excellent, 0.01-0.05 good, 0.05-0.15 noticeable,")
    print("            >0.15 expect visible quality loss in long generations.")
    print("  p99 matters more than the mean for creative writing: it is the")
    print("  tail of positions where the quant changed its mind.")
    print("  Neither number is a capability measure. For that, compare the")
    print("  near-certain agreement rate above across variants, and run a task")
    print("  benchmark -- see install-eval-bench_CreativeWriting.md.")

    if covered.mean() < 0.98:
        print("\n  WARNING: candidate top-k covers only "
              f"{covered.mean() * 100:.2f}% of reference mass; re-collect with "
              "a larger -k for a trustworthy KLD.")


# ---------------------------------------------------------------------------
# report


def cmd_report(args):
    """One row per candidate, ranked on the metric that matters.

    Format-agnostic by construction: candidates are just .npz files collected
    from served endpoints, so NVFP4, INT4 W4A16, FP8 and anything else compare
    on equal terms as long as they scored the same positions.
    """
    ref = np.load(args.reference, allow_pickle=False)
    ref_name = json.loads(str(ref["meta"]))["model"]

    print("\n=== reference ===")
    line("model", ref_name, str(args.reference))
    line("positions", len(ref["actual_lp"]))

    rows = []
    for path in args.candidates:
        cand = np.load(path, allow_pickle=False)
        name = json.loads(str(cand["meta"]))["model"]
        if not np.array_equal(ref["prompt_ids"], cand["prompt_ids"]):
            print(f"\n  SKIPPED {name}: scored different positions than the "
                  f"reference.")
            continue
        m = score_pair(ref, cand)
        sure = m["ref_conf"] > 0.9
        flips, per10k = confident_flip_rate(m["kept_top1"], m["ref_conf"])
        rows.append({
            "name": name,
            # A corpus with no near-certain position yields an empty slice.
            # Sorting and printing NaN would silently look like a real score.
            "sure_agree": (m["kept_top1"][sure].mean() * 100
                           if sure.any() else float("nan")),
            "sure_n": int(sure.sum()),
            "per10k": per10k,
            "all_agree": m["kept_top1"].mean() * 100,
            "ppl_delta": (m["cand_ppl"] / m["ref_ppl"] - 1) * 100,
            "kld": m["kld"].mean(),
            "ent_delta": (m["cand_entropy"].mean() / m["ref_entropy"].mean()
                          - 1) * 100,
            "covered": m["covered"].mean() * 100,
        })

    if not rows:
        sys.exit("no candidate scored the same positions as the reference")

    # Best confident agreement first: that is the decision metric, so it should
    # not require reading down a column to find. NaN sorts last.
    rows.sort(key=lambda r: (math.isnan(r["sure_agree"]), -r["sure_agree"]))

    if not rows[0]["sure_n"]:
        print("\n  note: no position reached >90% reference confidence, so there "
              "is no\n  confident-agreement column to rank on. Ranking falls back "
              "to mean KLD.")
        rows.sort(key=lambda r: r["kld"])

    print("\n=== ranked by confident agreement ===")
    print(f"  {'candidate':<28} {'conf.agree':>10} {'flips/10k':>10} "
          f"{'all':>7} {'PPL':>8} {'KLD':>9} {'entropy':>8}")
    for r in rows:
        agree = (f"{r['sure_agree']:9.3f}%" if not math.isnan(r["sure_agree"])
                 else f"{'n/a':>10}")
        print(f"  {r['name'][:28]:<28} {agree} "
              f"{r['per10k']:10.2f} {r['all_agree']:6.2f}% "
              f"{r['ppl_delta']:+7.2f}% {r['kld']:9.6f} "
              f"{r['ent_delta']:+7.2f}%")

    print("\n=== reading ===")
    print("  conf.agree  how often the candidate reproduced a decision the")
    print("              reference made at >90% confidence. THE metric here.")
    print("  flips/10k   the same thing as a rate: confident decisions changed")
    print("              per 10,000 generated tokens. Easier to reason about.")
    print("  all         agreement over every position. Mostly counts")
    print("              interchangeable word choice; do not rank on it.")
    print("  entropy     candidate vs reference distribution width. Negative")
    print("              means narrowed -> flatter, more repetitive prose.")
    print("  KLD         entropy-confounded, so weak across domains, but the")
    print("              best early warning that an export is structurally bad.")

    worst = min(r["covered"] for r in rows)
    if worst < 98.0:
        print(f"\n  WARNING: one candidate covers only {worst:.2f}% of reference "
              f"mass; re-collect with a larger -k.")


# ---------------------------------------------------------------------------

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
sub = ap.add_subparsers(dest="cmd", required=True)

c = sub.add_parser("collect", help="Score a corpus against one served model.")
c.add_argument("--base-url", default="http://localhost:8000/v1")
c.add_argument("--api-key", default="EMPTY")
c.add_argument("--model", required=True, help="Model name as served.")
c.add_argument("--tokenizer", required=True,
               help="Path used to tokenize locally. Both runs must use the same one.")
c.add_argument("--texts", required=True,
               help="JSONL with a 'messages' or 'text' field. Use held-out data, "
                    "not the calibration set.")
c.add_argument("--out", required=True, help="Destination .npz.")
c.add_argument("--seq-len", type=int, default=1024)
c.add_argument("--max-seqs", type=int, default=256)
c.add_argument("-k", type=int, default=64,
               help="Top-k logprobs per position. Server needs --max-logprobs >= this.")
c.add_argument("--concurrency", type=int, default=8)
c.add_argument("--timeout", type=int, default=600)
c.set_defaults(func=cmd_collect)

m = sub.add_parser("compare", help="Compare two collected .npz files.")
m.add_argument("reference", help=".npz from the BF16 model.")
m.add_argument("candidate", help=".npz from the quantized model.")
m.add_argument("--texts", default=None,
               help="Eval JSONL carrying 'bucket' labels. Only needed to get a "
                    "per-domain breakdown out of .npz files collected before "
                    "labels were saved; verified against the stored prompt IDs.")
m.add_argument("--tokenizer", default=None, help="Required with --texts.")
m.set_defaults(func=cmd_compare)

r = sub.add_parser("report", help="Rank many candidates against one reference.")
r.add_argument("reference", help=".npz from the BF16 model.")
r.add_argument("candidates", nargs="+",
               help="One .npz per quantized checkpoint. Any format.")
r.set_defaults(func=cmd_report)

args = ap.parse_args()
args.func(args)
