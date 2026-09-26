#!/usr/bin/env python3
"""
Prefix-caching correctness test for vLLM on hybrid GDN/attention models (+MTP).

Idea: send the exact same token-level request sequence to the server twice,
once launched with prefix caching OFF (reference) and once with it ON
(everything else identical), then compare every response.

Step 1  server launched with ENABLE_PREFIX_CACHING=0
    python prefix_cache_test.py record --label off --out pc-off.json

Step 2  relaunch with ENABLE_PREFIX_CACHING=1 (same STAGE, same everything)
    python prefix_cache_test.py record --label on --out pc-on.json \
        --rounds 2 --compare pc-off.json

What it exercises
  A  exact repeat of long prompts (full cache hit)               8K/32K/64K (+128K with --long)
  B  shared long prefix, different question                      8K/32K
  C  prefill ending 0,1,4,7,10,B/2,B-1 tokens past a cache block
     boundary, then a follow-up turn that hits (vllm#55766 shape)
  D  6-turn agent-style conversation, recalls facts from earlier turns
  E  two requests racing on a shared prefix; a cache hit arriving
     while another request is mid-decode

How each response is judged against the OFF reference
  EXACT    identical tokens
  BENIGN   diverged at a near-tie (<= 1.0 nat) -- normal float noise from
           a different prefill split, not corruption
  FAIL     HTTP error, NaN/inf logprobs, degenerate repetition, lost needle
           recall, or divergence where the reference token was not a close
           contender (state corruption signature)
"""
import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

DEFAULT_TOKENIZER = "/media/fmodels/lued/Qwen3.8-27B-INT8-W8A16-MTP/"

IM_USER = "<|im_start|>user\n"
IM_END = "<|im_end|>\n"
GEN = "<|im_start|>assistant\n<think>\n\n</think>\n\n"

NEAR_TIE_NATS = 1.0

WORDS = (
    "the a of and to in is was for on that with as by at from this be are "
    "or it an which were but not have has had one two three new first after "
    "system report river market engine council harbor archive signal budget "
    "winter garden station library circuit museum journal network valley "
    "committee survey protocol shipment ledger bridge tower forest island "
    "measured reviewed delayed approved recorded shifted repaired expanded "
    "northern quiet careful rapid formal local annual regional technical "
    "during before between across under within against without toward "
    "however meanwhile therefore although because while since unless "
    "engineers auditors farmers pilots clerks tenants volunteers analysts "
    "copper grain timber cotton glass paper steel salt wool coal "
    "morning evening season quarter decade century interval schedule"
).split()

VAULTS = ["amber", "birch", "cobalt", "delta", "ember", "falcon", "garnet",
          "harbor", "indigo", "juniper", "kestrel", "lumen", "marble", "nectar"]


# ---------------------------------------------------------------- tokenizer

def load_tokenizer(path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


class Prompter:
    def __init__(self, tok):
        self.tok = tok
        self.user = self.enc(IM_USER)
        self.end = self.enc(IM_END)
        self.gen = self.enc(GEN)

    def enc(self, s):
        return list(self.tok.encode(s, add_special_tokens=False))

    def first_turn(self, hay_ids, question):
        return self.user + hay_ids + self.enc(question) + self.end + self.gen

    def overhead(self, question):
        return len(self.first_turn([], question))

    def next_turn(self, prev_ids, assistant_text, user_ids):
        return prev_ids + self.enc(assistant_text) + self.end + self.user + user_ids + self.end + self.gen


def filler_ids(tok, seed, n):
    rng = random.Random(seed)
    ids = []
    while len(ids) < n:
        sents = []
        for _ in range(200):
            s = " ".join(rng.choice(WORDS) for _ in range(rng.randint(8, 18)))
            sents.append(s[0].upper() + s[1:] + ".")
        ids.extend(tok.encode(" ".join(sents) + " ", add_special_tokens=False))
    return ids[:n]


def make_needles(seed, k):
    rng = random.Random(seed * 7919 + 17)
    return [(name, str(rng.randint(100000, 999999))) for name in rng.sample(VAULTS, k)]


def needle_text(name, code):
    return f" Important record: the access code for vault {name} is {code}. "


def question_text(name):
    return (f"\n\nWhat is the access code for vault {name}? Reply with only the "
            f"six-digit code, then one short sentence about the document.")


def haystack(tok, seed, length, needles, depths):
    nids = [tok.encode(needle_text(n, c), add_special_tokens=False) for n, c in needles]
    base = filler_ids(tok, seed, max(length - sum(map(len, nids)), 0))
    out, prev = [], 0
    for ids, d in sorted(zip(nids, depths), key=lambda x: x[1]):
        cut = int(d * len(base))
        out += base[prev:cut] + list(ids)
        prev = cut
    return out + base[prev:]


# ---------------------------------------------------------------- cases

def req(role, ids, expect, max_tokens=None, group=None):
    return {"role": role, "ids": ids, "expect": expect, "max_tokens": max_tokens, "group": group}


def build_cases(tok, pb, block, long_ctx):
    cases, seed = [], 1000

    # A: exact repeat
    for L in [8192, 32768, 65536] + ([131072] if long_ctx else []):
        seed += 1
        nd = make_needles(seed, 3)
        q = question_text(nd[1][0])
        ids = pb.first_turn(haystack(tok, seed, L - pb.overhead(q), nd, [0.2, 0.5, 0.8]), q)
        cases.append({"id": f"A_exact_repeat_{L // 1024}k", "requests": [
            req("primer", ids, nd[1][1]), req("measured", ids, nd[1][1])]})

    # B: shared prefix, different question
    for L in [8192, 32768]:
        seed += 1
        nd = make_needles(seed, 3)
        q1, q2 = question_text(nd[0][0]), question_text(nd[2][0])
        hay = haystack(tok, seed, L - pb.overhead(q1), nd, [0.15, 0.5, 0.85])
        cases.append({"id": f"B_shared_prefix_{L // 1024}k", "requests": [
            req("primer", pb.first_turn(hay, q1), nd[0][1]),
            req("measured", pb.first_turn(hay, q2), nd[2][1])]})

    # C: block-boundary offsets
    if block and block >= 16:
        m = max(1, math.ceil(4096 / block))
        offs = sorted({0, 1, 4, 7, 10, block // 2, block - 1})
        targets = [(f"B{block}_m{mm}_off{o}", mm * block + o) for mm in (m, m + 2) for o in offs]
    else:
        targets = [(f"stride{k}", 4096 + k * 211) for k in range(12)]
    for tag, target in targets:
        seed += 1
        nd = make_needles(seed, 2)
        q = question_text(nd[0][0])
        primer = pb.first_turn(haystack(tok, seed, target - pb.overhead(q), nd, [0.3, 0.7]), q)
        follow = pb.next_turn(primer, nd[0][1] + ".", pb.enc(question_text(nd[1][0]).strip()))
        cases.append({"id": f"C_boundary_{tag}", "requests": [
            req("primer", primer, nd[0][1]), req("measured", follow, nd[1][1])]})

    # D: multi-turn agent chain
    seed += 1
    nd = make_needles(seed, 6)
    q = question_text(nd[0][0])
    prompt = pb.first_turn(haystack(tok, seed, 16000 - pb.overhead(q), nd[:1], [0.5]), q)
    reqs = [req("measured", prompt, nd[0][1])]
    last_answer = nd[0][1]
    for t in range(1, 6):
        section = haystack(tok, seed + 101 * t, 1500, [nd[t]], [0.5])
        ask = nd[max(0, t - 2)]
        user_ids = pb.enc("Here is another section of the document:\n") + section + \
            pb.enc(question_text(ask[0]).strip())
        prompt = pb.next_turn(prompt, last_answer + ".", user_ids)
        reqs.append(req("measured", prompt, ask[1]))
        last_answer = ask[1]
    cases.append({"id": "D_agent_chain_16k", "requests": reqs})

    # E: concurrency
    seed += 1
    nd = make_needles(seed, 3)
    q = question_text(nd[0][0])
    hay = haystack(tok, seed, 16000 - pb.overhead(q), nd, [0.2, 0.5, 0.8])
    seed += 1
    nd2 = make_needles(seed, 1)
    q2 = question_text(nd2[0][0])
    other = pb.first_turn(haystack(tok, seed, 4096 - pb.overhead(q2), nd2, [0.5]), q2)
    cases.append({"id": "E_concurrent_16k", "requests": [
        req("race", pb.first_turn(hay, question_text(nd[0][0])), nd[0][1], group="g1"),
        req("race", pb.first_turn(hay, question_text(nd[1][0])), nd[1][1], group="g1"),
        req("long_decode", other, nd2[0][1], max_tokens=384, group="g2"),
        req("measured", pb.first_turn(hay, question_text(nd[2][0])), nd[2][1], group="g2")]})
    return cases


# ---------------------------------------------------------------- client

class Client:
    def __init__(self, base, key, model, max_tokens):
        self.base, self.key, self.model = base.rstrip("/"), key, model
        self.max_tokens, self.logprobs = max_tokens, True

    def _open(self, path, body=None, timeout=3600):
        data = json.dumps(body).encode() if body is not None else None
        r = urllib.request.Request(self.base + path, data=data, method="POST" if body is not None else "GET")
        r.add_header("Authorization", f"Bearer {self.key}")
        if data:
            r.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return resp.read().decode()

    def healthy(self):
        try:
            self._open("/health", timeout=10)
            return True
        except Exception:
            return False

    def max_model_len(self):
        try:
            data = json.loads(self._open("/v1/models", timeout=30))["data"]
            return int(data[0].get("max_model_len") or 0) or None
        except Exception:
            return None

    def metrics(self):
        try:
            txt = self._open("/metrics", timeout=30)
        except Exception:
            return None
        out = {"hits": 0.0, "queries": 0.0, "found": False, "block_size": None, "mamba_block_size": None}
        for line in txt.splitlines():
            if line.startswith("#"):
                continue
            m = re.match(r"^vllm:prefix_cache_(hits|queries)(_total)?(\{[^}]*\})?\s+([0-9.eE+-]+)$", line)
            if m:
                out["found"] = True
                out[m.group(1)] += float(m.group(4))
            if "cache_config_info" in line:
                for k in ("mamba_block_size", "block_size"):
                    mm = re.search(rf'(?<![a-z_]){k}="(\d+)"', line)
                    if mm and out[k] is None:
                        out[k] = int(mm.group(1))
        return out

    def complete(self, ids, max_tokens=None):
        body = {"model": self.model, "prompt": ids, "max_tokens": max_tokens or self.max_tokens,
                "temperature": 0.0, "seed": 0, "return_tokens_as_token_ids": True}
        if self.logprobs:
            body["logprobs"] = 5
        t0 = time.time()
        try:
            r = json.loads(self._open("/v1/completions", body))
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:400]
            if self.logprobs and "logprob" in msg.lower():
                print("  (server rejected logprobs; falling back to token-text comparison)")
                self.logprobs = False
                return self.complete(ids, max_tokens)
            return {"error": f"HTTP {e.code}: {msg}"}
        except Exception as e:
            return {"error": repr(e)}
        c = r["choices"][0]
        lp = c.get("logprobs") or {}
        return {"text": c.get("text", ""), "tokens": lp.get("tokens"),
                "token_logprobs": lp.get("token_logprobs"), "top_logprobs": lp.get("top_logprobs"),
                "finish_reason": c.get("finish_reason"), "latency_s": round(time.time() - t0, 2)}


# ---------------------------------------------------------------- record

def ids_sha(ids):
    return hashlib.sha1(",".join(map(str, ids)).encode()).hexdigest()[:16]


def summarize(rq, res, hit):
    res = dict(res)
    res.update({"role": rq["role"], "prompt_len": len(rq["ids"]), "sha": ids_sha(rq["ids"]),
                "expect": rq["expect"], "cache_hit_tokens": hit})
    res["needle_ok"] = (rq["expect"] in res.get("text", "")) if "error" not in res else False
    return res


def run_case(cl, case):
    out, reqs, i = [], case["requests"], 0
    while i < len(reqs):
        g = reqs[i]["group"]
        batch = [reqs[i]]
        if g is not None:
            while i + len(batch) < len(reqs) and reqs[i + len(batch)]["group"] == g:
                batch.append(reqs[i + len(batch)])
        before = cl.metrics()
        if len(batch) == 1:
            results = [cl.complete(batch[0]["ids"], batch[0]["max_tokens"])]
        else:
            with ThreadPoolExecutor(len(batch)) as ex:
                futs = []
                for rq in batch:
                    futs.append(ex.submit(cl.complete, rq["ids"], rq["max_tokens"]))
                    time.sleep(0.5)
                results = [f.result() for f in futs]
        after = cl.metrics()
        hit = None
        if before and after and before["found"]:
            hit = int(after["hits"] - before["hits"])
        for rq, res in zip(batch, results):
            out.append(summarize(rq, res, hit if len(batch) == 1 else None))
            if len(batch) > 1:
                out[-1]["group_hit_tokens"] = hit
        i += len(batch)
    return {"requests": out}


def record(args):
    tok = load_tokenizer(args.tokenizer)
    pb = Prompter(tok)
    cl = Client(args.base_url, args.api_key, args.model, args.max_tokens)
    if not cl.healthy():
        sys.exit(f"Server at {args.base_url} is not healthy.")
    m0 = cl.metrics()
    max_len = cl.max_model_len()
    auto = m0 and m0["block_size"]
    if args.block_size:
        block = args.block_size
    elif auto and 16 <= auto <= 8192:
        block = auto
        print(f"[{args.label}] WARNING: using block size {block} read from /metrics. The OFF and ON servers can "
              f"report different values; pass --block-size explicitly (from the server log line "
              f"'Setting attention block size to N tokens') so both runs build identical prompts.")
    else:
        block = None
    print(f"[{args.label}] metrics: {'found' if m0 and m0['found'] else 'prefix-cache counters not found'}; "
          f"reported block_size={m0 and m0['block_size']} mamba_block_size={m0 and m0['mamba_block_size']}; "
          f"max_model_len={max_len}; boundary cases use block={block or 'none (stride sweep)'}")
    print(f"[{args.label}] building prompts ...")
    cases = build_cases(tok, pb, block, args.long)
    if max_len:
        limit = max_len - 384
        kept = [c for c in cases if all(len(r["ids"]) <= limit for r in c["requests"])]
        for c in cases:
            if c not in kept:
                print(f"[{args.label}] skipping {c['id']}: prompt exceeds max_model_len {max_len}")
        cases = kept
    total_prompt = sum(len(r["ids"]) for c in cases for r in c["requests"])
    print(f"[{args.label}] {len(cases)} cases, {sum(len(c['requests']) for c in cases)} requests/round, "
          f"{total_prompt:,} prompt tokens/round")

    if total_prompt > 3_000_000:
        sys.exit(f"Refusing to run: {total_prompt:,} prompt tokens per round looks wrong (block size {block}?)")

    out = {"meta": {"label": args.label, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "model": args.model, "block_size": block, "max_model_len": max_len, "rounds": args.rounds,
                    "logprobs": True}, "cases": {}}
    t_start = time.time()
    for rnd in range(1, args.rounds + 1):
        for case in cases:
            res = run_case(cl, case)
            out["cases"].setdefault(case["id"], []).append(res)
            for r in res["requests"]:
                hit = r.get("cache_hit_tokens")
                if hit is None:
                    hit = r.get("group_hit_tokens")
                status = "ERROR " + r["error"][:80] if "error" in r else ("needle OK" if r["needle_ok"] else "needle MISS")
                print(f"[{args.label} r{rnd}] {case['id']:<28} {r['role']:<11} {r['prompt_len']:>7} tok  "
                      f"hit={hit if hit is not None else '?':>7}  {r.get('latency_s', '-'):>7}s  {status}")
            if not cl.healthy():
                print(f"!!! server died during {case['id']} (round {rnd}); check dmesg | grep -i xid")
                out["meta"]["server_died"] = case["id"]
                break
        if out["meta"].get("server_died"):
            break
    out["meta"]["logprobs"] = cl.logprobs
    out["meta"]["elapsed_s"] = round(time.time() - t_start)
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"[{args.label}] saved {args.out}  ({out['meta']['elapsed_s']}s)")
    if args.compare:
        with open(args.compare) as f:
            ref = json.load(f)
        return compare(ref, out)
    return 0


# ---------------------------------------------------------------- compare

def nonfinite(r):
    for v in r.get("token_logprobs") or []:
        if v is None:
            continue
        if not math.isfinite(v) or v <= -9000:
            return True
    return False


def degenerate(r):
    toks = r.get("tokens") or []
    if len(toks) < 24:
        return False
    return Counter(toks).most_common(1)[0][1] / len(toks) >= 0.8


def gap_at(r, d, other_tok):
    """How far other_tok was behind r's chosen token at position d (nats); None if not in top-5."""
    tops = r.get("top_logprobs") or []
    if d >= len(tops) or not tops[d] or other_tok not in tops[d]:
        return None
    return r["token_logprobs"][d] - tops[d][other_tok]


def judge(a, b):
    if a["sha"] != b["sha"]:
        return "INVALID", "prompts differ between runs"
    if "error" in a:
        return "SKIP", "reference request errored"
    if "error" in b:
        return "FAIL", b["error"][:90]
    if nonfinite(b):
        return "FAIL", "NaN/inf logprob"
    if degenerate(b) and not degenerate(a):
        return "FAIL", "degenerate repetition"
    if a["needle_ok"] and not b["needle_ok"]:
        return "FAIL", f"lost recall (expected {a['expect']}): {b.get('text', '')[:50]!r}"

    ta, tb = a.get("tokens"), b.get("tokens")
    if ta is None or tb is None:
        if a["text"] == b["text"]:
            return "EXACT", ""
        n = next((i for i, (x, y) in enumerate(zip(a["text"], b["text"])) if x != y),
                 min(len(a["text"]), len(b["text"])))
        return ("BENIGN" if n >= 0.5 * min(len(a["text"]), len(b["text"])) else "FAIL",
                f"text diverged at char {n} (no logprobs available)")
    d = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), None)
    if d is None:
        if len(ta) == len(tb):
            return "EXACT", ""
        return "BENIGN", f"identical for {min(len(ta), len(tb))} tokens, stopped at different length"
    gaps = [g for g in (gap_at(b, d, ta[d]), gap_at(a, d, tb[d])) if g is not None]
    if gaps and min(gaps) <= NEAR_TIE_NATS:
        return "BENIGN", f"diverged at token {d}, near-tie {min(gaps):.2f} nat"
    if gaps:
        return "FAIL", f"diverged at token {d}, gap {min(gaps):.2f} nat (not a near-tie)"
    return "FAIL", f"diverged at token {d}, reference token not in top-5"


def compare(ref, test):
    print()
    print(f"Reference: {ref['meta']['label']} ({ref['meta']['time']})   "
          f"Test: {test['meta']['label']} ({test['meta']['time']}, {test['meta']['rounds']} rounds)")

    # self-consistency of reference: A primer vs measured are identical prompts, no cache
    ctrl = []
    for cid, rounds in ref["cases"].items():
        if cid.startswith("A_"):
            reqs = rounds[0]["requests"]
            if len(reqs) == 2:
                ctrl.append((cid, *judge(reqs[0], reqs[1])))
    print("\nReference determinism control (same prompt twice, cachingoff):")
    for cid, v, why in ctrl:
        print(f"  {cid:<28} {v:<7} {why}")

    ref_hits = sum((r.get("cache_hit_tokens") or 0) for rr in ref["cases"].values()
                   for rnd in rr for r in rnd["requests"])
    test_hits = sum((r.get("cache_hit_tokens") or 0) for rr in test["cases"].values()
                    for rnd in rr for r in rnd["requests"] if r["role"] == "measured")

    counts, rows = Counter(), []
    for cid, ref_rounds in ref["cases"].items():
        ref_reqs = ref_rounds[0]["requests"]
        for ri, rnd in enumerate(test["cases"].get(cid, []), 1):
            for a, b in zip(ref_reqs, rnd["requests"]):
                v, why = judge(a, b)
                counts[v] += 1
                hit = b.get("cache_hit_tokens")
                if hit is None:
                    hit = b.get("group_hit_tokens")
                rows.append((cid, ri, b["role"], b["prompt_len"], hit, v, why))

    print()
    print(f"{'case':<28} {'rnd':>3} {'role':<11} {'prompt':>7} {'hit':>7}  verdict  detail")
    print("-" * 110)
    for cid, ri, role, plen, hit, v, why in rows:
        print(f"{cid:<28} {ri:>3} {role:<11} {plen:>7} {hit if hit is not None else '?':>7}  {v:<7}  {why}")

    problems = []
    if counts["FAIL"]:
        problems.append(f"{counts['FAIL']} FAIL")
    if counts["INVALID"]:
        problems.append(f"{counts['INVALID']} INVALID (prompt mismatch -- different tokenizer or script version)")
    if ref_hits > 0:
        problems.append(f"reference server reported {ref_hits} cache-hit tokens: it was NOT launched with caching off")
    hits_known = any(r.get("cache_hit_tokens") is not None for rr in test["cases"].values()
                     for rnd in rr for r in rnd["requests"])
    if hits_known and test_hits == 0:
        problems.append("test server reported 0 cache-hit tokens: prefix caching is not actually on")
    if test["meta"].get("server_died"):
        problems.append(f"server died during {test['meta']['server_died']}")
    if ref["meta"].get("block_size") != test["meta"].get("block_size"):
        problems.append(f"block size differs between runs ({ref['meta'].get('block_size')} vs "
                        f"{test['meta'].get('block_size')}): boundary cases are not comparable")
    missing = set(ref["cases"]) - set(test["cases"])
    if missing:
        problems.append(f"{len(missing)} cases missing from test run")

    print("-" * 110)
    print(f"EXACT {counts['EXACT']}   BENIGN {counts['BENIGN']}   FAIL {counts['FAIL']}   "
          f"SKIP {counts['SKIP']}   cache-hit tokens on measured requests: "
          f"{test_hits if hits_known else 'unknown (no /metrics counters)'}")
    if problems:
        print("VERDICT: NOT SAFE -> " + "; ".join(problems))
        return 1
    print("VERDICT: PASS -- cached responses match the no-cache reference (exact or near-tie float noise only)")
    return 0


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("record")
    r.add_argument("--label", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--rounds", type=int, default=1)
    r.add_argument("--compare", help="reference JSON to compare against after recording")
    r.add_argument("--base-url", default="http://localhost:8080")
    r.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", ""))
    r.add_argument("--model", default="qwen38-27b-int8")
    r.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    r.add_argument("--max-tokens", type=int, default=96)
    r.add_argument("--block-size", type=int, default=0, help="override cache block size for boundary cases")
    r.add_argument("--long", action="store_true", help="add a 128K exact-repeat case")

    c = sub.add_parser("compare")
    c.add_argument("reference")
    c.add_argument("test")

    args = ap.parse_args()
    if args.cmd == "record":
        if not args.api_key:
            sys.exit("Set VLLM_API_KEY or pass --api-key")
        sys.exit(record(args))
    with open(args.reference) as f:
        ref = json.load(f)
    with open(args.test) as f:
        test = json.load(f)
    sys.exit(compare(ref, test))


if __name__ == "__main__":
    main()
