
import json, os, re, sys, hashlib, math
from collections import Counter

from datasets import load_dataset
from vllm import LLM, SamplingParams
MODEL, TAG, OUTD = sys.argv[1:4]; os.makedirs(OUTD, exist_ok=True)
K = int(os.environ.get("K", "4")); Q = int(os.environ.get("Q", "6")); NREV = int(os.environ.get("N_REV", "3")); LIMIT = int(os.environ.get("LIMIT", "0"))
RUN_MA = int(os.environ.get("MA", "1")); TD_SKIP = int(os.environ.get("TD_SKIP", "0")); MA_REPEAT = int(os.environ.get("MA_REPEAT", "1")); N_SC = int(os.environ.get("N_SC", "0"))
R_AGREE, R_SINGLE, Q_EXPOSE = 0.9, 0.6, 0.5
PREC_HI = 0.75
VARIANT = dict(MA=RUN_MA, TD_SKIP=TD_SKIP, MA_REPEAT=MA_REPEAT, N_SC=N_SC)
PAIRS = os.environ.get("PAIRS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "mbppplus_pairs.json"))
BENCHES = os.environ.get("BENCHES", "hep,hepx,mbppplus").split(",")
todo = [b for b in BENCHES if not os.path.exists(f"{OUTD}/topdown_{TAG}_{b}_verdicts.json")]
if not todo: sys.exit("nothing to do: all outputs exist")

def load(bench):
    if bench == "hepx":
        items = []
        for lang in ("cpp", "go", "rust"):
            for r in load_dataset("bigcode/humanevalpack", lang, split="test"):
                for label, sol in ((1, r["canonical_solution"]), (0, r["buggy_solution"])):
                    items.append(dict(id=f'{r["task_id"]}|{label}', lang=lang, label=label, spec=r["prompt"], code=r["prompt"] + sol))
        assert len(items) == 984, f"expected 984 HumanEvalPack C++/Go/Rust items, got {len(items)}"; return items
    if bench == "mbppplus":
        items = [dict({"lang": "python"}, **it) for it in json.load(open(PAIRS))]
        assert len(items) == 418, f"expected 418 MBPP+ items, got {len(items)}"; return items
    items = []
    for lang in ("python", "js", "java"):
        for r in load_dataset("bigcode/humanevalpack", lang, split="test"):
            for label, sol in ((1, r["canonical_solution"]), (0, r["buggy_solution"])):
                items.append(dict(id=f'{r["task_id"]}|{label}', lang=lang, label=label, spec=r["prompt"], code=r["prompt"] + sol))
    assert len(items) == 984, f"expected 984 HumanEvalPack items, got {len(items)}"; return items

llm = LLM(MODEL, dtype=os.environ.get("DTYPE", "bfloat16"), tensor_parallel_size=int(os.environ.get("TP", "1")), max_model_len=int(os.environ.get("MAXLEN", "8192")),
          gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.9")), seed=0, enforce_eager=bool(int(os.environ.get("EAGER", "0"))))
tok = llm.get_tokenizer()
SYS = "You are an expert software engineer reviewing code."
def P(msgs): return tok.apply_chat_template([{"role": "system", "content": SYS}] + msgs, tokenize=False, add_generation_prompt=True)
ASK = "Is the code correct? Answer with exactly one word: YES (fully correct) or NO (has a bug)."
spec_blk = lambda it: f"Specification ({it['lang']}):\n```\n{it['spec']}\n```"
code_blk = lambda it: f"Candidate implementation:\n```\n{it['code']}\n```"
NOTSEEN = "You have NOT seen any implementation yet. "

COST = {}
def account(method, outs):
    c = COST.setdefault(method, Counter()); c["calls"] += sum(len(o.outputs) for o in outs)
    c["prompt_tokens"] += sum(len(o.prompt_token_ids) * len(o.outputs) for o in outs); c["gen_tokens"] += sum(len(s.token_ids) for o in outs for s in o.outputs)

def gen(texts, method, mx, temp=0.0, n=1, seed=0):
    if not texts: return []
    outs = llm.generate(texts, SamplingParams(temperature=temp, top_p=0.95 if temp else 1.0, n=n, max_tokens=mx, seed=seed), use_tqdm=False)
    account(method, outs)
    c = COST.setdefault(method, Counter()); c["gen_outputs"] += sum(len(o.outputs) for o in outs)
    c["capped_outputs"] += sum(1 for o in outs for s_ in o.outputs if len(s_.token_ids) >= mx)
    return [[s_.text.strip() for s_ in o.outputs] for o in outs]

def first_token_logprobs(texts, method):
    if not texts: return []
    outs = llm.generate(texts, SamplingParams(temperature=0, max_tokens=1, logprobs=20), use_tqdm=False); account(method, outs)
    return [[(lp.decoded_token.strip().upper(), lp.logprob) for lp in o.outputs[0].logprobs[0].values()] for o in outs]

def verdict(convs, method):
    dv = []
    for top in first_token_logprobs([P(c + [{"role": "user", "content": ASK}]) for c in convs], method):
        y = n = -30.0
        for t, lp in top:
            if t.startswith("YES"): y = max(y, lp)
            if t.startswith("NO"): n = max(n, lp)
        dv.append(y - n)
    return dv

def norm(v):
    v = v.strip().strip("`*").strip().rstrip(".;").strip()
    v = re.sub(r"^(?:returns?|output|result|value|=>|->|=)\s*:?\s*", "", v, flags=re.I)
    v = re.sub(r"(?:List\.of|Arrays\.asList|new\s*\w+\s*\[\s*\])", "", v).replace('"', "'").replace(" ", "").replace("{", "[").replace("}", "]")
    if v.startswith("(") and v.endswith(")"): v = "[" + v[1:-1] + "]"
    v = {"true": "true", "false": "false", "none": "null", "null": "null", "nil": "null", "undefined": "null"}.get(v.lower(), v)
    v = v.replace("'", "")
    if re.match(r"(?i)^(error|panic|.*exception\b|.*error\b|<nil>$)", v): v = "ERROR" if not v.lower() == "<nil>" else "null"
    try: return repr(round(float(v), 6))
    except ValueError: return v

def numbered(text, n):
    d = {}
    for m in re.finditer(r"^\s*(?:[-*]\s*)?\(?(\d+)\)?\s*[:.)]\s*(.+?)\s*$", text, re.M):
        i = int(m.group(1))
        if 1 <= i <= n and i not in d: d[i] = m.group(2)
    return d

JSONFMT = ("Write every value as a JSON literal: numbers, true/false, null, \"strings\", [lists] (also for arrays, vectors, slices and tuples), "
           "{\"key\": value} for maps; write ERROR if the call raises an exception, panics or does not terminate.")
INTENT = (f"\n\n{NOTSEEN}From the specification alone, list at most {Q} valid test inputs (inputs the specification allows) that best "
          "separate correct from buggy implementations: every docstring example first, then edge cases. One per line as 'PROBE: <call expression "
          "in the specification's language>', using the function name from the specification. Output nothing else.")
PRED = (f"\n\n{NOTSEEN}For each call below, state the exact value a correct implementation returns; it must have the return type declared "
        f"in the function signature. {JSONFMT} One per line as 'RESULT <number>: <value>'. Output nothing else.\n")
CODE_TRACE = ("\n\nWithout running it, trace the code step by step on the call `{probe}`, carefully; summarize repeated loop iterations instead of listing "
              "each one. End with a line 'RESULT: <value the code returns>'. " + JSONFMT.replace("{", "{{").replace("}", "}}"))
MA_TRACE = (f"\n\nWithout running it, choose at most {Q} valid inputs (every docstring example first, then edge cases) and trace the candidate step by step "
            "on each, keeping each trace brief and summarizing repeated loop iterations instead of listing each one. Directly after each input's trace "
            "write one line 'RESULT <number>: <call> | code returns: <value> | specification requires: <value>'.")
GROUNDED = ("Only state rules the specification states or that follow necessarily from it; do not invent error handling or input validation it does not ask for. ")
MA_INSPECT = ("\n\nWithout running it, list at most 6 boundary rules implied by the specification and check the candidate against each. " + GROUNDED + "One line per rule: "
              "'<number>: <rule> - OK' or '<number>: <rule> - VIOLATED: <reason>'. Output nothing else.")
MA_REVIEW = "\n\nWithout running it, review the candidate for bugs against the specification. List each concrete bug, or state that none is found. Be concise."
SC_REVIEW = "\n\nWithout running it, review the candidate step by step against the specification and decide whether it is correct. Be concise."
MA_CAPS = (4096, 768, 768, 1024)
FINDINGS = "\n\nFindings from reviewers who analysed the candidate without running it:\n"

def value_of(t):
    if t is None: return None
    spans = [x for x in re.findall(r"`+([^`]+?)`+", t) if x.strip()]
    return (spans[-1] if spans else t).strip().strip("*").strip()
def results_v2(text, n):
    d = {}
    for m in re.finditer(r"RESULT\**[ \t]*(\d+)[ \t]*\**[ \t]*:[ \t]*\**(.+?)[ \t]*$", text, re.M | re.I):
        i = int(m.group(1))
        if 1 <= i <= n: d[i] = value_of(m.group(2))
    return d
def parse_preds(text, n):
    vals = []
    for l in text.splitlines():
        l = re.sub(r"^\s*(?:[-*]\s*)?(?:\d+\s*(?::|\)|\.(?=\s)))?\s*(?:\**RESULT\**(?:\b[^:\n]{0,12}:|\s+))?\s*", "", l).strip()
        if l: vals.append(value_of(l))
    if len(vals) == n: return {i: v for i, v in enumerate(vals, 1)}
    d = results_v2(text, n)
    return d if d else {i: value_of(re.sub(r"^\**RESULT\**\s*\d*\s*\**\s*:\s*", "", v)) for i, v in numbered(text, n).items()}
def strip_doc(c, lang):
    c = re.sub(r'("""|\'\'\')[\s\S]*?\1', "", c) if lang == "python" else re.sub(r"/\*[\s\S]*?\*/", "", c)
    c = re.sub(r"(?m)^\s*#.*$", "", c) if lang == "python" else re.sub(r"(?m)^\s*//.*$", "", c)
    return "\n".join(l for l in c.splitlines() if l.strip())
def violated(text): return [m.group(0).strip() for m in re.finditer(r"^.*\bVIOLATED\b.*$", text, re.M)]

for bench in todo:
    items = load(bench); tasks = {}
    for it in items: tasks.setdefault(it["id"].split("|")[0], it)
    tids = sorted(tasks)
    if LIMIT: tids = tids[:LIMIT]; items = [it for it in items if it["id"].split("|")[0] in set(tids)]
    print(f"{MODEL} {bench}: {len(items)} candidates, {len(tids)} tasks, K={K} Q={Q} N_REV={NREV}", flush=True)
    COST.clear(); tid = lambda it: it["id"].split("|")[0]; R = {it["id"]: {} for it in items}; T = {it["id"]: {} for it in items}

    for it, d in zip(items, verdict([[{"role": "user", "content": spec_blk(it) + "\n\n" + code_blk(it)}] for it in items], "vanilla")): R[it["id"]]["vanilla"] = d

    if TD_SKIP: td_stats = {}
    else:
        intent = dict(zip(tids, (o[0] for o in gen([P([{"role": "user", "content": spec_blk(tasks[t]) + INTENT}]) for t in tids], "td", 512))))
        tagged = lambda tag, txt: [x.strip().strip("`").strip() for x in re.findall(rf"^\s*(?:[-*]|\d+[.)])?\s*\**{tag}\**\s*:\s*\**(.+?)\s*$", txt, re.M | re.I)]
        probes = {t: [p for p in tagged("PROBE", intent[t]) if p][:Q] for t in tids}
        ptids = [t for t in tids if probes[t]]
        preds = dict(zip(ptids, gen([P([{"role": "user", "content": spec_blk(tasks[t]) + PRED + "\n".join(f"{i}: {p}" for i, p in enumerate(probes[t], 1))}]) for t in ptids], "td", 384, 0.7, K)))
        prior = {}
        for t in tids:
            prior[t] = {}
            if t not in preds: continue
            parsed = [parse_preds(s, len(probes[t])) for s in preds[t]]
            for i in range(1, len(probes[t]) + 1):
                vals = [(norm(p[i]), p[i]) for p in parsed if i in p]
                if not vals: continue
                (nv, cnt), = Counter(v for v, _ in vals).most_common(1)
                prior[t][i] = (next(r for v, r in vals if v == nv), nv, cnt / K)
        print(f"  TD predictions: {sum(map(len, probes.values())) / len(tids):.1f} inputs per task; "
              f"{sum(1 for t in tids for v in prior[t].values() if v[2] >= PREC_HI) / max(1, sum(map(len, prior.values()))):.0%} of expectations at p >= {PREC_HI}", flush=True)

        byid = {it["id"]: it for it in items}
        tr_items = [it for it in items if probes[tid(it)]]
        keys = [(it["id"], i) for it in tr_items for i in prior[tid(it)]]
        views = [(f"Code ({byid[k]['lang']}):\n```\n{strip_doc(byid[k]['code'], byid[k]['lang'])}\n```", probes[tid(byid[k])][i - 1]) for k, i in keys]
        ta = gen([P([{"role": "user", "content": v + CODE_TRACE.format(probe=pb)}]) for v, pb in views], "td", 768)
        ta = {key: (re.findall(r"RESULT\**\s*:\s*\**(.+?)\s*$", o[0], re.M | re.I) or [None])[-1] for key, o in zip(keys, ta)}
        ta = {k: value_of(v) for k, v in ta.items()}
        tbs = gen([P([{"role": "user", "content": spec_blk(byid[k]) + "\n\n" + code_blk(byid[k]) + CODE_TRACE.format(probe=probes[tid(byid[k])][i - 1])}])
                   for k, i in keys], "td", 768)
        tbs = {key: value_of((re.findall(r"RESULT\**\s*:\s*\**(.+?)\s*$", o[0], re.M | re.I) or [None])[-1]) for key, o in zip(keys, tbs)}
        stats = Counter()
        for it in items:
            t = tid(it); recs = []
            for i, (raw, nv, p) in prior[t].items():
                a, bb = ta.get((it["id"], i)), tbs.get((it["id"], i))
                if a is None and bb is None: continue
                r = R_AGREE if (a is not None and bb is not None and norm(a) == norm(bb)) else R_SINGLE
                d = norm(a if a is not None else bb) != nv
                stats["checks"] += 1; stats["discrepancies"] += d; stats["sensors_agree"] += r == R_AGREE
                recs.append(dict(call=probes[t][i - 1], prior=raw, precision=p, code_only=a, with_spec=bb, r=r, disc=d))
            T[it["id"]].update(intent=intent[t], checks=recs, preds=preds.get(t, []), probes=probes[t])
        td_stats = dict(stats)
    print(f"  TD done: {td_stats}", flush=True)

    if RUN_MA:
      base = lambda it: spec_blk(it) + "\n\n" + code_blk(it); ma_scores = {it["id"]: [] for it in items}
      for rep in range(MA_REPEAT):
        tt = 0.0 if rep == 0 else 0.7
        ma_tr = [o[0] for o in gen([P([{"role": "user", "content": base(it) + MA_TRACE}]) for it in items], "ma", MA_CAPS[0], tt, 1, rep)]
        ma_in = [o[0] for o in gen([P([{"role": "user", "content": base(it) + MA_INSPECT}]) for it in items], "ma", MA_CAPS[1], tt, 1, rep)]
        ma_rv = gen([P([{"role": "user", "content": base(it) + MA_REVIEW}]) for it in items], "ma", MA_CAPS[2], 0.7, NREV, rep) if NREV else [[] for _ in items]
        ma_conv = []
        for it, tr, ins, rv in zip(items, ma_tr, ma_in, ma_rv):
            res = "\n".join(re.findall(r"^.*RESULT\**\s*\d+\s*\**\s*:.*$", tr, re.M | re.I))
            notes = f"Tracer:\n{res.strip() or tr[-1500:]}\n\nBoundary check:\n{ins}" + "".join(f"\n\nReviewer {j}:\n{r}" for j, r in enumerate(rv, 1))
            if rep == 0: T[it["id"]].update(ma_trace=tr, ma_inspect=ins, ma_reviews=rv)
            ma_conv.append([{"role": "user", "content": base(it) + FINDINGS + notes}])
        for it, d in zip(items, verdict(ma_conv, "ma")): ma_scores[it["id"]].append(d)
      for k, v in ma_scores.items(): R[k]["ma_arbiter"] = sum(v) / len(v)

    if N_SC:
        revs = gen([P([{"role": "user", "content": spec_blk(it) + "\n\n" + code_blk(it) + SC_REVIEW}]) for it in items], "vsc", MA_CAPS[3], 0.7, N_SC)
        convs = [[{"role": "user", "content": spec_blk(it) + "\n\n" + code_blk(it) + SC_REVIEW}, {"role": "assistant", "content": r}] for it, rs in zip(items, revs) for r in rs]
        ds = verdict(convs, "vsc")
        for n_, it in enumerate(items): R[it["id"]]["vsc"] = sum(ds[n_ * N_SC:(n_ + 1) * N_SC]) / N_SC

    vf = f"{OUTD}/topdown_{TAG}_{bench}_verdicts.json"; raw = json.dumps(R, sort_keys=True).encode()
    n = len(items); per = {m: {k: round(v / n, 1) for k, v in c.items()} for m, c in COST.items()}
    json.dump(dict(model=MODEL, variant=VARIANT, K=K, Q=Q, N_REV=NREV, LIMIT=LIMIT, n_items=n, n_tasks=len(tids), per_candidate=per, td_stats=td_stats,
                   verdicts_sha256=hashlib.sha256(raw).hexdigest()), open(f"{OUTD}/topdown_{TAG}_{bench}_meta.json", "w"), indent=1)
    open(vf, "wb").write(raw)
    with open(f"{OUTD}/topdown_{TAG}_{bench}_transcripts.jsonl", "w") as fh:
        for it in items: fh.write(json.dumps(dict(id=it["id"], **T[it["id"]])) + "\n")
    json.dump({it["id"]: it["label"] for it in items}, open(f"{OUTD}/topdown_{TAG}_{bench}_labels.json", "w"))
    print(f"{MODEL} {bench}: saved; per-candidate cost {json.dumps(per)}", flush=True)
