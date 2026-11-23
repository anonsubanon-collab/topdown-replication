"""RQ1 numbers for one judge: HumanEvalPack pooled over all 6 languages (hep + hepx) and MBPP+.
Judges: vanilla, self-consistency, multi-agent (x5 where a x5 run exists, else x4), TopDown (= first pass + feedback round).
Reports Kendall tau-b and MCC (score > 0), 95% bootstrap intervals resampled by problem (HumanEvalPack: the problem across its 6 languages; MBPP+: the task),
two-sided bootstrap p-values with Holm correction over every TopDown-vs-baseline comparison of this judge, generated tokens per program and the fraction of
outputs cut at max_tokens.
Usage: python score_paper.py MODEL   (qwen7b | llama8b)"""
import json, math, sys, os, hashlib, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evidence import evidence
from scipy.stats import kendalltau
from sklearn.metrics import matthews_corrcoef
MODEL = sys.argv[1]
D = os.environ.get("TD_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs")); NB = 2000
def lines(p):
    """Read a transcripts file stored plain (.jsonl) or compressed (.jsonl.gz)."""
    import gzip
    return gzip.open(p + ".gz", "rt") if not os.path.exists(p) and os.path.exists(p + ".gz") else open(p)
def load(tag, b):
    f = f"{D}/topdown_{tag}_{b}_verdicts.json"
    if not os.path.exists(f): return None
    raw = open(f, "rb").read(); meta = json.load(open(f.replace("_verdicts", "_meta"))); assert hashlib.sha256(raw).hexdigest() == meta["verdicts_sha256"], f
    T = {json.loads(l)["id"]: json.loads(l) for l in lines(f.replace("_verdicts.json", "_transcripts.jsonl"))}
    return dict(V=json.loads(raw), L=json.load(open(f.replace("_verdicts", "_labels"))), T=T, meta=meta)
tau = lambda y, s: kendalltau(y, s).statistic
mcc = lambda y, s: matthews_corrcoef(y, (s > 0).astype(int))
MET = [("tau", tau), ("MCC", mcc)]
GROUPS = {"HumanEvalPack (6 languages)": ["hep", "hepx"], "MBPP+": ["mbppplus"]}
rows, tests, frames = {}, [], {}
for G, benches in GROUPS.items():
    ids, y, S, tok, cut, unp = [], [], {}, {}, {}, {}
    ntot = 0
    for b in benches:
        first, fb = load(f"{MODEL}_firstpass", b), load(f"{MODEL}_feedback", b)
        base = load(f"{MODEL}_baselines", b); ma5 = load(f"{MODEL}_ma5", b)
        sc50 = load(f"{MODEL}_sc50", b)
        if not (first and fb and base): print(f"[{G}] {b}: missing runs, skipped"); continue
        bid = sorted(i for i in first["L"] if i in base["V"] and i in fb["T"] and (not ma5 or i in ma5["V"]) and (not sc50 or i in sc50["V"])); n = len(bid); ntot += n
        nsc, krep = base["meta"]["variant"].get("N_SC", 0), base["meta"]["variant"].get("MA_REPEAT", 1)
        mak = ("x5", ma5) if ma5 else (f"x{krep}", base); scs = sc50 or base
        if sc50: nsc = sc50["meta"]["variant"].get("N_SC", 50)
        cols = {"vanilla": [first["V"][i]["vanilla"] for i in bid],
                f"self-consistency x{nsc}": [scs["V"][i]["vsc"] for i in bid],
                f"multi-agent {mak[0]}": [mak[1]["V"][i]["ma_arbiter"] for i in bid],
                "TopDown": [first["V"][i]["vanilla"] + evidence(fb["T"][i]["checks"]) for i in bid]}
        for k, v in cols.items(): S.setdefault(k, []).extend(v)
        ids += bid; y += [first["L"][i] for i in bid]
        pf, pfb, pb, pm = first["meta"]["per_candidate"], fb["meta"]["per_candidate"], base["meta"]["per_candidate"], mak[1]["meta"]["per_candidate"]
        for k, t in (((f"self-consistency x{nsc}", scs["meta"]["per_candidate"]["vsc"]), (f"multi-agent {mak[0]}", pm["ma"])) + (("TopDown", None),)):
            g = pf["td"]["gen_tokens"] + pfb["td"]["gen_tokens"] if t is None else t["gen_tokens"]
            tok[k] = tok.get(k, 0) + g * n
            if t is not None and t.get("gen_outputs"): c = cut.setdefault(k, [0, 0]); c[0] += t["capped_outputs"]; c[1] += t["gen_outputs"]
        if pfb["td"].get("gen_outputs"): c = cut.setdefault("TopDown feedback calls", [0, 0]); c[0] += pfb["td"]["capped_outputs"]; c[1] += pfb["td"]["gen_outputs"]
        if pf["td"].get("gen_outputs"): c = cut.setdefault("TopDown first pass", [0, 0]); c[0] += pf["td"]["capped_outputs"]; c[1] += pf["td"]["gen_outputs"]
        u = unp.setdefault("first-pass code-only reading missing", [0, 0])
        for i in bid:
            for c in first["T"][i]["checks"]: u[0] += c["code_only"] is None; u[1] += 1
        r = unp.setdefault("feedback re-trace missing", [0, 0])
        for i in bid:
            for c in fb["T"][i]["checks"]:
                if "retrace" in c: r[0] += c["retrace"] is None; r[1] += 1
    if not ids: continue
    y = np.array(y); S = {k: np.array(v) for k, v in S.items()}
    grp = np.array([i.split("|")[0].split("/", 1)[1] if G.startswith("Human") else i.split("|")[0] for i in ids]); ug = np.unique(grp)
    by = {g: np.where(grp == g)[0] for g in ug}; rng = np.random.default_rng(0)
    boots = [ix for ix in (np.concatenate([by[g] for g in rng.choice(ug, len(ug))]) for _ in range(NB)) if len(set(y[ix])) > 1]
    frames[G] = (y, S, boots)
    print(f"\n##### {MODEL} | {G}: {len(ids)} programs ({int(y.sum())} correct, {len(y) - int(y.sum())} buggy), {len(ug)} resampling units")
    print("generated tokens / program: " + " | ".join(f"{k} {v / ntot:.0f}" for k, v in tok.items()))
    print("outputs cut at max_tokens:  " + " | ".join(f"{k} {a / b_:.1%}" for k, (a, b_) in cut.items()))
    print("TopDown readings missing:    " + " | ".join(f"{k} {a / max(1, b_):.1%} ({a}/{b_})" for k, (a, b_) in unp.items()))
    print(f"{'judge':28s} " + " ".join(f"{m:>7s}" for m, _ in MET))
    for k, s in S.items(): print(f"{k:28s} " + " ".join(f"{f(y, s):7.3f}" for _, f in MET))
    for mn, f in MET:
        for r in [k for k in S if k != "TopDown"]:
            d = np.array([f(y[ix], S["TopDown"][ix]) - f(y[ix], S[r][ix]) for ix in boots]); lo, hi = np.percentile(d, [2.5, 97.5])
            p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()) + 1 / len(d))
            va, vr = f(y, S["TopDown"]), f(y, S[r]); tests.append([G, mn, r, va - vr, 100 * (va - vr) / abs(vr), lo, hi, p])
print(f"\n##### TopDown vs baselines ({MODEL}): difference, relative gain, 95% CI, bootstrap p, Holm-adjusted p over {len(tests)} comparisons")
order = np.argsort([t[-1] for t in tests]); m = len(tests); adj, run = [0.0] * m, 0.0
for rank, j in enumerate(order): run = max(run, min(1.0, (m - rank) * tests[j][-1])); adj[j] = run
for t, a in zip(tests, adj):
    G, mn, r, dd, rel, lo, hi, p = t
    print(f"{G[:14]:14s} {mn:4s} vs {r:24s} {dd:+.3f} ({rel:+5.1f}%) [{lo:+.3f}, {hi:+.3f}]  p={p:.4f}  Holm p={a:.4f} {'*' if a < 0.05 else 'ns'}")
