"""RQ2: TopDown against the same-model MCTS-Judge re-implementation (mcts_judge_run.py), for one judge, on HumanEvalPack (hep + hepx pooled over its six
languages) and MBPP+. TopDown = vanilla + evidence of the feedback-round checks (evidence.py), as in score_paper.py. MCTS-Judge is read from
topdown_<MODEL>_mctsjudge_<bench>_*; its output is its binary verdict (+1 correct / -1 incorrect).
Metrics:
  pairwise accuracy  the fraction of correct-buggy pairs of one problem (984 in HumanEvalPack, 209 in MBPP+) in which the judge scores the correct program
                     higher; ties count half, so a binary verdict scores its balanced accuracy.
  MCC                Matthews correlation with the execution label at threshold 0.
95% bootstrap intervals of the TopDown-minus-MCTS-Judge difference and two-sided p-values from 2,000 resamples by problem (HumanEvalPack: the problem across its
six languages; MBPP+: the task), seed 0, Holm correction over the four comparisons (2 benchmarks x 2 metrics) of this judge. Cost per program: generated
tokens and LLM calls, counted the same way for both methods (MCTS-Judge's test generation included).
Usage: python score_mcts.py MODEL   (qwen7b | llama8b); TD_DIR = frozen runs (default ../runs), MCTS_DIR = MCTS-Judge outputs (default TD_DIR)"""
import json, math, sys, os, hashlib, numpy as np
from sklearn.metrics import matthews_corrcoef
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evidence import evidence
MODEL = sys.argv[1]; HERE = os.path.dirname(os.path.abspath(__file__))
D = os.environ.get("TD_DIR", os.path.join(HERE, "..", "runs")); DM = os.environ.get("MCTS_DIR", D); MTAG = f"{MODEL}_mctsjudge"; NB = 2000
def lines(p):
    """Read a transcripts file stored plain (.jsonl) or compressed (.jsonl.gz)."""
    import gzip
    return gzip.open(p + ".gz", "rt") if not os.path.exists(p) and os.path.exists(p + ".gz") else open(p)
def load(tag, b, d=D):
    f = f"{d}/topdown_{tag}_{b}_verdicts.json"
    if not os.path.exists(f): return None
    raw = open(f, "rb").read(); meta = json.load(open(f.replace("_verdicts", "_meta"))); assert hashlib.sha256(raw).hexdigest() == meta["verdicts_sha256"], f
    T = {json.loads(l)["id"]: json.loads(l) for l in lines(f.replace("_verdicts.json", "_transcripts.jsonl"))}
    return dict(V=json.loads(raw), L=json.load(open(f.replace("_verdicts", "_labels"))), T=T, meta=meta)
mcc = lambda y, s: matthews_corrcoef(y, (s > 0).astype(int))
GROUPS = {"HumanEvalPack (6 languages)": ["hep", "hepx"], "MBPP+": ["mbppplus"]}
tests = []
for G, benches in GROUPS.items():
    ids, y, S, tok, calls, ntot, extra = [], [], {}, {}, {}, 0, []
    for b in benches:
        first, fb, mj = load(f"{MODEL}_firstpass", b), load(f"{MODEL}_feedback", b), load(MTAG, b, DM)
        if not (first and fb and mj): print(f"[{G}] {b}: missing runs ({', '.join(n for n, x in (('firstpass', first), ('feedback', fb), (MTAG, mj)) if not x)}), skipped"); continue
        if mj["meta"].get("LIMIT"): extra.append(f"{b}: MCTS-Judge run has LIMIT={mj['meta']['LIMIT']} (subset)")
        bid = sorted(i for i in first["L"] if i in fb["T"] and i in mj["V"])
        assert all(mj["L"][i] == first["L"][i] for i in bid), f"{b}: labels differ between the MCTS-Judge and TopDown runs"
        n = len(bid); ntot += n
        cols = {"TopDown": [first["V"][i]["vanilla"] + evidence(fb["T"][i]["checks"]) for i in bid], "MCTS-Judge": [mj["V"][i]["mcts_bin"] for i in bid]}
        for k, v in cols.items(): S.setdefault(k, []).extend(v)
        ids += bid; y += [first["L"][i] for i in bid]
        td = {k: first["meta"]["per_candidate"]["td"][k] + fb["meta"]["per_candidate"]["td"][k] for k in ("gen_tokens", "calls")}
        mc = mj["meta"]["per_candidate"]["mcts"]
        for k, src in (("TopDown", td), ("MCTS-Judge", mc)):
            tok[k] = tok.get(k, 0) + src["gen_tokens"] * n; calls[k] = calls.get(k, 0) + src["calls"] * n
        st = mj["meta"].get("stats", {})
        if st: extra.append(f"{b}: MCTS-Judge stored {st.get('tests_stored', 0) / mj['meta']['n_tasks']:.2f} tests/task, {st.get('tasks_without_tests', 0)} tasks without tests; "
                            f"{mc.get('capped_outputs', 0) / max(mc.get('gen_outputs', 1), 1):.1%} of outputs cut at 2048")
    if not ids: continue
    y = np.array(y); S = {k: np.array(v, float) for k, v in S.items()}
    tk = [i.split("|")[0] for i in ids]; grp = np.array([t.split("/", 1)[1] if G.startswith("Human") else t for t in tk]); ug = np.unique(grp)
    pos = {}
    for k, (t, lab) in enumerate(zip(tk, y)): pos.setdefault(t, {})[int(lab)] = k
    assert all(len(p) == 2 for p in pos.values()), "every problem needs one correct and one buggy program"
    pair = {t: (p[1], p[0]) for t, p in pos.items()}
    unit_prog = {g: np.where(grp == g)[0] for g in ug}
    unit_pair = {g: np.array([pair[t] for t in sorted(pair) if (t.split("/", 1)[1] if G.startswith("Human") else t) == g]) for g in ug}
    allpairs = np.array([pair[t] for t in sorted(pair)])
    def pairwise(s, pr): a, c = s[pr[:, 0]], s[pr[:, 1]]; return float(np.mean((a > c) + 0.5 * (a == c)))
    rng = np.random.default_rng(0); boots = []
    for _ in range(NB):
        pick = rng.choice(ug, len(ug)); ix = np.concatenate([unit_prog[g] for g in pick])
        if len(set(y[ix])) > 1: boots.append((ix, np.concatenate([unit_pair[g] for g in pick])))
    print(f"\n##### {MODEL} | {G}: {len(ids)} programs ({int(y.sum())} correct, {len(y) - int(y.sum())} buggy), {len(allpairs)} correct-buggy pairs, {len(ug)} resampling units")
    print("generated tokens / program: " + " | ".join(f"{k} {v / ntot:.0f}" for k, v in tok.items()))
    print("LLM calls / program:        " + " | ".join(f"{k} {v / ntot:.1f}" for k, v in calls.items()))
    print(f"MCTS-Judge / TopDown:       {tok['MCTS-Judge'] / tok['TopDown']:.1f}x tokens, {calls['MCTS-Judge'] / calls['TopDown']:.1f}x calls")
    for e in extra: print("   " + e)
    print(f"{'judge':28s} {'pairwise':>9s} {'MCC':>7s}")
    for k, s in S.items(): print(f"{k:28s} {pairwise(s, allpairs):9.3f} {mcc(y, s):7.3f}")
    for mn, f in (("pairwise", lambda s, ix, pr: pairwise(s, pr)), ("MCC", lambda s, ix, pr: mcc(y[ix], s[ix]))):
        d = np.array([f(S["TopDown"], ix, pr) - f(S["MCTS-Judge"], ix, pr) for ix, pr in boots]); lo, hi = np.percentile(d, [2.5, 97.5])
        p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()) + 1 / len(d))
        va, vr = f(S["TopDown"], np.arange(len(y)), allpairs), f(S["MCTS-Judge"], np.arange(len(y)), allpairs)
        tests.append([G, mn, "MCTS-Judge", va - vr, 100 * (va - vr) / abs(vr) if vr else float("nan"), lo, hi, p])
if tests:
    print(f"\n##### TopDown vs MCTS-Judge ({MODEL}): difference, relative gain, 95% CI, bootstrap p, Holm-adjusted p over {len(tests)} comparisons")
    order = np.argsort([t[-1] for t in tests]); m = len(tests); adj, run = [0.0] * m, 0.0
    for rank, j in enumerate(order): run = max(run, min(1.0, (m - rank) * tests[j][-1])); adj[j] = run
    for t, a in zip(tests, adj):
        G, mn, r, dd, rel, lo, hi, p = t
        print(f"{G[:14]:14s} {mn:8s} vs {r:12s} {dd:+.3f} ({rel:+6.1f}%) [{lo:+.3f}, {hi:+.3f}]  p={p:.4f}  Holm p={a:.4f} {'*' if a < 0.05 else 'ns'}")
