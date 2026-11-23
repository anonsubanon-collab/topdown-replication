"""Number tables behind the two result figures (plain text). Reads results/paper_<model>.txt (RQ1) and, if present, results/rq2_<model>.txt (RQ2)."""
import re, os
RES = os.environ.get("RES_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
M = {"qwen7b": "Qwen-7B", "llama8b": "Llama-8B"}
print("TABLE A (RQ1): scores of every judge. tau = Kendall, MCC at threshold 0. Tokens = generated tokens per program.")
print(f"{'Model':9s} {'Benchmark':14s} {'Judge':22s} {'tau':>6s} {'MCC':>6s} {'tokens':>7s}")
for m, mn in M.items():
    t = open(f"{RES}/paper_{m}.txt").read()
    for blk in t.split("##### ")[1:]:
        h = blk.split("\n")[0]
        if "TopDown vs" in h: continue
        bench = "HumanEvalPack" if "HumanEvalPack" in h else "MBPP+"
        tok = {}
        tl = [l for l in blk.split("\n") if l.startswith("generated tokens")]
        if tl:
            for part in tl[0].split(":", 1)[1].split("|"):
                k, v = part.strip().rsplit(" ", 1); tok[k] = v
        for l in blk.split("\n"):
            mm = re.match(r"^(vanilla|self-consistency x\d+|multi-agent x\d+|TopDown)\s+([-\d.]+)\s+([-\d.]+)\s*$", l)
            if mm:
                j = mm.group(1); print(f"{mn:9s} {bench:14s} {j:22s} " + " ".join(f"{float(x):6.3f}" for x in mm.groups()[1:]) + f" {tok.get(j, '1'):>7s}")
print("\nTABLE B (RQ1): TopDown minus each baseline: difference, relative gain, 95% bootstrap CI (2,000 resamples by problem),")
print("Holm-adjusted p over the 12 comparisons of each model (tau and MCC). * = significant (p < 0.05).")
print(f"{'Model':9s} {'Benchmark':14s} {'Metric':6s} {'vs':22s} {'diff':>7s} {'gain':>8s} {'95% CI':>18s} {'Holm p':>7s}")
for m, mn in M.items():
    for l in open(f"{RES}/paper_{m}.txt"):
        mm = re.match(r"^(HumanEvalPack|MBPP\+)\s+(tau|MCC)\s+vs (.+?)\s+([+-][\d.]+) \(\s*([+-][\d.]+%)\) \[([^\]]+)\].*Holm p=([\d.]+) (\*|ns)", l)
        if mm:
            b, me, vs, d, g, ci, p, s = mm.groups(); print(f"{mn:9s} {b:14s} {me:6s} {vs:22s} {d:>7s} {g:>8s} {'[' + ci + ']':>18s} {p:>7s} {s}")
files = [m for m in M if os.path.exists(f"{RES}/rq2_{m}.txt")]
if files:
    print("\nTABLE C (RQ2): TopDown against MCTS-Judge. pairwise = pairwise accuracy (ties count half); MCC at threshold 0; cost = MCTS-Judge / TopDown.")
    print(f"{'Model':9s} {'Benchmark':14s} {'judge':12s} {'pairwise':>9s} {'MCC':>6s}   {'tokens':>7s} {'calls':>6s}")
    for m in files:
        for blk in open(f"{RES}/rq2_{m}.txt").read().split("##### ")[1:]:
            h = blk.split("\n")[0]
            if "TopDown vs" in h: continue
            bench = "HumanEvalPack" if "HumanEvalPack" in h else "MBPP+"
            cost = re.search(r"MCTS-Judge / TopDown:\s+([\d.]+)x tokens, ([\d.]+)x calls", blk)
            for l in blk.split("\n"):
                mm = re.match(r"^(TopDown|MCTS-Judge)\s+([\d.]+)\s+([-\d.]+)\s*$", l)
                if mm: print(f"{M[m]:9s} {bench:14s} {mm.group(1):12s} {float(mm.group(2)):9.3f} {float(mm.group(3)):6.3f}" + (f"   {cost.group(1) + 'x':>7s} {cost.group(2) + 'x':>6s}" if cost and mm.group(1) == "MCTS-Judge" else ""))
    print("\nTABLE D (RQ2): TopDown minus MCTS-Judge: difference, relative gain, 95% bootstrap CI, Holm-adjusted p over the 4 comparisons of each model. * = significant (p < 0.05).")
    print(f"{'Model':9s} {'Benchmark':14s} {'Metric':9s} {'diff':>7s} {'gain':>8s} {'95% CI':>18s} {'Holm p':>7s}")
    for m in files:
        for l in open(f"{RES}/rq2_{m}.txt"):
            mm = re.match(r"^(HumanEvalPack|MBPP\+)\s+(pairwise|MCC)\s+vs MCTS-Judge\s+([+-][\d.]+) \(\s*([+-][\d.]+%)\) \[([^\]]+)\].*Holm p=([\d.]+) (\*|ns)", l)
            if mm:
                b, me, d, g, ci, p, s = mm.groups(); print(f"{M[m]:9s} {b:14s} {me:9s} {d:>7s} {g:>8s} {'[' + ci + ']':>18s} {p:>7s} {s}")
print("\nEvery baseline generates at least as many tokens per program as TopDown; MCTS-Judge keeps its own settings.")
