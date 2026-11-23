"""TopDown's feedback round, applied to a saved first pass (topdown_run.py). Training-free and execution-free.
For every check whose observation differs from its expectation (a prediction error):
  UP   (specification side, still blind to the code): told only that a check contradicted its expectation for this call (the code's value is NOT
       shown), the re-deriver agent works the result out step by step K_UP = 3 times; the new samples are pooled with the original K -> revised
       expectation and precision.
  DOWN (code side, still blind to the expectation): the re-tracer agent, which sees only the code, traces that single call again with attention on
       every variable, loop and arithmetic step -> a third reading; the majority of {code tracer, spec-aware tracer, re-trace} is the observation,
       reliability 0.9 if at least two agree, else 0.6.
The comparison is then repeated. Writes topdown_<TAG>_<bench>_* like topdown_run.py, with td_arbiter = vanilla + evidence (evidence.py).
Usage: BENCHES=hep,hepx,mbppplus BASE_TAG=<first-pass tag> python topdown_feedback.py MODEL TAG OUTDIR   (reads topdown_<BASE_TAG>_<bench>_transcripts.jsonl)"""
import os, sys, json, re, math, hashlib
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from evidence import evidence
SRC = open(os.path.join(HERE, "topdown_run.py")).read()
exec(SRC[:SRC.index("for bench in todo:")])
BASE = os.environ["BASE_TAG"]; K_UP = 3
def lines(p):
    """Read a transcripts file stored plain (.jsonl) or compressed (.jsonl.gz)."""
    import gzip
    return gzip.open(p + ".gz", "rt") if not os.path.exists(p) and os.path.exists(p + ".gz") else open(p)
UP = ("\n\n" + NOTSEEN + "A check suggests that a correct implementation may NOT return the value previously expected for the call `{probe}`. "
      "From the specification alone, work out step by step what a correct implementation returns for this call; it must have the return type declared "
      "in the function signature. " + JSONFMT.replace("{", "{{").replace("}", "}}") + " End with a line 'RESULT: <value>'.")
DOWN = ("\n\nWithout running it, trace the code step by step on the call `{probe}`. An earlier reading of this call may contain a mistake: track every "
        "variable, every loop iteration and every arithmetic operation explicitly, and do not skip steps. End with a line 'RESULT: <value the code returns>'. "
        + JSONFMT.replace("{", "{{").replace("}", "}}"))
lastres = lambda t: value_of((re.findall(r"RESULT\**\s*:\s*\**(.+?)\s*$", t, re.M | re.I) or [None])[-1])
for bench in todo:
    items = {it["id"]: it for it in load(bench)}
    OLD = {json.loads(l)["id"]: json.loads(l) for l in lines(f"{OUTD}/topdown_{BASE}_{bench}_transcripts.jsonl")}
    VB = json.load(open(f"{OUTD}/topdown_{BASE}_{bench}_verdicts.json"))
    ids = [i for i in OLD if i in items]
    if LIMIT: keep = sorted({i.split("|")[0] for i in ids})[:LIMIT]; ids = [i for i in ids if i.split("|")[0] in keep]
    COST.clear()
    jobs = [(i, j) for i in ids for j, c in enumerate(OLD[i]["checks"]) if c["disc"]]
    print(f"{MODEL} {bench}: {len(ids)} candidates, {len(jobs)} prediction errors sent through the feedback round", flush=True)
    upkeys = sorted({(i.split("|")[0], OLD[i]["checks"][j]["call"]) for i, j in jobs})
    task_item = {i.split("|")[0]: items[i] for i in ids}
    up = dict(zip(upkeys, gen([P([{"role": "user", "content": spec_blk(task_item[t]) + UP.format(probe=call)}]) for t, call in upkeys], "td", 768, 0.7, K_UP)))
    down = dict(zip(jobs, (o[0] for o in gen([P([{"role": "user", "content": f"Code ({items[i]['lang']}):\n```\n{strip_doc(items[i]['code'], items[i]['lang'])}\n```"
                + DOWN.format(probe=OLD[i]["checks"][j]["call"])}]) for i, j in jobs], "td", 1024))))
    R, T, st = {}, {}, Counter()
    for i in ids:
        old = OLD[i]; t = i.split("|")[0]; probes_ = old.get("probes", []); checks = []
        orig = [parse_preds(s, len(probes_)) for s in old.get("preds", [])]
        for j, c in enumerate(old["checks"]):
            c = dict(c)
            if c["disc"]:
                st["revisited"] += 1
                k = probes_.index(c["call"]) + 1 if c["call"] in probes_ else None
                pool = [norm(p[k]) for p in orig if k and k in p] + [norm(v) for v in (lastres(x) for x in up[(t, c["call"])]) if v is not None]
                rawmap = {norm(p[k]): p[k] for p in orig if k and k in p}; rawmap.update({norm(v): v for v in (lastres(x) for x in up[(t, c["call"])]) if v is not None})
                if pool:
                    (nv, cnt), = Counter(pool).most_common(1); newp = cnt / len(pool)
                    st["expectation_changed"] += nv != norm(c["prior"]); c["prior"], c["precision"] = rawmap[nv], newp
                rt = lastres(down[(i, j)]); c["retrace"] = rt
                obs = [v for v in (c["code_only"], c["with_spec"], rt) if v is not None]
                if obs:
                    (ov, oc), = Counter(norm(v) for v in obs).most_common(1)
                    c["r"] = R_AGREE if oc >= 2 else R_SINGLE; c["observed"] = next(v for v in obs if norm(v) == ov)
                    c["disc"] = ov != norm(c["prior"])
                st["still_discrepant"] += c["disc"]
            checks.append(c)
        ev = evidence(checks); R[i] = dict(vanilla=VB[i]["vanilla"], td_errors=ev, td_arbiter=VB[i]["vanilla"] + ev)
        T[i] = dict(id=i, intent=old.get("intent", ""), probes=probes_, preds=old.get("preds", []), checks=checks)
    raw = json.dumps(R, sort_keys=True).encode(); n = len(ids); per = {m: {k: round(v / n, 1) for k, v in c.items()} for m, c in COST.items()}
    json.dump(dict(model=MODEL, variant=dict(first_pass=BASE, rederive_samples=K_UP), n_items=n, per_candidate=per, td_stats=dict(st), verdicts_sha256=hashlib.sha256(raw).hexdigest()),
              open(f"{OUTD}/topdown_{TAG}_{bench}_meta.json", "w"), indent=1)
    open(f"{OUTD}/topdown_{TAG}_{bench}_verdicts.json", "wb").write(raw)
    with open(f"{OUTD}/topdown_{TAG}_{bench}_transcripts.jsonl", "w") as fh:
        for i in ids: fh.write(json.dumps(T[i]) + "\n")
    json.dump({i: items[i]["label"] for i in ids}, open(f"{OUTD}/topdown_{TAG}_{bench}_labels.json", "w"))
    print(f"{bench}: saved; {dict(st)}; feedback cost per candidate {json.dumps(per)}", flush=True)
