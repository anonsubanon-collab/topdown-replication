"""MCTS-Judge (Wang et al., arXiv 2502.12468) re-implemented as a same-model baseline on the items of topdown_run.py. No code is released, so the
method is rebuilt from the paper: Sec. 3, Table 6 and the prompts of Tables 8 and 10-13.

FOLLOWED AS IN THE PAPER
  Table 6   max depth 9, rollouts 8, UCT constant alpha 3, LLM weight w_l 0.1, UCT weight w_u 0.9, validations beta 5, test cases gamma 3,
            simulations delta 7, reward eps 1.1, temperature 0.4, top_p 0.95, top_k 40, maximum output tokens 2048 on every LLM call.
  Prompts   Tables 8 (vanilla, used as g), 10 (self-assessment), 11 (code-logic subtask, analysis summarization), 12 (test generation,
            JSON reformatting, validation), 13 (simulated execution, comparison), with the reference-code lines removed (the paper's
            no-reference setting). Each table's <|assistant|> text is a prefill that the model continues from.
  Tree      Every node has two children: one non-repeating subtask (drawn at random when the node is created) and the null action.
            Selection descends while the node is fully expanded, taking UCT's child (Eq. 1) with probability w_u and the LLM
            self-assessment's choice with probability w_l. Expansion adds a random unused child; simulation then takes non-null actions
            until max depth; backpropagation adds the terminal reward to Q and 1 to N of every node on the trajectory.
  Reward    Test cases are built from the problem statement only, reformatted to JSON and validated beta times; a test is kept only if
            every validation says PASS. gamma stored tests are drawn at random, their outputs masked, each simulated delta times, the
            simulated output compared with the stored one, majority per test; h(x) = correct iff every drawn test passes. A trajectory
            gets reward eps iff f(t, g) == h(x).
  Answer    R(t) = sum of r(s) over the trajectory's nodes; weighted sampling over the trajectories picks t_b; the prediction is f(t_b, g).

WHERE THIS DIFFERS FROM THE PAPER
  D1  Test cases are generated, reformatted and validated by the judged model itself (paper: GPT-4o), so every method runs on one model.
  D2  Only the code-logic subtask prompt is published (Table 11). The other eight subtasks use the same template with our wording of the task
      and of evaluation step 2 (FOCUS below).

CHOICES WHERE THE PAPER IS SILENT
  I1  Number of subtasks: 9, equal to the max depth.
  I2  {difficulty} in Table 12: "medium".
  I3  Subtask prompts contain exactly Table 11's fields.
  I4  {example} in Tables 8 and 11: MBPP+ - its 'Your code should satisfy this example' assert, split off the problem statement;
      HumanEvalPack - left empty, since its examples are inside the docstring that forms the problem statement.
  I5  g = the Table 8 vanilla verdict, one sample.
  I6  f(t, g) = majority over the trajectory's subtask verdicts plus g; a tie goes to g.
  I7  h(x) depends on x only, so it is computed once per program, at the first rollout that reaches max depth, and reused.
  I8  r(s) = Q(s)/N(s), so R(t) is the sum of the mean rewards of t's nodes; if every R(t) is 0 the draw is uniform.
  I9  Nodes reached during simulation are stored in the tree, so later rollouts can select them.
  I10 Answer parsing: the first Yes/No word of a yes/no answer (none -> No); the last PASS/FAIL of a validation (none -> FAIL); the last
      MATCH / NOT MATCH of a comparison (none -> NOT MATCH). Test cases: every {name: {"input", "expected_output"}} object in the reformatted
      JSON. The test case shown to the simulator is the same object without "expected_output".
  I11 Every LLM call gets its own seed, derived from what the call is (program, node, rollout, ...), so repeated calls are independent samples.
  I12 Each assistant prefill is followed by a newline.
  I13 Self-assessment (Table 10): only an explicit No skips the proposed subtask. For the yes/no prompts g and analysis summarization a missing
      answer counts as No.

OUTPUT per program: mcts_bin = f(t_b, g) as +1 (correct) / -1 (incorrect), the paper's binary verdict; h, g. The meta file records generated tokens
and LLM calls per program (test construction included) and per stage.
Usage: BENCHES=hep,hepx,mbppplus python mcts_judge_run.py MODEL TAG OUTDIR"""
import json, os, re, sys, math, random, hashlib
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL, TAG, OUTD = sys.argv[1:4]; os.makedirs(OUTD, exist_ok=True)
BENCHES = os.environ.get("BENCHES", "hep,hepx,mbppplus").split(","); LIMIT = int(os.environ.get("LIMIT", "0"))
DEPTH, ROLLOUTS, ALPHA, W_L, W_U, BETA, GAMMA, DELTA, EPS = 9, 8, 3.0, 0.1, 0.9, 5, 3, 7, 1.1
TEMP, TOP_P, TOP_K, MAX_TOKENS = 0.4, 0.95, 40, 2048
DIFFICULTY = os.environ.get("MCTS_DIFFICULTY", "medium")
PAIRS = os.environ.get("PAIRS", os.path.join(HERE, "..", "data", "mbppplus_pairs.json"))
src = open(os.path.join(HERE, "topdown_run.py")).read()
from datasets import load_dataset
exec(src[src.index("def load(bench):"):src.index("\nllm = LLM(")])

from vllm import LLM, SamplingParams
llm = LLM(MODEL, dtype=os.environ.get("DTYPE", "bfloat16"), max_model_len=int(os.environ.get("MAXLEN", "8192")),
          gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.9")), seed=0, enforce_eager=bool(int(os.environ.get("EAGER", "0"))))
tok = llm.get_tokenizer()

import time
COST, STATS = Counter(), Counter()
STAGE, PROG, PROG_P = {}, Counter(), Counter()
GROUP = {"test_generation": "test construction", "test_reformat": "test construction", "test_validation": "test construction",
         "global_g": "search", "self_assessment": "search", "subtask_analysis": "search", "subtask_summary": "search",
         "simulation": "simulated execution", "comparison": "simulated execution"}
seed_of = lambda *key: int(hashlib.md5("|".join(map(str, key)).encode()).hexdigest()[:8], 16)
rng_of = lambda *key: random.Random(seed_of(*key))
def chat(system, user, prefill):
    return tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}], tokenize=False, add_generation_prompt=True) + prefill + "\n"
def gen(prompts, seeds, stage, owners, n=1):
    if not prompts: return []
    t0 = time.time()
    outs = llm.generate(prompts, [SamplingParams(temperature=TEMP, top_p=TOP_P, top_k=TOP_K, n=n, max_tokens=MAX_TOKENS, seed=s) for s in seeds], use_tqdm=False)
    st = STAGE.setdefault(stage, Counter()); st["seconds"] += time.time() - t0; st["requests"] += len(outs)
    for o, own in zip(outs, owners):
        g = sum(len(s.token_ids) for s in o.outputs); p = len(o.prompt_token_ids) * len(o.outputs); c = sum(1 for s in o.outputs if len(s.token_ids) >= MAX_TOKENS)
        st["calls"] += len(o.outputs); st["gen_tokens"] += g; st["prompt_tokens"] += p; st["capped_outputs"] += c; st["empty_outputs"] += sum(1 for s in o.outputs if not s.text.strip())
        PROG[own] += g; PROG_P[own] += p
        COST["calls"] += len(o.outputs); COST["gen_tokens"] += g; COST["gen_outputs"] += len(o.outputs); COST["capped_outputs"] += c; COST["prompt_tokens"] += p
    return [[s.text for s in o.outputs] for o in outs]
UNPARSED = {}
def _keep(what, t):
    if len(UNPARSED.setdefault(what, [])) < 25: UNPARSED[what].append(t[-600:])
def yes_no(t, what):
    m = re.search(r"\b(yes|no)\b", t, re.I)
    if not m: STATS[f"{what}_unparsed"] += 1; _keep(what, t)
    return bool(m) and m.group(1).lower() == "yes"
def last_of(t, pattern, what):
    m = re.findall(pattern, t, re.I)
    if not m: STATS[f"{what}_unparsed"] += 1; _keep(what, t)
    return m[-1].upper() if m else None

VAN_SYS = "Determine the correctness of the code snippet. Output Yes or No."
VAN_USR = "Problem Statement:{problem}\nExample:{example}\nCode Snippet ({language}):{code}"
VAN_PRE = "Answer (Yes or No only):"
SA_SYS = ("You are a code evaluation planning expert. Your task is to assess whether the suggested evaluator agent should proceed based on "
          "the provided problem statement, code snippet, and evaluation history. Determine if this evaluator will enhance coverage or "
          "completeness of the assessment.")
SA_USR = ("Problem Statement: {problem}\nCode Snippet:{language}{code}\nProposed Next Evaluator: {agent}\nEvaluation History: Agents previously "
          "used: {history}\nInstruction: Skip the evaluation only if it very negatively affect the assessment. Otherwise, please respond 'Yes' "
          "to include the evaluator.")
SA_PRE = "Decision: (Yes or No only)"
SUB_SYS = ("You will be provided with a problem statement and a code snippet that supposedly addresses the problem in {language}. Your task is "
           "to {task}. Do not provide a corrected version.\nEvaluation Steps:1. Read the problem statement carefully and identify the required "
           "functionalities of the implementation. You can refer to the example to understand the problem better.\n2. {step} 3. Finally, "
           "conclude your evaluation.")
SUB_USR = "Problem Statement:{problem}\nExample:{example}\nCode Snippet ({language}):{code}"
SUB_PRE = "Evaluation ({label}):"
SUM_SYS = 'You will be provided with an analysis result of a code snippet. If the analysis believes that the code snippet is correct, output: "Yes". Otherwise, output: "No".'
SUM_USR = "Analysis Result:{analysis}"
SUM_PRE = "Final Answer (Yes or No only):"
GEN_SYS = ("You are an AI assistant specializing in problem analysis and test case generation, with particular expertise in {difficulty} test "
           "cases. Your task is to generate comprehensive test cases based on the given problem description.")
GEN_USR = ("Problem:{problem}\n### Instructions: Please analyze the problem statement carefully and create five well-rounded test cases. The test "
           "cases should be at {difficulty} difficulty level. Your test cases should: 1. Cover a variety of scenarios to thoroughly validate code "
           "correctness 2. Avoid excessively large computations 3. Avoid extreme edge cases unless specifically required 4. Include common use "
           "cases and reasonable boundary conditions 5. Focus on practical, real-world scenarios\nFor each test case, please provide:- Input "
           "values - Expected output- Brief explanation of what the test case verifies.")
GEN_PRE = ("I'll analyze the problem systematically and create carefully curated test cases.\n**Analysis Approach:**1. First, I'll identify the "
           "key requirements and constraints 2. Then, I'll determine important edge cases and boundary conditions 3. Finally, I'll design test "
           "cases that progressively increase in complexity\nLet's examine each test case:")
JSON_SYS = "You are an AI assistant specializing in test case reformatting. Your task is to extract and reformat the test cases based on the examples provided."
JSON_USR = ('Problem:{problem}\nTest Cases:{test_cases}\n### Instructions:Reformat the test cases and return them in a list of JSON format, where '
            'each test case is structured as follows:\n{{"<root_function_name>": {{ "input": "<input>" (as a string), "expected_output": '
            '"<expected_ output>" (as a string)}}}}\nUse the main function name identified in each test case as the key. Retain the original '
            'input and output formats, ensuring that all provided test cases are included.')
JSON_PRE = "Reformatted Test Cases only, without explanation:"
VAL_SYS = "You are an AI assistant specializing in test case validation. Your task is to assess the correctness of a given test case based on the problem description."
VAL_USR = ("Problem:{problem}\nTest Case:{test_case}\n### Instructions: Validate the correctness of the test case and determine if it aligns with "
           "the expected behavior outlined in the problem description. Do not provide a corrected version. Return PASS or FAIL to indicate the "
           "accuracy of the given test case.")
VAL_PRE = "Let's validate the correctness of the test case step-by-step:"
SIM_SYS = "You are an AI assistant skilled in executing {language} scripts."
SIM_USR = ("Execute the following {language} script. Analyze the code and run each subfunction in the test case step-by-step.\nCode:{code}\n"
           "Test Case:{test_case}\nInstruction: Act as a {language} interpreter to execute the code line-by-line, tracking changes in each "
           "variable throughout the process. Based on this execution trace, determine the output of the unit test case.")
SIM_PRE = "Let's execute the code step-by-step, analyzing the input test case:"
CMP_SYS = ("You are an expert in code validation with a focus on comparing code execution outputs. Your goal is to determine if the actual output "
           "matches the expected behavior, being flexible about formatting and minor differences that don't affect correctness.")
CMP_USR = ("Problem: {problem}\nTest Case:{test_case}\nActual Output:{answer}\n### Instruction: 1. Extract the final execution result from the "
           "actual output. Compare this result with the expected answer, focusing on correctness rather than specific formatting. Note: Do not "
           "judge the code; simply validate whether the results match.")
CMP_PRE = "Let's check the results, and conclude with **MATCH** or **NOT MATCH** after the comparison."
FOCUS = [("Code Logic Evaluation", "Code Logic Analysis", "check if the code snippet covers the required functionalities",
          "Read the code snippet and analyze its logic. Check if the code snippet covers all the required functionalities of the problem."),
         ("Boundary Condition Evaluation", "Boundary Condition Analysis", "check if the code snippet handles the boundary conditions implied by the problem",
          "Identify the boundary inputs (empty, minimal, maximal, equal values) and check how the code snippet handles each of them."),
         ("Edge Case Evaluation", "Edge Case Analysis", "check if the code snippet handles the special cases the problem mentions",
          "List the special cases stated or implied by the problem and check the code snippet against each of them."),
         ("Input Constraint Evaluation", "Input Constraint Analysis", "check if the code snippet respects the input constraints and assumptions of the problem",
          "Identify the input constraints and check that the code snippet works for every input the problem allows."),
         ("Output Format Evaluation", "Output Format Analysis", "check if the code snippet returns outputs in the required type and format",
          "Identify the required return type and format and check that every path of the code snippet returns it."),
         ("Example Consistency Evaluation", "Example Consistency Analysis", "check if the code snippet reproduces the examples of the problem",
          "Take each example of the problem and check whether the code snippet produces the stated result."),
         ("Control Flow Evaluation", "Control Flow Analysis", "check if the loops and branches of the code snippet are correct",
          "Check loop bounds, termination, off-by-one errors and branch conditions of the code snippet against the problem."),
         ("Variable and Operator Evaluation", "Variable and Operator Analysis", "check if the code snippet uses the right variables, operators and constants",
          "Check every operator, constant and variable use of the code snippet against what the problem requires."),
         ("Requirement Completeness Evaluation", "Requirement Completeness Analysis", "check if the code snippet implements every requirement of the problem",
          "List every requirement of the problem and check that the code snippet implements each of them and nothing that contradicts them.")]
assert len(FOCUS) == DEPTH

def fields(it, bench):
    k = "Your code should satisfy this example:"
    if bench == "mbppplus" and k in it["spec"]:
        d, e = it["spec"].split(k, 1); return d.replace('"""', "").strip(), e.replace('"""', "").strip()
    return it["spec"], ""
def parse_tests(txt):
    found, dec, i = [], json.JSONDecoder(), 0
    def walk(o, name=None):
        if isinstance(o, dict) and "input" in o and "expected_output" in o:
            v = lambda x: x if isinstance(x, str) else json.dumps(x)
            found.append({name or "function": {"input": v(o["input"]), "expected_output": v(o["expected_output"])}})
        elif isinstance(o, dict):
            for k, x in o.items(): walk(x, k)
        elif isinstance(o, list):
            for x in o: walk(x, name)
    while True:
        j = min([p for p in (txt.find("[", i), txt.find("{", i)) if p >= 0], default=-1)
        if j < 0: break
        try: o, end = dec.raw_decode(txt, j); walk(o); i = end
        except ValueError: i = j + 1
    return list({json.dumps(t, sort_keys=True): t for t in found}.values())
masked = lambda t: json.dumps({n: {"input": v["input"]} for n, v in t.items()})

for bench in BENCHES:
    if os.path.exists(f"{OUTD}/topdown_{TAG}_{bench}_verdicts.json"): print(f"{bench}: exists, skipped"); continue
    items = load(bench); tasks = {}
    for it in items: tasks.setdefault(it["id"].split("|")[0], it)
    tids = sorted(tasks)
    if LIMIT: tids = tids[:LIMIT]; items = [it for it in items if it["id"].split("|")[0] in set(tids)]
    COST.clear(); STATS.clear(); STAGE.clear(); UNPARSED.clear(); PROG.clear(); PROG_P.clear(); T_START = time.time(); tid = lambda it: it["id"].split("|")[0]; byid = {it["id"]: it for it in items}; ids = [it["id"] for it in items]
    F = {k: fields(byid[k], bench) for k in ids}
    print(f"MCTS-Judge {MODEL} {bench}: {len(items)} programs, {len(tids)} tasks", flush=True)

    raw = dict(zip(tids, (o[0] for o in gen([chat(GEN_SYS.format(difficulty=DIFFICULTY), GEN_USR.format(problem=tasks[t]["spec"], difficulty=DIFFICULTY), GEN_PRE)
                                             for t in tids], [seed_of("gen", t) for t in tids], "test_generation", [("task", t) for t in tids]))))
    js = dict(zip(tids, (o[0] for o in gen([chat(JSON_SYS, JSON_USR.format(problem=tasks[t]["spec"], test_cases=raw[t]), JSON_PRE) for t in tids],
                                           [seed_of("json", t) for t in tids], "test_reformat", [("task", t) for t in tids]))))
    cand = {t: parse_tests(js[t]) for t in tids}
    vkeys = [(t, j) for t in tids for j in range(len(cand[t]))]
    vout = gen([chat(VAL_SYS, VAL_USR.format(problem=tasks[t]["spec"], test_case=json.dumps(cand[t][j])), VAL_PRE) for t, j in vkeys],
               [seed_of("val", t, j) for t, j in vkeys], "test_validation", [("task", t) for t, j in vkeys], n=BETA)
    tests = {t: [] for t in tids}
    for (t, j), outs in zip(vkeys, vout):
        if all(last_of(o, r"\b(PASS|FAIL)\b", "validation") == "PASS" for o in outs): tests[t].append(cand[t][j])
    STATS["tests_generated"] = sum(map(len, cand.values())); STATS["tests_stored"] = sum(map(len, tests.values()))
    STATS["tasks_without_tests"] = sum(1 for t in tids if not tests[t])
    print(f"  tests per task: {STATS['tests_generated'] / len(tids):.2f} generated, {STATS['tests_stored'] / len(tids):.2f} stored; "
          f"{STATS['tasks_without_tests']} tasks without a stored test; empty validation outputs {STAGE['test_validation']['empty_outputs']}/{STAGE['test_validation']['calls']}", flush=True)
    if STATS["tests_stored"] == 0 or STAGE["test_validation"]["empty_outputs"] > 0.5 * STAGE["test_validation"]["calls"]:
        sys.exit(f"ABORT {bench}: test construction produced no usable tests (stored {STATS['tests_stored']}, empty validations "
                 f"{STAGE['test_validation']['empty_outputs']}/{STAGE['test_validation']['calls']}); not running the search")

    def simulate(keys):
        pick = {key: rng_of("pick", *key).sample(tests[tid(byid[key[0]])], min(GAMMA, len(tests[tid(byid[key[0]])]))) for key in keys}
        sk = [(key, q) for key in keys for q in range(len(pick[key]))]
        sims = gen([chat(SIM_SYS.format(language=byid[key[0]]["lang"]), SIM_USR.format(language=byid[key[0]]["lang"], code=byid[key[0]]["code"],
                    test_case=masked(pick[key][q])), SIM_PRE) for key, q in sk], [seed_of("sim", *key, q) for key, q in sk], "simulation", [key[0] for key, q in sk], n=DELTA)
        ck = [(key, q, d) for (key, q), outs in zip(sk, sims) for d in range(len(outs))]
        txt = {(key, q, d): outs[d] for (key, q), outs in zip(sk, sims) for d in range(len(outs))}
        cmp_ = gen([chat(CMP_SYS, CMP_USR.format(problem=byid[key[0]]["spec"], test_case=json.dumps(pick[key][q]), answer=txt[(key, q, d)]), CMP_PRE)
                    for key, q, d in ck], [seed_of("cmp", *key, q, d) for key, q, d in ck], "comparison", [key[0] for key, q, d in ck])
        hit, tot = Counter(), Counter()
        for (key, q, d), o in zip(ck, cmp_):
            tot[(key, q)] += 1; hit[(key, q)] += last_of(o[0], r"\bNOT[\s*_]*MATCH\b|\bMATCH\b", "comparison") == "MATCH"
        return {key: (all(hit[(key, q)] * 2 > tot[(key, q)] for q in range(len(pick[key]))) if pick[key] else None) for key in keys}, pick

    G = dict(zip(ids, (yes_no(o[0], "g") for o in gen([chat(VAN_SYS, VAN_USR.format(problem=F[k][0], example=F[k][1], language=byid[k]["lang"],
                                                                                         code=byid[k]["code"]), VAN_PRE) for k in ids], [seed_of("g", k) for k in ids], "global_g", ids))))

    tree = {k: {} for k in ids}
    val = {k: {} for k in ids}
    def add(k, node):
        used = {a for a in node if a >= 0}; free = [j for j in range(len(FOCUS)) if j not in used]
        tree[k][node] = dict(Q=0.0, N=0, sub=rng_of("sub", k, node).choice(free) if free else -1)
    for k in ids: add(k, ())
    children = lambda k, node: ([node + (tree[k][node]["sub"],)] if tree[k][node]["sub"] >= 0 else []) + [node + (-1,)]
    H, trajs, hpick = {}, {k: [] for k in ids}, {}
    for r in range(ROLLOUTS):
        path = {k: () for k in ids}; sim = {k: False for k in ids}
        for depth in range(DEPTH):
            choice, sa_jobs = {}, []
            for k in ids:
                node = path[k]; ch = children(k, node); new = [c for c in ch if c not in tree[k]]
                if sim[k]: choice[k] = ch[0]
                elif new: choice[k] = rng_of("expand", k, r, depth).choice(new); sim[k] = True; STATS["expansions"] += 1
                elif len(ch) == 1: choice[k] = ch[0]
                elif rng_of("w", k, r, depth).random() < W_U:
                    u = lambda c: tree[k][c]["Q"] / tree[k][c]["N"] + ALPHA * math.sqrt(math.log(tree[k][node]["N"]) / tree[k][c]["N"])
                    best = max(map(u, ch)); choice[k] = rng_of("tie", k, r, depth).choice([c for c in ch if u(c) == best]); STATS["uct"] += 1
                else: sa_jobs.append((k, node)); STATS["self_assessment"] += 1
            sa = gen([chat(SA_SYS, SA_USR.format(problem=byid[k]["spec"], language=byid[k]["lang"], code=byid[k]["code"], agent=FOCUS[tree[k][node]["sub"]][0],
                      history=", ".join(FOCUS[a][0] for a in node if a >= 0) or "None"), SA_PRE) for k, node in sa_jobs],
                     [seed_of("sa", k, node, r) for k, node in sa_jobs], "self_assessment", [k for k, node in sa_jobs])
            for (k, node), o in zip(sa_jobs, sa):
                m = re.search(r"\b(yes|no)\b", o[0], re.I); STATS["self_assessment_no"] += bool(m) and m.group(1).lower() == "no"
                if not m: STATS["self_assessment_unparsed"] += 1; _keep("self_assessment", o[0])
                choice[k] = children(k, node)[1 if (m and m.group(1).lower() == "no") else 0]
            for k in ids:
                path[k] = choice[k]
                if path[k] not in tree[k]: add(k, path[k])
            ex = [(k, path[k]) for k in ids if path[k][-1] >= 0 and path[k] not in val[k]]
            ana = gen([chat(SUB_SYS.format(language=byid[k]["lang"], task=FOCUS[n[-1]][2], step=FOCUS[n[-1]][3]),
                            SUB_USR.format(problem=F[k][0], example=F[k][1], language=byid[k]["lang"], code=byid[k]["code"]), SUB_PRE.format(label=FOCUS[n[-1]][1]))
                       for k, n in ex], [seed_of("sub", k, n) for k, n in ex], "subtask_analysis", [k for k, n in ex])
            dec = gen([chat(SUM_SYS, SUM_USR.format(analysis=a[0]), SUM_PRE) for a in ana], [seed_of("sum", k, n) for k, n in ex], "subtask_summary", [k for k, n in ex])
            for (k, n), a, d in zip(ex, ana, dec): val[k][n] = (a[0], yes_no(d[0], "summary"))
        need = [(k, "once") for k in ids if k not in H]
        if need:
            h_new, p_new = simulate(need); hpick.update({key[0]: p_new[key] for key in need}); H.update({key[0]: h_new[key] for key in need})
        for k in ids:
            node = path[k]; votes = [val[k][node[:i + 1]][1] for i, a in enumerate(node) if a >= 0] + [G[k]]
            f = G[k] if sum(votes) * 2 == len(votes) else sum(votes) * 2 > len(votes)
            rew = EPS if (H[k] is not None and f == H[k]) else 0.0
            for i in range(len(node) + 1): tree[k][node[:i]]["Q"] += rew; tree[k][node[:i]]["N"] += 1
            trajs[k].append(dict(path=list(node), f=f, h=H[k], reward=rew))
        print(f"  rollout {r + 1}/{ROLLOUTS} done", flush=True)

    R, Tr = {}, {}
    for k in ids:
        Rt = [sum(tree[k][tuple(t["path"][:i + 1])]["Q"] / tree[k][tuple(t["path"][:i + 1])]["N"] for i in range(len(t["path"]))) for t in trajs[k]]
        w = Rt if sum(Rt) > 0 else [1.0] * len(Rt)
        b = rng_of("final", k).choices(range(len(w)), weights=w)[0]; fb = trajs[k][b]["f"]
        R[k] = dict(mcts_bin=1.0 if fb else -1.0, h=H[k], g=G[k])
        Tr[k] = dict(id=k, tests=hpick.get(k, []), h=H[k], g=G[k], best=b, R=Rt, trajectories=trajs[k],
                     nodes={json.dumps(list(n)): dict(subtask=FOCUS[n[-1]][0], verdict=v[1], analysis=v[0]) for n, v in val[k].items()})
        STATS["subtask_nodes_executed"] += len(val[k])
    STATS["h_none"] = sum(1 for k in ids if H[k] is None); STATS["h_correct"] = sum(1 for k in ids if H[k] is True)
    raw_v = json.dumps(R, sort_keys=True).encode(); n = len(items)
    ntask = Counter(tid(byid[k]) for k in ids)
    ptok = {k: PROG[k] + PROG[("task", tid(byid[k]))] / ntask[tid(byid[k])] for k in ids}
    ppro = {k: PROG_P[k] + PROG_P[("task", tid(byid[k]))] / ntask[tid(byid[k])] for k in ids}
    for k in ids: Tr[k]["gen_tokens"] = round(ptok[k], 1); Tr[k]["prompt_tokens"] = round(ppro[k], 1)
    srt = sorted(ptok.values()); q = lambda f: round(srt[min(len(srt) - 1, int(f * len(srt)))], 1)
    budget = dict(
        gen_tokens_per_program=dict(mean=round(sum(srt) / n, 1), median=q(0.5), p10=q(0.1), p90=q(0.9), min=round(srt[0], 1), max=round(srt[-1], 1)),
        prompt_tokens_per_program_mean=round(sum(ppro.values()) / n, 1),
        per_stage={s: dict(gen_tokens_per_program=round(c["gen_tokens"] / n, 1), prompt_tokens_per_program=round(c["prompt_tokens"] / n, 1),
                           calls_per_program=round(c["calls"] / n, 2), capped_outputs=c["capped_outputs"], calls=c["calls"],
                           capped_share=round(c["capped_outputs"] / max(c["calls"], 1), 4), empty_share=round(c["empty_outputs"] / max(c["calls"], 1), 4),
                           seconds=round(c["seconds"], 1)) for s, c in STAGE.items()},
        per_group={g: round(sum(c["gen_tokens"] for s, c in STAGE.items() if GROUP[s] == g) / n, 1) for g in dict.fromkeys(GROUP.values())},
        wall_clock_seconds=round(time.time() - T_START, 1))
    print(f"  token budget per program: mean {budget['gen_tokens_per_program']['mean']:.0f} generated (median {budget['gen_tokens_per_program']['median']:.0f}, "
          f"p90 {budget['gen_tokens_per_program']['p90']:.0f}, max {budget['gen_tokens_per_program']['max']:.0f}); by group "
          + ", ".join(f"{g} {v:.0f}" for g, v in budget["per_group"].items()) + f"; wall clock {budget['wall_clock_seconds'] / 60:.1f} min", flush=True)
    for s, c in budget["per_stage"].items():
        print(f"    {s:18s} {c['gen_tokens_per_program']:8.1f} gen tok/program  {c['calls_per_program']:6.2f} calls/program  {c['capped_share']:6.1%} cut at {MAX_TOKENS}  {c['empty_share']:6.1%} empty  {c['seconds']:7.0f} s", flush=True)
    params = dict(DEPTH=DEPTH, ROLLOUTS=ROLLOUTS, ALPHA=ALPHA, W_L=W_L, W_U=W_U, BETA=BETA, GAMMA=GAMMA, DELTA=DELTA, EPS=EPS, TEMP=TEMP, TOP_P=TOP_P,
                  TOP_K=TOP_K, MAX_TOKENS=MAX_TOKENS, DIFFICULTY=DIFFICULTY, N_SUBTASKS=len(FOCUS), TEST_MODEL="judged model (D1)",
                  MAXLEN=int(os.environ.get("MAXLEN", "8192")))
    json.dump(dict(model=MODEL, method="MCTS-Judge re-implementation (arXiv 2502.12468)", params=params, n_items=n, n_tasks=len(tids), LIMIT=LIMIT,
                   per_candidate={"mcts": {k: round(v / n, 1) for k, v in COST.items()}}, budget=budget, stats=dict(STATS), unparsed_samples=UNPARSED, verdicts_sha256=hashlib.sha256(raw_v).hexdigest()),
              open(f"{OUTD}/topdown_{TAG}_{bench}_meta.json", "w"), indent=1)
    open(f"{OUTD}/topdown_{TAG}_{bench}_verdicts.json", "wb").write(raw_v)
    with open(f"{OUTD}/topdown_{TAG}_{bench}_transcripts.jsonl", "w") as fh:
        for k in ids: fh.write(json.dumps(Tr[k]) + "\n")
    json.dump({it["id"]: it["label"] for it in items}, open(f"{OUTD}/topdown_{TAG}_{bench}_labels.json", "w"))
    print(f"MCTS-Judge {bench}: saved; per-program cost {json.dumps({k: round(v / n, 1) for k, v in COST.items()})}; stats {dict(STATS)}", flush=True)
