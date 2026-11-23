# TopDown: replication package

**Project website: [anonsubanon-collab.github.io/topdown-replication](https://anonsubanon-collab.github.io/topdown-replication/)**

Replication package for the paper *TopDown: A Feedback-Driven Multi-Agent Method for Strengthening LLM Code Judges* (anonymous submission).

TopDown strengthens an existing LLM code judge. Specification-only agents propose inputs and predict the expected output on each; code-only agents trace
the program on the same inputs; a feedback round re-examines every disagreement; the surviving disagreements are weighted and added to the judge's own
log-odds. The package contains the code that ran every judge, the frozen outputs of every run in the paper, and the scripts that turn those outputs into
the paper's numbers and figures.

Two open judges (Qwen2.5-Coder-7B-Instruct, Llama-3.1-8B-Instruct), two benchmarks (HumanEvalPack in six languages, 1,968 programs; MBPP+ pairs, 418 programs)
and five methods: the vanilla judge, self-consistency, multi-agent judging, MCTS-Judge (our re-implementation) and TopDown.

## Layout

```
code/
  topdown_run.py       TopDown's first pass (probe, predictor, two tracers), the vanilla judge, and the baselines
                       (self-consistency, multi-agent judging); needs a GPU and vLLM
  topdown_feedback.py  the feedback round (re-deriver, re-tracer) on top of a first-pass run
  mcts_judge_run.py    MCTS-Judge, re-implemented from its paper (RQ2)
  evidence.py          the evidence rule: from the checks of one program to the number added to the judge's log-odds
  score_paper.py       RQ1: scores, bootstrap intervals, Holm-corrected p-values        (CPU only)
  score_mcts.py        RQ2: pairwise accuracy, MCC, cost, against MCTS-Judge            (CPU only)
  make_tables.py       collects the scorers' output into plain-text tables
  score.sh             runs the scorers on runs/ and writes results/
  run_paper.sh         the GPU commands that produced runs/
data/
  mbppplus_pairs.json  the 418 MBPP+ programs (209 tasks x {correct, buggy}) with labels
runs/                  frozen outputs of every run in the paper (see below)
results/               the tables the paper's numbers and figures were read from
docs/                  the project web page (open docs/index.html)
```

HumanEvalPack is loaded from the Hugging Face dataset `bigcode/humanevalpack` (`test` split, all six languages); it is not copied here.

## Runs

Each run is four files in `runs/`, named `topdown_<tag>_<bench>_{verdicts,labels,meta,transcripts}` (the transcripts are stored compressed, `.jsonl.gz`; the scripts read them as they are):

| tag | what it is |
|---|---|
| `<judge>_firstpass` | TopDown's first pass and the vanilla judge: the `vanilla` score in the verdicts, and per-check transcripts (expectation, precision, both tracer readings) |
| `<judge>_feedback` | the feedback round applied to `<judge>_firstpass`; TopDown = `vanilla` + the evidence of these checks |
| `<judge>_baselines` | self-consistency (`vsc`) and multi-agent (`ma_arbiter`) baselines |
| `<judge>_ma5` | multi-agent x5 on MBPP+ (where x4 generated fewer tokens than TopDown) |
| `llama8b_sc50` | self-consistency x50 on MBPP+ for Llama (where x30 generated fewer tokens than TopDown) |
| `<judge>_mctsjudge` | MCTS-Judge: the binary verdict `mcts_bin` (+1 correct, -1 incorrect), with generated tokens and LLM calls per program |

`<judge>` is `qwen7b` or `llama8b`; `<bench>` is `hep` (HumanEvalPack Python/JS/Java), `hepx` (C++/Go/Rust) or `mbppplus`. The paper pools `hep` and `hepx` as
"HumanEvalPack".

`*_meta.json` records the model, the settings, generated tokens and LLM calls per program, and the SHA-256 of the verdict file; the scorers check that hash
before reading. `*_labels.json` holds the execution labels, which are read only by the scorers, after all verdicts are frozen.

The baselines in the paper are self-consistency x50 (Qwen, both benchmarks; Llama MBPP+) and x30 (Llama HumanEvalPack), and multi-agent x4 (HumanEvalPack)
and x5 (MBPP+). Repeat counts were raised per benchmark until each baseline generated at least as many tokens per program as TopDown. `score_paper.py`
picks the runs automatically: it uses `ma5` and `sc50` where they exist.

## Reproduce the tables (CPU, a few minutes)

```bash
pip install -r requirements.txt
bash code/score.sh
```

This rewrites `results/paper_<judge>.txt` (RQ1: Kendall's tau-b and MCC, intervals, Holm p-values, tokens), `results/rq2_<judge>.txt` (RQ2: pairwise accuracy,
MCC, tokens and LLM calls against MCTS-Judge) and `results/paper_tables.txt` (the values plotted in Figures 2 and 3). The bootstrap uses a fixed seed, so the
output is identical to the committed files. RQ2 is scored only when the `<judge>_mctsjudge` runs are in `runs/`.

## Rerun the judges (GPU)

`code/run_paper.sh` lists the exact commands, in order, with the environment variables that select each configuration. Requirements: one GPU with 80 GB
(bf16), vLLM 0.11.0, a Hugging Face token with the Llama 3.1 licence accepted (`HF_TOKEN`). The runs are seeded (`seed=0`), but vLLM sampling is not
bit-reproducible across hardware, so a rerun gives the same conclusions, not the same verdict files.

## Metrics

- **Kendall's tau-b** between the judge's score and the execution label, pooled over programs (RQ1).
- **MCC** of `score > 0` against the execution label, i.e. accept when P(correct) > 0.5 (RQ1, RQ2).
- **Pairwise accuracy**: the fraction of correct-buggy pairs of one problem (984 in HumanEvalPack, 209 in MBPP+) in which the judge scores the correct
  program higher; ties count half, so a binary verdict scores its balanced accuracy (RQ2).
- **Uncertainty**: 95% bootstrap intervals of the difference, 2,000 resamples of whole problems (a HumanEvalPack problem with its six language variants is one
  unit; an MBPP+ task is one unit), seed 0. p-values are two-sided bootstrap p-values with Holm correction over the comparisons of each judge (12 in RQ1,
  4 in RQ2).
- **Cost**: generated tokens and LLM calls per program, counted the same way for every method. Task-level calls (probe and predictor agents; MCTS-Judge's test
  construction) are divided over the task's programs.

## Method constants

All constants are fixed in the code and were not tuned on labels: predictor samples K = 4, inputs Q = 6, re-derivations and re-traces 3, tracer reliability
r = 0.9 when the two tracers agree and 0.6 otherwise, exposure Q = 0.5 in the evidence term. The evidence rule is `evidence()` in `code/evidence.py`;
TopDown's score is the vanilla log-odds plus that evidence, accepted when positive.

## MCTS-Judge (RQ2)

MCTS-Judge (Wang et al., arXiv 2502.12468) releases no code, so `code/mcts_judge_run.py` rebuilds it from the paper with the same judge model. It follows the
paper's search settings (depth 9, 8 rollouts, 3 simulated tests, 7 simulations each, UCT constant 3, up to 2,048 generated tokens per call) and its
published prompts, without the reference-code lines. Where it differs from the paper: the judged model writes and validates the test cases (the paper uses
GPT-4o), and the eight subtask prompts the paper does not publish use the same template with our wording. The full list of choices is at the top of the
script. MCTS-Judge is not budget-matched: it keeps its own settings, and its tokens and calls per program are reported in `results/rq2_<judge>.txt`.
