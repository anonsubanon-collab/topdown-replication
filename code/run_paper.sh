#!/bin/bash
# The GPU commands that produce runs/, in order. One 80 GB GPU, vLLM 0.11.0, HF_TOKEN with the Llama 3.1 licence accepted.
# Every run writes topdown_<tag>_<bench>_{verdicts,labels,meta,transcripts} into $OUT.
set -e
cd "$(dirname "$0")"
OUT=${OUT:-../runs}
QWEN=Qwen/Qwen2.5-Coder-7B-Instruct
LLAMA=meta-llama/Llama-3.1-8B-Instruct

for b in hep hepx mbppplus; do
  # 1. TopDown first pass and the vanilla judge
  BENCHES=$b MA=0 python topdown_run.py $QWEN  qwen7b_firstpass  $OUT
  BENCHES=$b MA=0 python topdown_run.py $LLAMA llama8b_firstpass $OUT
  # 2. Feedback round on top of the first pass (TopDown = vanilla + evidence from these checks)
  BENCHES=$b BASE_TAG=qwen7b_firstpass  python topdown_feedback.py $QWEN  qwen7b_feedback  $OUT
  BENCHES=$b BASE_TAG=llama8b_firstpass python topdown_feedback.py $LLAMA llama8b_feedback $OUT
  # 3. Baselines at matched tokens (TD_SKIP=1: no TopDown; MA_REPEAT: multi-agent runs; N_SC: self-consistency samples)
  BENCHES=$b TD_SKIP=1 MA=1 MA_REPEAT=4 N_SC=50 python topdown_run.py $QWEN  qwen7b_baselines  $OUT
  BENCHES=$b TD_SKIP=1 MA=1 MA_REPEAT=4 N_SC=30 python topdown_run.py $LLAMA llama8b_baselines $OUT
done
# 4. MBPP+: baselines that generated fewer tokens than TopDown were rerun with more samples
BENCHES=mbppplus TD_SKIP=1 MA=1 MA_REPEAT=5 N_SC=0  python topdown_run.py $QWEN  qwen7b_ma5   $OUT
BENCHES=mbppplus TD_SKIP=1 MA=1 MA_REPEAT=5 N_SC=0  python topdown_run.py $LLAMA llama8b_ma5  $OUT
BENCHES=mbppplus TD_SKIP=1 MA=0 N_SC=50             python topdown_run.py $LLAMA llama8b_sc50 $OUT
# 5. MCTS-Judge re-implementation, same judges, all benchmarks (RQ2)
BENCHES=hep,hepx,mbppplus python mcts_judge_run.py $QWEN  qwen7b_mctsjudge  $OUT
BENCHES=hep,hepx,mbppplus python mcts_judge_run.py $LLAMA llama8b_mctsjudge $OUT
# 6. Tables
bash score.sh
