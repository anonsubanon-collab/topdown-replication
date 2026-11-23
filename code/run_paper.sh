#!/bin/bash
set -e
cd "$(dirname "$0")"
OUT=${OUT:-../runs}
QWEN=Qwen/Qwen2.5-Coder-7B-Instruct
LLAMA=meta-llama/Llama-3.1-8B-Instruct

for b in hep hepx mbppplus; do
  BENCHES=$b MA=0 python topdown_run.py $QWEN  qwen7b_firstpass  $OUT
  BENCHES=$b MA=0 python topdown_run.py $LLAMA llama8b_firstpass $OUT
  BENCHES=$b BASE_TAG=qwen7b_firstpass  python topdown_feedback.py $QWEN  qwen7b_feedback  $OUT
  BENCHES=$b BASE_TAG=llama8b_firstpass python topdown_feedback.py $LLAMA llama8b_feedback $OUT
  BENCHES=$b TD_SKIP=1 MA=1 MA_REPEAT=4 N_SC=50 python topdown_run.py $QWEN  qwen7b_baselines  $OUT
  BENCHES=$b TD_SKIP=1 MA=1 MA_REPEAT=4 N_SC=30 python topdown_run.py $LLAMA llama8b_baselines $OUT
done
BENCHES=mbppplus TD_SKIP=1 MA=1 MA_REPEAT=5 N_SC=0  python topdown_run.py $QWEN  qwen7b_ma5   $OUT
BENCHES=mbppplus TD_SKIP=1 MA=1 MA_REPEAT=5 N_SC=0  python topdown_run.py $LLAMA llama8b_ma5  $OUT
BENCHES=mbppplus TD_SKIP=1 MA=0 N_SC=50             python topdown_run.py $LLAMA llama8b_sc50 $OUT
BENCHES=hep,hepx,mbppplus python mcts_judge_run.py $QWEN  qwen7b_mctsjudge  $OUT
BENCHES=hep,hepx,mbppplus python mcts_judge_run.py $LLAMA llama8b_mctsjudge $OUT
bash score.sh
