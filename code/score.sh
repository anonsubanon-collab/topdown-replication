#!/bin/bash
# Recompute every table in results/ from the frozen runs in runs/. CPU only; a few minutes.
set -e
cd "$(dirname "$0")/.."
mkdir -p results
for m in qwen7b llama8b; do
  python -I code/score_paper.py $m > results/paper_$m.txt                       # RQ1
  if ls runs/topdown_${m}_mctsjudge_*_verdicts.json > /dev/null 2>&1; then      # RQ2 (needs the MCTS-Judge runs)
    python -I code/score_mcts.py $m > results/rq2_$m.txt
  else
    echo "RQ2: no MCTS-Judge runs for $m in runs/, skipped"
  fi
done
python -I code/make_tables.py > results/paper_tables.txt
echo "done: results/"
