#!/bin/bash
# Inference + metrics (Dice / IoU / Precision / Recall / ASD on the key frame) for a finished run.
# Run from the repository root:  bash scripts/test.sh <RUN_DIR> [DATA_DIR] [SPLIT]
# Predictions go to <RUN_DIR>/inference/<SPLIT>/, metrics to <RUN_DIR>/inference/metrics_<SPLIT>.{csv,txt}
set -e
RUN_DIR=$1
DATA_DIR=${2:-./data/DSA_sequences}
SPLIT=${3:-test}
[ -z "$RUN_DIR" ] && { echo "usage: bash scripts/test.sh <RUN_DIR> [DATA_DIR] [SPLIT]"; exit 1; }

python ./code/test.py --model_path "$RUN_DIR" --data_dir "$DATA_DIR" --split "$SPLIT"
