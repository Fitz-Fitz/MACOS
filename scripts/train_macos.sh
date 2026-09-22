#!/bin/bash
# Train MACOS (MA-GRU + Global Motion Loss + Physics-Informed Vessel Loss) with the settings used for the paper.
# Run from the repository root:  bash scripts/train_macos.sh [DATA_DIR]
# Results are written to ./results/<dataset name>/<run name>/
set -e
DATA_DIR=${1:-./data/DSA_sequences}
EPOCHS=${EPOCHS:-150}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_WORKERS=${NUM_WORKERS:-8}
FRAME_NUM=${FRAME_NUM:-6}        # frames sampled per sequence (6 = 2 FPS for the 3-second sequences of the paper)
REG_WEIGHT=${REG_WEIGHT:-0.3}    # lambda_1: Global Motion Loss weight
FILLING_WEIGHT=${FILLING_WEIGHT:-0.5}   # lambda_2: Physics-Informed Vessel Loss weight

python ./code/train.py \
    --data_dir "$DATA_DIR" --frame_num $FRAME_NUM \
    --batch_size $BATCH_SIZE --epochs $EPOCHS --num_workers $NUM_WORKERS \
    --optimizer adamw --lr 3e-4 --weight_decay 1e-4 \
    --scheduler_type cosine --warmup_epochs 30 --min_lr_ratio 0.00005 \
    --augmentation --deep_supervision --mixed_precision \
    --focal_weight 0.8 --focal_alpha 0.25 --focal_gamma 2.0 \
    --reg_weight $REG_WEIGHT --reg_type mask --filling_weight $FILLING_WEIGHT
