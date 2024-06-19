#!/bin/bash

TRAIN_FILE="./data/nq-train.jsonl"
VAL_FILE="./data/nq-dev.jsonl"
OUTPUT_DIR="./output/my-dpr-nq-bert-base-uncased"

mkdir -p $OUTPUT_DIR

.venv/bin/torchrun --nproc_per_node 4 src/train.py \
    --query_model bert-base-uncased \
    --document_model bert-base-uncased \
    --train_file $TRAIN_FILE \
    --validation_file $VAL_FILE \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 40 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 64 \
    --fuse_batch \
    --learning_rate 2e-5 \
    --optim adamw_hf \
    --warmup_steps 1237 \
    --max_grad_norm 2.0 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --metric_for_best_model average_rank \
    --greater_is_better false \
    --load_best_model_at_end \
    --max_seq_length 256 \
    --use_negative 0 \
    --do_train \
    --do_eval \
    "$@"
