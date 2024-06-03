#!/bin/bash

TRAIN_FILE="./data/wiki1m_for_simcse.txt"
OUTPUT_DIR="./output/my-unsup-simcse-bert-base-uncased"

torchrun --nproc_per_node 4 src/train.py \
    --model bert-base-uncased \
    --train_file $TRAIN_FILE \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --fuse_batch \
    --learning_rate 3e-5 \
    --evaluation_strategy steps \
    --eval_steps 125 \
    --save_strategy steps \
    --sts_eval \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --max_length 32 \
    --overwrite_output_dir \
    --temperature 0.05 \
    --do_train \
    --do_eval \
    "$@"
