#!/bin/bash

TRAIN_FILE="./data/nli_for_simcse.csv"
OUTPUT_DIR="./output/my-sup-simcse-bert-base-uncased"

mkdir -p $OUTPUT_DIR

torchrun --nproc_per_node 4 src/train.py \
    --model bert-base-uncased \
    --train_file $TRAIN_FILE \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 3 \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --fuse_batch \
    --learning_rate 5e-5 \
    --evaluation_strategy steps \
    --eval_steps 125 \
    --save_strategy steps \
    --save_steps 125 \
    --sts_eval \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --max_seq_length 32 \
    --temperature 0.05 \
    --mlp_only_training false \
    --do_train \
    --do_eval \
    "$@"
