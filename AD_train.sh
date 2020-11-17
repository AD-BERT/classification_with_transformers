#!/bin/bash

TASK_NAME="SST-2"
MODEL_DIR="/home/ling.liu/ad/models/scibert"
DATA_DIR="/home/ling.liu/PycharmProjects/AD/transformers/examples/text-classification/ADdata"

python run_glue.py \
	--model_name_or_path $MODEL_DIR \
	--task_name $TASK_NAME \
	--data_dir $DATA_DIR \
	--do_train \
	--do_eval \
	--max_seq_length 500 \
	--per_device_train_batch_size 8 \
	--learning_rate 2e-5 \
	--num_train_epochs 20.0 \
	--logging_steps 100 \
	--save_steps 100 \
	--overwrite_output_dir \
	--output_dir AD_results/

