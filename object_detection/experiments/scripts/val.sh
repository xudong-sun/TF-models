#! /bin/bash

set -x
set -e

GPU_ID=$1
PIPELINE=$2
TRAIN_DIR=$3
LOG_DIR=$4

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

python eval.py \
  --logtostderr \
  --pipeline_config_path=$PIPELINE \
  --checkpoint_dir=$TRAIN_DIR \
  --eval_dir=$LOG_DIR

