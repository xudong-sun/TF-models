#! /bin/bash

set -x
set -e

GPU_ID=$1
PIPELINE=$2
LOG_DIR=$3

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

python train.py \
  --logtostderr \
  --pipeline_config_path=$PIPELINE \
  --train_dir=$LOG_DIR

