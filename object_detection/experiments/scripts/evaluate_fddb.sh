#!/bin/bash
# 1. generate frozen inference graph from checkpoint
# 2. run detection on FDDB fold07
# 3. calculate ROC curves
# experiments/scripts/evaluate_fddb.sh 0 ssd_mobilenet_v1_wider 251616 ssd_mobilenet_v1_fddb.config

set -e

GPU_ID=$1
FOLDER=$2
CKPT_STEP=$3
PIPELINE_FILE=$4

export CUDA_VISIBLE_DEVICES=$1

# export inference graph
INFERENCE_GRAPH=ckpt/save/$FOLDER/frozen_inference_graph-$CKPT_STEP.pb
python export_inference_graph.py \
 --pipeline_config_path models/pipelines/$PIPELINE_FILE \
 --checkpoint_path ckpt/save/$FOLDER/model.ckpt-$CKPT_STEP \
 --inference_graph_path $INFERENCE_GRAPH

# write FDDB txt
FDDB_BASE=data/datasets/FDDB
FDDB_WORK_DIR=ckpt/fddb/$FOLDER
FDDB_RESULT=$FDDB_WORK_DIR/result-$CKPT_STEP.txt
python test.py \
 --task_type fddb \
 --ckpt_path $INFERENCE_GRAPH \
 --conf_thresh 0.01 \
 --fddb_root $FDDB_BASE \
 --fddb_output $FDDB_RESULT

# evaluate
ROC_PREFIX=$FDDB_WORK_DIR/${CKPT_STEP}_
$FDDB_BASE/evaluation/evaluate \
 -a $FDDB_BASE/FDDB-folds/FDDB-fold-07-ellipseList.txt \
 -d $FDDB_RESULT \
 -f 0 \
 -i $FDDB_BASE/originalPics/ \
 -l $FDDB_BASE/FDDB-folds/FDDB-fold-07.txt \
 -r $ROC_PREFIX \
 -t -1

