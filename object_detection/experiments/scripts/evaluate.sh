#!/bin/bash
# 1. generate frozen inference graph from checkpoint
# 2. run detection on FDDB fold07
# 3. calculate FDDB ROC curves
# 4. run detection on AFW
# 5. calculate AFW ROC and PR curves
# Example: experiments/scripts/evaluate.sh 0 ssd_mobilenet_v1_wider 251616 ssd_mobilenet_v1_fddb.config

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

# evaluate FDDB
FDDB_REPORT_PREFIX=$FDDB_WORK_DIR/${CKPT_STEP}_
$FDDB_BASE/evaluation/evaluate \
 -a $FDDB_BASE/FDDB-folds/FDDB-fold-07-ellipseList.txt \
 -d $FDDB_RESULT \
 -f 0 \
 -i $FDDB_BASE/originalPics/ \
 -l $FDDB_BASE/FDDB-folds/FDDB-fold-07.txt \
 -r $FDDB_REPORT_PREFIX \
 -t -1

# write AFW txt
AFW_BASE=data/datasets/AFW
AFW_WORK_DIR=ckpt/afw/$FOLDER
AFW_RESULT=$AFW_WORK_DIR/result-$CKPT_STEP.txt
python test.py \
 --task_type afw \
 --ckpt_path $INFERENCE_GRAPH \
 --conf_thresh 0.1 \
 --afw_root $AFW_BASE \
 --afw_output $AFW_RESULT

# evaluate AFW
AFW_REPORT_PREFIX=$AFW_WORK_DIR/${CKPT_STEP}_
python $AFW_BASE/evaluation/evaluate.py \
 $AFW_RESULT \
 $AFW_BASE/annotations.txt \
 $AFW_REPORT_PREFIX

