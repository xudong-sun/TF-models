#!/bin/bash
FDDB_BASE=data/datasets/FDDB
FOLDER=wider2
RESULT_FILE=FDDB_result_7.txt

$FDDB_BASE/evaluation/evaluate \
 -a $FDDB_BASE/FDDB-folds/FDDB-fold-07-ellipseList.txt \
 -d ckpt/train/$FOLDER/$RESULT_FILE \
 -f 0 \
 -i $FDDB_BASE/originalPics/ \
 -l $FDDB_BASE/FDDB-folds/FDDB-fold-07.txt \
 -r ckpt/train/$FOLDER/result_7_ \
 -t -1
