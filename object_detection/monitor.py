#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Monitoring TF TRAIN_DIR
# It moves latest checkpoints to a new directory on a specified interval
# To run, go to the "object_detection" directory, then run "python monitor.py"
# This will monitor all directories in "TRAIN_DIR"
# You may want to keep this script to run in the background: nohup python monitor.py >> ckpt/monitor/monitor.log &
#
# To override the default SAVE_EVERY, go to a specific directory and modify the number kept in ".moniter" file

import os
import re
import cPickle
import shutil
import logging
import time
import yaml
from collections import defaultdict
from commons import assure_dir, simple_logger

TRAIN_DIR = 'ckpt/train'
SAVE_DIR = 'ckpt/save'

WORK_DIR = 'ckpt/monitor'

PATTERN = r'model_checkpoint_path: "model.ckpt-(\d+)"'
LATEST_CKPT_PATTERN = re.compile(PATTERN)

logger = simple_logger()

def read_config():
    config_file = os.path.join(WORK_DIR, 'monitor.pkl')
    if os.path.isfile(config_file):
        with open(config_file) as f: config = yaml.load(f)
    else: config = {}
    return config
config = read_config()

def list_train_dir():
    dirs = [os.path.join(TRAIN_DIR, d) for d in os.listdir(TRAIN_DIR)]
    return [d for d in dirs if os.path.isdir(d) and os.path.isfile(os.path.join(d, 'checkpoint'))]

def latest_checkpoint(folder):
    '''return the step of the latest checkpoint
    '''
    with open(os.path.join(folder, 'checkpoint')) as f: lines = [s.strip() for s in f.readlines()]
    for line in lines:
        matcher = re.match(LATEST_CKPT_PATTERN, line)
        if matcher:
            return int(matcher.group(1))
    return 0

def get_target(targets, folder):
    '''read target steps from "targets"
    if .moniter file does not exist, this means that "folder" is a newly created directory, so return 0
    .monitor file also keeps a number specifying save_every
    '''
    moniter_file = os.path.join(folder, '.moniter')
    if os.path.isfile(moniter_file):
        with open(moniter_file) as f: save_every = int(f.read())
        retval = targets[folder], save_every
    else:
        logger.info('Found new train directory: {}'.format(folder))
        with open(moniter_file, 'w') as f: f.write(str(config['default_save_interval']]))
        retval = 0, config['default_save_interval']
    return retval

if __name__ == '__main__':
    if os.path.isfile(config['monitor_cache_file']):
        with open(config['monitor_cache_file'], 'rb') as f: targets = cPickle.load(f)
    else:
        targets = defaultdict(int)
    while True:
        dump_new_cache = False
        for d in list_train_dir():
            latest = latest_checkpoint(d)
            target, save_every = get_target(targets, d)
            if latest > target:
                copy_to = os.path.join(SAVE_DIR, os.path.basename(d))
                assure_dir(copy_to)
                files_to_copy = [f for f in os.listdir(d) if f.startswith('model.ckpt-{}.'.format(latest))]
                for f in files_to_copy:
                    shutil.copy2(os.path.join(d, f), copy_to)
                logger.info('Copied {}'.format(os.path.join(d, 'model.ckpt-{}'.format(latest))))
                while target < latest: target += save_every
                targets[d] = target
                dump_new_cache = True
        if dump_new_cache:
            with open(config['monitor_cache_file'], 'wb') as f: cPickle.dump(targets, f)
        time.sleep(config['check_interval'])
    
