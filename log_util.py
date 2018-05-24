import logging
import os
import subprocess
import sys
from datetime import datetime

import pygit2

FORMAT_STR = '%(asctime)s : %(name)s : %(levelname)s : %(message)s'
repo = pygit2.Repository(os.getcwd())


def log_setup(filename=None):
    if filename:
        logger = logging.basicConfig(filename=filename, format=FORMAT_STR, level=logging.INFO)
    else:
        logger = logging.basicConfig(stream=sys.stdout, format=FORMAT_STR, level=logging.INFO)
    return logger


def log_prediction(logger, ground_truth, prediction, subset_name):
    logger.write(f'Ground truth ({subset_name}): '.ljust(30) + ground_truth)
    logger.write(f'Prediction ({subset_name}): '.ljust(30) + prediction)


def print_prediction(ground_truth, prediction, subset_name):
    print(f'Ground truth ({subset_name}): '.ljust(30) + ground_truth)
    print(f'Prediction ({subset_name}): '.ljust(30) + prediction)


def get_commit():
    branch_name = repo.head.name
    commit = repo.head.peel()
    revision = commit.id
    timestamp = datetime.fromtimestamp(commit.commit_time).strftime('%Y-%m-%d %H:%M:%S')
    message = commit.message
    return revision, branch_name, timestamp, message


def create_args_str(args):
    return ', '.join(f'{key}={value}' for key, value in args.__dict__.items())
