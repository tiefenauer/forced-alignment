import builtins
import logging
import os
import sys
from datetime import datetime
from os import makedirs

import pygit2
from os.path import exists

FORMAT_STR = '%(asctime)s : %(name)s : %(levelname)s : %(message)s'
repo = pygit2.Repository(os.path.join(os.getcwd(), os.pardir))


def log_setup(filename=None):
    if filename:
        logger = logging.basicConfig(filename=filename, format=FORMAT_STR, level=logging.INFO)
    else:
        logger = logging.basicConfig(stream=sys.stdout, format=FORMAT_STR, level=logging.INFO)
    return logger


def print_to_file_and_console(log_dir):
    if not exists(log_dir):
        makedirs(log_dir)
    log_file = open(os.path.join(log_dir, 'train.log'), 'w')
    _print = builtins.print

    def my_print(args):
        _print(args)
        _print(args, file=log_file)

    builtins.print = my_print


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
