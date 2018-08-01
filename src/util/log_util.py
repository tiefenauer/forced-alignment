"""
Utility functions for logging tasks
"""
import logging
import sys
from datetime import datetime
from os import makedirs
from os.path import exists, dirname

import pygit2
import streamtologger

from constants import ROOT_DIR

FORMAT_STR = '%(asctime)s : %(name)s : %(levelname)s : %(message)s'
repo = pygit2.Repository(ROOT_DIR)

stdout = sys.stdout
stderr = sys.stderr


def log_setup(filename=None):
    if filename:
        logger = logging.basicConfig(filename=filename, format=FORMAT_STR, level=logging.INFO)
    else:
        logger = logging.basicConfig(stream=sys.stdout, format=FORMAT_STR, level=logging.INFO)
    return logger


def redirect_to_file(log_file_path, append=False, format="[{timestamp:%Y-%m-%d %H:%M:%S} - {level:5}] "):
    """
    Print to log file and stdout simultaneously. Use reset_redirect() to only print to stdout again
    """
    if not exists(dirname(log_file_path)):
        makedirs(dirname(log_file_path))
    streamtologger.redirect(log_file_path, append=append, header_format=format)


def reset_redirect():
    """
    Disable printing to log file and stdout simultaneously
    """
    global stdout
    global stderr
    sys.stdout = stdout
    sys.stderr = stderr
    streamtologger._is_redirected = False


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


def create_args_str(args, keys=None):
    return ', '.join(f'{key}={value}' for key, value in args.__dict__.items() if keys == None or key in keys)
