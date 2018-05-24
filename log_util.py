import logging
import subprocess
import sys

FORMAT_STR = '%(asctime)s : %(name)s : %(levelname)s : %(message)s'


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


def get_commit_id():
    git_commit = subprocess.check_output(["git", "describe", '--always']).strip()
    return git_commit


def create_args_str(args):
    return ', '.join(f'{key}={value}' for key, value in args.__dict__.items())
