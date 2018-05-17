import logging
import sys

FORMAT_STR = '%(asctime)s : %(name)s : %(levelname)s : %(message)s'


def log_setup(filename=None):
    if filename:
        logger = logging.basicConfig(filename=filename, format=FORMAT_STR, level=logging.INFO)
    else:
        logger = logging.basicConfig(stream=sys.stdout, format=FORMAT_STR, level=logging.INFO)
    return logger


def log_prediction(ground_truth, prediction, subset_name):
    print(f'Ground truth ({subset_name}): '.ljust(30) + ground_truth)
    print(f'Prediction ({subset_name}): '.ljust(30) + prediction)


def get_commit_id():
    git_commit = subprocess.check_output(["git", "describe", '--always'], cwd='..').strip()
    return git_commit
