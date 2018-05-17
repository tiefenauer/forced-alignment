import logging
import sys

FORMAT_STR = '%(asctime)s : %(name)s : %(levelname)s : %(message)s'


def log_setup(filename=None):
    if filename:
        logger = logging.basicConfig(filename=filename, format=FORMAT_STR, level=logging.INFO)
    else:
        logger = logging.basicConfig(stream=sys.stdout, format=FORMAT_STR, level=logging.INFO)
    return logger


def log_prediction(original_txt, rnn_txt, prediction_txt, subset_name):
    print(f'Original text ({subset_name}): '.ljust(30) + original_txt)
    print(f'RNN labels ({subset_name}): '.ljust(30) + rnn_txt)
    print(f'RNN prediction ({subset_name}): '.ljust(30) + prediction_txt)


def get_commit_id():
    git_commit = subprocess.check_output(["git", "describe", '--always'], cwd='..').strip()
    return git_commit
