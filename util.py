import logging
import sys

FORMAT_STR = '%(asctime)s : %(name)s : %(levelname)s : %(message)s'


def log_setup(filename=None):
    if filename:
        logger = logging.basicConfig(filename=filename, format=FORMAT_STR, level=logging.INFO)
    else:
        logger = logging.basicConfig(stream=sys.stdout, format=FORMAT_STR, level=logging.INFO)
    return logger
